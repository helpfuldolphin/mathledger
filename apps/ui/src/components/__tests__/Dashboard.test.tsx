import React from "react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import Dashboard from "@/components/Dashboard";
import attestationFixture from "@fixtures/first_organism_attestation.json";
import type { StatementDetail, StatementSummary, AttestationSummary } from "@/lib/api";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockHeartbeat = vi.fn();
const mockStatements = vi.fn();
const mockStatementDetail = vi.fn();
const mockAttestation = vi.fn();
const mockPostUiEvent = vi.fn();

vi.mock("@/lib/api", () => ({
  fetchHeartbeat: () => mockHeartbeat(),
  fetchRecentStatements: () => mockStatements(),
  fetchStatementDetail: (hash: string) => mockStatementDetail(hash),
  fetchLatestAttestation: () => mockAttestation(),
  postUiEvent: (payload: Record<string, unknown>) => mockPostUiEvent(payload),
}));

type CytoscapeMockInstance = {
  nodes: any[];
  edges: any[];
  add: (elements: unknown) => void;
  on: (..._args: unknown[]) => unknown;
  elements: () => { remove: () => void };
  layout: (_opts?: unknown) => { run: () => void };
  fit: () => void;
  resize: () => void;
  destroy: () => void;
};

const cytoscapeInstances: CytoscapeMockInstance[] = [];

vi.mock("cytoscape", () => {
  class CytoscapeMock implements CytoscapeMockInstance {
    nodes: any[] = [];
    edges: any[] = [];
    add(elements: any) {
      const arr = Array.isArray(elements) ? elements : [elements];
      arr.forEach((element) => {
        if (element?.group === "nodes") {
          this.nodes.push(element);
        } else if (element?.group === "edges") {
          this.edges.push(element);
        }
      });
    }
    on() {
      return this;
    }
    elements() {
      return { remove: () => undefined };
    }
    layout() {
      return { run: () => undefined };
    }
    fit() {
      return undefined;
    }
    resize() {
      return undefined;
    }
    destroy() {
      return undefined;
    }
  }

  return {
    default: (_options: unknown) => {
      const instance = new CytoscapeMock();
      cytoscapeInstances.push(instance);
      return instance;
    },
    __instances: cytoscapeInstances,
  };
});

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const heartbeatPayload = {
  ok: true,
  ts: new Date("2025-11-26T00:00:00Z"),
  proofs: { success: 4, failure: 1 },
  proofs_per_sec: 1,
  proofsPerHour: 3600,
  blocks: { height: 7, latestMerkle: "deadbeef" },
  policyHash: "policyhash",
  redisQueueLength: 2,
};

const statementHash =
  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

const statementSummary: StatementSummary = {
  hash: statementHash,
  display: "First organism proof",
  text: "∀x, x = x",
};

const parentHash = "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd";

const statementDetail: StatementDetail = {
  hash: statementHash,
  display: "First organism proof",
  text: "∀x, x = x",
  normalizedText: "forall x, x = x",
  proofs: [
    {
      method: "auto",
      status: "ABSTAIN",
      success: false,
      createdAt: new Date(),
      prover: "lean",
      durationMs: 123,
    },
  ],
  parents: [{ hash: parentHash, display: "Parent Statement" }],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function shortHash(hash: string, prefix = 12, suffix = 8): string {
  const trimmed = hash.trim();
  if (trimmed.length <= prefix + suffix + 1) {
    return trimmed;
  }
  return `${trimmed.slice(0, prefix)}…${trimmed.slice(-suffix)}`;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Dashboard attestation integration (virtual)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    cytoscapeInstances.length = 0;
    mockHeartbeat.mockResolvedValue(heartbeatPayload);
    mockStatements.mockResolvedValue([statementSummary]);
    mockStatementDetail.mockResolvedValue(statementDetail);
    mockAttestation.mockResolvedValue(attestationFixture as unknown as AttestationSummary);
    mockPostUiEvent.mockResolvedValue({
      event_id: "mock",
      timestamp: Date.now(),
      leaf_hash: "leaf",
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders R_t, U_t, H_t from fixture and matches exact truncated values", async () => {
    render(
      <Dashboard
        initialHeartbeat={null}
        initialStatements={[]}
        initialDetail={null}
      />
    );

    await screen.findByText("Proof Graph Console");

    await waitFor(() => {
      expect(screen.getByText("Dual Attestation")).toBeInTheDocument();
    });

    // Reasoning root (R_t)
    const rTruncated = shortHash(attestationFixture.reasoningMerkleRoot, 12, 8);
    expect(screen.getByText(rTruncated, { exact: false })).toBeInTheDocument();

    // UI root (U_t)
    const uTruncated = shortHash(attestationFixture.uiMerkleRoot, 12, 8);
    expect(screen.getByText(uTruncated, { exact: false })).toBeInTheDocument();

    // Composite root (H_t)
    const hTruncated = shortHash(attestationFixture.compositeAttestationRoot, 12, 8);
    expect(screen.getByText(hTruncated, { exact: false })).toBeInTheDocument();

    // Metadata version badge
    const version = attestationFixture.metadata.attestation_version;
    expect(screen.getByText(version, { exact: false })).toBeInTheDocument();
  });

  it("logs dashboard_mount on initial render", async () => {
    render(
      <Dashboard
        initialHeartbeat={null}
        initialStatements={[]}
        initialDetail={null}
      />
    );

    await waitFor(() => {
      expect(mockPostUiEvent).toHaveBeenCalledWith(
        expect.objectContaining({ event_type: "dashboard_mount" })
      );
    });
  });

  it("logs select_statement with correct payload on statement click", async () => {
    render(
      <Dashboard
        initialHeartbeat={null}
        initialStatements={[statementSummary]}
        initialDetail={null}
      />
    );

    const statementButton = await screen.findByText(statementSummary.hash);
    await userEvent.click(statementButton);

    expect(mockPostUiEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        event_type: "select_statement",
        statement_hash: statementSummary.hash,
      })
    );
  });

  it("logs refresh_statements with statement_count on refresh", async () => {
    render(
      <Dashboard
        initialHeartbeat={null}
        initialStatements={[]}
        initialDetail={null}
      />
    );

    // Wait for the Statements section to render
    await screen.findByText("Statements");

    // Find all buttons with "Refresh" text, then find the one in the Statements section
    const refreshBtns = await screen.findAllByRole("button", { name: /Refresh/i });
    // The statements section refresh button has text "Refresh" (not "Refresh Heartbeat")
    // and is inside the Statements section (the third button)
    const statementsRefreshBtn = refreshBtns.find(
      (btn) => btn.textContent === "Refresh" && btn.closest("div")?.textContent?.includes("Statements")
    );
    expect(statementsRefreshBtn).toBeDefined();
    await userEvent.click(statementsRefreshBtn!);

    await waitFor(() => {
      expect(mockPostUiEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          event_type: "refresh_statements",
          statement_count: 1,
        })
      );
    });
  });

  // NOTE: DAG rendering test is skipped because the GraphCanvas component uses
  // dynamic import (`await import("cytoscape")`) which is difficult to mock in JSDOM.
  // This should be tested via Cypress or Playwright in a real browser environment.
  it.skip("constructs DAG nodes matching statement and parent hashes", async () => {
    render(
      <Dashboard
        initialHeartbeat={null}
        initialStatements={[statementSummary]}
        initialDetail={statementDetail}
      />
    );

    // Wait for cytoscape to be initialized (async import)
    await waitFor(
      () => {
        expect(cytoscapeInstances.length).toBeGreaterThan(0);
      },
      { timeout: 3000 }
    );

    // Wait for the detail to be rendered and nodes to be added
    await waitFor(
      () => {
        const instance = cytoscapeInstances[0];
        expect(instance).toBeDefined();
        expect(instance.nodes.length).toBeGreaterThan(0);
      },
      { timeout: 3000 }
    );

    const instance = cytoscapeInstances[0];
    expect(instance.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          data: expect.objectContaining({ id: statementDetail.hash }),
        }),
        expect.objectContaining({
          data: expect.objectContaining({ id: parentHash }),
        }),
      ])
    );
  });

  it("postUiEvent payloads are backend-compatible (event_type, timestamp, statement_hash)", async () => {
    render(
      <Dashboard
        initialHeartbeat={null}
        initialStatements={[statementSummary]}
        initialDetail={null}
      />
    );

    const statementButton = await screen.findByText(statementSummary.hash);
    await userEvent.click(statementButton);

    const call = mockPostUiEvent.mock.calls.find(
      (c: [Record<string, unknown>]) => c[0]?.event_type === "select_statement"
    );
    expect(call).toBeDefined();
    const payload = call![0];

    // Required fields per backend/ledger/ui_events schema
    expect(typeof payload.event_type).toBe("string");
    expect(typeof payload.timestamp).toBe("number");
    expect(typeof payload.statement_hash).toBe("string");
  });
});
