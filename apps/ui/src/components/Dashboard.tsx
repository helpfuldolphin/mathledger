"use client";

import {
  fetchHeartbeat,
  fetchLatestAttestation,
  fetchRecentStatements,
  fetchStatementDetail,
  postUiEvent,
  type AttestationSummary,
  type Heartbeat,
  type StatementDetail,
  type StatementSummary,
} from "@/lib/api";
import {
  assertAttestationInvariants,
  truncateHash,
} from "@/lib/attestation-invariants";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type CytoscapeModule = typeof import("cytoscape");
type CytoscapeCore = import("cytoscape").Core;

interface DashboardProps {
  initialHeartbeat: Heartbeat | null;
  initialStatements: StatementSummary[];
  initialDetail: StatementDetail | null;
}

const nf = new Intl.NumberFormat("en-US");
const throughputFormat = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});
const timestampFormat = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  year: "numeric",
  month: "short",
  day: "2-digit",
});

function truncateLabel(value: string, limit = 88): string {
  const trimmed = value.trim();
  if (trimmed.length <= limit) {
    return trimmed;
  }
  return `${trimmed.slice(0, limit - 1)}…`;
}

/**
 * Truncate a hash for display.
 * For attestation roots (R_t, U_t, H_t), use prefix=12, suffix=8 for consistency.
 */
function shortHash(hash: string, prefix = 6, suffix = 4): string {
  return truncateHash(hash, prefix, suffix);
}

function successBadge(success: boolean | null | undefined): string {
  if (success === true) return "bg-emerald-500/15 text-emerald-300 border border-emerald-500/40";
  if (success === false) return "bg-rose-500/15 text-rose-300 border border-rose-500/40";
  return "bg-slate-700/60 text-slate-300 border border-slate-600/60";
}

export default function Dashboard({
  initialHeartbeat,
  initialStatements,
  initialDetail,
}: DashboardProps) {
  const [heartbeat, setHeartbeat] = useState<Heartbeat | null>(initialHeartbeat);
  const [heartbeatError, setHeartbeatError] = useState<string | null>(null);
  const [statements, setStatements] =
    useState<StatementSummary[]>(initialStatements);
  const [selectedHash, setSelectedHash] = useState<string | null>(() => {
    if (initialDetail?.hash) return initialDetail.hash;
    if (initialStatements.length > 0) return initialStatements[0].hash;
    return null;
  });
  const [listLoading, setListLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [attestation, setAttestation] = useState<AttestationSummary | null>(null);
  const [attestationError, setAttestationError] = useState<string | null>(null);

  const detailStore = useRef<Map<string, StatementDetail>>(new Map());
  const [detailVersion, setDetailVersion] = useState(0);

  useEffect(() => {
    if (initialDetail && !detailStore.current.has(initialDetail.hash)) {
      detailStore.current.set(initialDetail.hash, initialDetail);
      setDetailVersion((v) => v + 1);
    }
  }, [initialDetail]);

  const refreshHeartbeat = useCallback(async () => {
    try {
      const next = await fetchHeartbeat();
      setHeartbeat(next);
      setHeartbeatError(null);
    } catch (err) {
      setHeartbeatError(
        err instanceof Error ? err.message : "Unable to refresh heartbeat"
      );
    }
  }, []);

  const refreshAttestation = useCallback(async () => {
    try {
      const summary = await fetchLatestAttestation();
      setAttestation(summary);
      setAttestationError(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to load attestation";
      if (message.includes("404")) {
        setAttestation(null);
        setAttestationError(null);
        return;
      }
      setAttestationError(message);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const interval = setInterval(() => {
      refreshHeartbeat().catch(() => {
        /* handled */
      });
    }, 7500);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [refreshHeartbeat]);

  useEffect(() => {
    refreshAttestation().catch(() => undefined);
    const interval = setInterval(() => {
      refreshAttestation().catch(() => undefined);
    }, 10000);
    return () => {
      clearInterval(interval);
    };
  }, [refreshAttestation]);

  useEffect(() => {
    postUiEvent({
      event_type: "dashboard_mount",
      timestamp: Date.now(),
    }).catch(() => undefined);
  }, []);

  const refreshStatements = useCallback(async () => {
    setListLoading(true);
    try {
      const latest = await fetchRecentStatements(32);
      setStatements(latest);
      postUiEvent({
        event_type: "refresh_statements",
        timestamp: Date.now(),
        statement_count: latest.length,
      }).catch(() => undefined);
    } catch (err) {
      setDetailError(
        err instanceof Error ? err.message : "Unable to refresh statements"
      );
    } finally {
      setListLoading(false);
    }
  }, []);

  const ensureDetail = useCallback(async (hash: string) => {
    const existing = detailStore.current.get(hash);
    if (existing) {
      return existing;
    }
    setDetailLoading(true);
    setDetailError(null);
    try {
      const detail = await fetchStatementDetail(hash);
      detailStore.current.set(hash, detail);
      setDetailVersion((v) => v + 1);
      setStatements((prev) => {
        if (prev.some((item) => item.hash === hash)) {
          return prev;
        }
        const summary: StatementSummary = {
          hash,
          display: detail.display,
          text: detail.text,
          normalizedText: detail.normalizedText,
        };
        return [summary, ...prev].slice(0, 64);
      });
      return detail;
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to load statement detail";
      setDetailError(message);
      throw err;
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const handleSelect = useCallback(
    (hash: string) => {
      setSelectedHash(hash);
      void ensureDetail(hash);
      postUiEvent({
        event_type: "select_statement",
        timestamp: Date.now(),
        statement_hash: hash,
      }).catch(() => undefined);
    },
    [ensureDetail]
  );

  useEffect(() => {
    if (selectedHash) {
      void ensureDetail(selectedHash);
    }
  }, [selectedHash, ensureDetail]);

  const currentDetail = useMemo<StatementDetail | null>(() => {
    if (!selectedHash) return null;
    return detailStore.current.get(selectedHash) ?? null;
  }, [selectedHash, detailVersion]);

  const proofs = currentDetail?.proofs ?? [];
  const parents = currentDetail?.parents ?? [];

  const metrics = useMemo(() => {
    if (!heartbeat) {
      return null;
    }
    const success = heartbeat.proofs?.success ?? 0;
    const failure = heartbeat.proofs?.failure ?? null;
    const merkle = heartbeat.blocks?.latestMerkle ?? null;
    const policy = heartbeat.policyHash ?? null;
    return {
      success,
      failure,
      throughput: heartbeat.proofsPerHour,
      blockHeight: heartbeat.blocks?.height ?? 0,
      merkle,
      policy,
      queue: heartbeat.redisQueueLength,
    };
  }, [heartbeat]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-blue-400">
                MathLedger
              </p>
              <h1 className="text-2xl font-semibold text-slate-100">
                Proof Graph Console
              </h1>
            </div>
            <span className="rounded border border-blue-500/40 bg-blue-500/10 px-2 py-1 text-xs font-semibold text-blue-300">
              Deterministic • Attested • Canonical
            </span>
          </div>
          <button
            type="button"
            onClick={() => refreshHeartbeat().catch(() => undefined)}
            className="rounded border border-slate-700 bg-slate-800/70 px-3 py-1.5 text-sm text-slate-200 transition hover:border-blue-500/60 hover:bg-slate-800"
          >
            Refresh Heartbeat
          </button>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 py-8">
        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            title="Proofs (success)"
            value={metrics ? nf.format(metrics.success) : "—"}
            hint={
              metrics?.failure != null
                ? `${nf.format(metrics.failure)} pending/failing`
                : "Awaiting data"
            }
          />
          <MetricCard
            title="Throughput (per hour)"
            value={
              metrics ? throughputFormat.format(metrics.throughput) : "—"
            }
            hint={heartbeatError ?? "Window: 5 min"}
          />
          <MetricCard
            title="Block Height"
            value={metrics ? nf.format(metrics.blockHeight) : "—"}
            hint={
              metrics?.merkle
                ? shortHash(metrics.merkle, 10, 6)
                : "Merkle pending"
            }
          />
          <MetricCard
            title="Policy Hash"
            value={
              metrics?.policy ? shortHash(metrics.policy, 10, 6) : "Unpinned"
            }
            hint={
              metrics?.queue != null
                ? `Queue depth: ${metrics.queue >= 0 ? metrics.queue : "n/a"}`
                : "Queue depth unavailable"
            }
          />
        </section>

        <AttestationPanel
          attestation={attestation}
          error={attestationError}
          onRefresh={refreshAttestation}
        />

        <section className="grid gap-6 lg:grid-cols-[340px_1fr]">
          <StatementList
            statements={statements}
            onRefresh={refreshStatements}
            onSelect={handleSelect}
            selectedHash={selectedHash}
            isRefreshing={listLoading}
          />

          <div className="flex flex-col gap-6">
            <StatementDetailPanel
              detail={currentDetail}
              isLoading={detailLoading}
              error={detailError}
              onSelect={handleSelect}
            />
            <GraphCanvas detail={currentDetail} onSelect={handleSelect} />
          </div>
        </section>
      </main>

      <footer className="border-t border-slate-800/80 bg-slate-950/90">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 py-4 text-xs text-slate-500">
          <span>© {new Date().getFullYear()} MathLedger</span>
          <span className="font-mono text-slate-400">
            UI integrity: {heartbeat?.ok ? "stable" : "degraded"}
          </span>
        </div>
      </footer>
    </div>
  );
}

interface MetricCardProps {
  title: string;
  value: string;
  hint?: string | null;
}

function AttestationPanel({ attestation, error, onRefresh }: AttestationPanelProps) {
  // Validate attestation invariants on every render (dev mode only logs)
  useEffect(() => {
    assertAttestationInvariants(attestation, "AttestationPanel");
  }, [attestation]);

  const metadata = attestation?.metadata ?? {};
  const reasoningCount =
    typeof metadata["reasoning_event_count"] === "number"
      ? Number(metadata["reasoning_event_count"])
      : attestation?.reasoningLeaves.length ?? 0;
  const uiCount =
    typeof metadata["ui_event_count"] === "number"
      ? Number(metadata["ui_event_count"])
      : attestation?.uiLeaves.length ?? 0;
  const version =
    typeof metadata["attestation_version"] === "string"
      ? String(metadata["attestation_version"])
      : null;

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">
            Dual Attestation
          </h2>
          <p className="text-xs text-slate-500">
            Cryptographic binding of reasoning and UI event streams
          </p>
        </div>
        <div className="flex items-center gap-2">
          {version && (
            <span className="rounded border border-blue-500/40 bg-blue-500/10 px-2 py-0.5 text-[11px] font-semibold text-blue-300">
              {version}
            </span>
          )}
          <button
            type="button"
            onClick={() => onRefresh().catch(() => undefined)}
            className="rounded border border-slate-700 bg-slate-800/70 px-3 py-1 text-xs text-slate-200 transition hover:border-blue-500/60 hover:text-blue-200"
          >
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
          {error}
        </div>
      )}

      {!attestation && !error && (
        <div className="mt-4 rounded border border-slate-800 bg-slate-900 px-4 py-5 text-xs text-slate-500">
          No attestation has been published yet.
        </div>
      )}

      {attestation && (
        <>
          <div className="mt-4 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <AttestationField
              label="Block"
              value={
                attestation.blockNumber != null
                  ? `#${nf.format(attestation.blockNumber)}`
                  : "—"
              }
              hint={
                attestation.blockHash
                  ? shortHash(attestation.blockHash, 10, 6)
                  : "hash pending"
              }
            />
            <AttestationField
              label="Reasoning Root (R_t)"
              value={shortHash(attestation.reasoningMerkleRoot, 12, 8)}
              hint={`${reasoningCount} leaves`}
              titleValue={attestation.reasoningMerkleRoot}
            />
            <AttestationField
              label="UI Root (U_t)"
              value={shortHash(attestation.uiMerkleRoot, 12, 8)}
              hint={`${uiCount} leaves`}
              titleValue={attestation.uiMerkleRoot}
            />
            <AttestationField
              label="Composite Root (H_t)"
              value={shortHash(attestation.compositeAttestationRoot, 12, 8)}
              hint="SHA256(R_t || U_t)"
              titleValue={attestation.compositeAttestationRoot}
            />
          </div>

          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            {attestation.reasoningLeaves.length > 0 && (
              <div className="rounded border border-slate-800 bg-slate-900/70 px-4 py-3">
                <p className="text-xs uppercase tracking-wide text-slate-400">
                  Reasoning Leaf Sample
                </p>
                <p className="mt-2 font-mono text-[11px] text-slate-300 break-words">
                  {attestation.reasoningLeaves[0].canonicalValue}
                </p>
                <p className="mt-2 text-[11px] text-slate-500">
                  Hash: {attestation.reasoningLeaves[0].leafHash}
                </p>
              </div>
            )}
            {attestation.uiLeaves.length > 0 && (
              <div className="rounded border border-slate-800 bg-slate-900/70 px-4 py-3">
                <p className="text-xs uppercase tracking-wide text-slate-400">
                  UI Leaf Sample
                </p>
                <p className="mt-2 font-mono text-[11px] text-slate-300 break-words">
                  {attestation.uiLeaves[0].canonicalValue}
                </p>
                <p className="mt-2 text-[11px] text-slate-500">
                  Hash: {attestation.uiLeaves[0].leafHash}
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </section>
  );
}

interface AttestationPanelProps {
  attestation: AttestationSummary | null;
  error: string | null;
  onRefresh(): Promise<void>;
}

interface AttestationFieldProps {
  label: string;
  value: string;
  hint?: string | null;
  titleValue?: string | null;
}

function AttestationField({
  label,
  value,
  hint,
  titleValue,
}: AttestationFieldProps) {
  return (
    <div className="rounded border border-slate-800 bg-slate-950/60 p-4">
      <p className="text-[11px] uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p
        className="mt-2 text-sm font-semibold text-slate-100"
        title={titleValue ?? undefined}
      >
        {value}
      </p>
      {hint && <p className="mt-1 text-[11px] text-slate-500">{hint}</p>}
    </div>
  );
}

function MetricCard({ title, value, hint }: MetricCardProps) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 shadow-inner shadow-black/20">
      <p className="text-xs uppercase tracking-wide text-slate-400">{title}</p>
      <p className="mt-2 text-2xl font-semibold text-slate-100">{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}

interface StatementListProps {
  statements: StatementSummary[];
  selectedHash: string | null;
  isRefreshing: boolean;
  onSelect(hash: string): void;
  onRefresh(): void;
}

function StatementList({
  statements,
  selectedHash,
  isRefreshing,
  onSelect,
  onRefresh,
}: StatementListProps) {
  return (
    <div className="flex h-full flex-col rounded-xl border border-slate-800 bg-slate-900/60">
      <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">
            Statements
          </h2>
          <p className="text-xs text-slate-500">
            Most recent canonical hashes
          </p>
        </div>
        <button
          type="button"
          onClick={() => onRefresh()}
          className="rounded border border-slate-700 px-3 py-1 text-xs text-slate-300 transition hover:border-blue-500/60 hover:text-blue-300"
        >
          {isRefreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {statements.length === 0 ? (
          <div className="px-5 py-6 text-sm text-slate-500">
            No statements available yet.
          </div>
        ) : (
          <ul className="divide-y divide-slate-800">
            {statements.map((statement) => {
              const isSelected = statement.hash === selectedHash;
              return (
                <li key={statement.hash}>
                  <button
                    type="button"
                    onClick={() => onSelect(statement.hash)}
                    className={`flex w-full flex-col gap-1 px-5 py-4 text-left transition ${
                      isSelected
                        ? "bg-blue-500/10 text-slate-100"
                        : "hover:bg-slate-800/60"
                    }`}
                  >
                    <span className="text-xs font-mono text-slate-400">
                      {statement.hash}
                    </span>
                    <span className="text-sm text-slate-100">
                      {truncateLabel(statement.display)}
                    </span>
                    {statement.normalizedText && (
                      <span className="text-xs text-slate-500">
                        {truncateLabel(statement.normalizedText, 72)}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

interface StatementDetailPanelProps {
  detail: StatementDetail | null;
  isLoading: boolean;
  error: string | null;
  onSelect(hash: string): void;
}

function StatementDetailPanel({
  detail,
  isLoading,
  error,
  onSelect,
}: StatementDetailPanelProps) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">
            Statement Detail
          </h2>
          <p className="text-xs text-slate-500">
            Canonical text, proofs, and dependencies
          </p>
        </div>
        {isLoading && (
          <span className="text-xs text-blue-300">Loading…</span>
        )}
      </div>

      {error && (
        <div className="mt-4 rounded border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
          {error}
        </div>
      )}

      {!detail && !isLoading && !error && (
        <div className="mt-6 rounded border border-slate-800 bg-slate-900 px-4 py-8 text-sm text-slate-500">
          Select a statement to inspect its detail.
        </div>
      )}

      {detail && (
        <div className="mt-5 space-y-6">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-400">
              Hash
            </p>
            <p className="font-mono text-sm text-blue-200">
              {detail.hash}
            </p>
          </div>

          <div>
            <p className="text-xs uppercase tracking-wide text-slate-400">
              Display
            </p>
            <p className="mt-1 text-base text-slate-100">{detail.display}</p>
            {detail.normalizedText && (
              <p className="mt-2 rounded border border-slate-800 bg-slate-900 px-3 py-2 text-xs text-slate-300">
                {detail.normalizedText}
              </p>
            )}
          </div>

          <div>
            <p className="mb-2 text-xs uppercase tracking-wide text-slate-400">
              Parents
            </p>
            {detail.parents.length === 0 ? (
              <p className="text-xs text-slate-500">No recorded parents.</p>
            ) : (
              <div className="grid gap-2 sm:grid-cols-2">
                {detail.parents.map((parent) => (
                  <button
                    key={parent.hash}
                    onClick={() => onSelect(parent.hash)}
                    className="rounded border border-slate-800 bg-slate-900 px-3 py-2 text-left transition hover:border-blue-500/60 hover:bg-slate-800"
                  >
                    <p className="text-[11px] font-mono text-slate-500">
                      {parent.hash}
                    </p>
                    <p className="mt-1 text-xs text-slate-200">
                      {parent.display
                        ? truncateLabel(parent.display, 80)
                        : "—"}
                    </p>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div>
            <p className="mb-2 text-xs uppercase tracking-wide text-slate-400">
              Proof Attempts
            </p>
            {detail.proofs.length === 0 ? (
              <p className="text-xs text-slate-500">
                No proofs have been recorded for this statement.
              </p>
            ) : (
              <div className="overflow-x-auto rounded border border-slate-800">
                <table className="min-w-full divide-y divide-slate-800 text-left text-xs text-slate-300">
                  <thead className="bg-slate-900/70 text-[11px] uppercase tracking-wider text-slate-400">
                    <tr>
                      <th className="px-3 py-2">Method</th>
                      <th className="px-3 py-2">Status</th>
                      <th className="px-3 py-2">Outcome</th>
                      <th className="px-3 py-2">Prover</th>
                      <th className="px-3 py-2">Duration</th>
                      <th className="px-3 py-2">Created</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-900/70">
                    {detail.proofs.map((proof, idx) => (
                      <tr key={`${proof.method ?? "proof"}-${idx}`}>
                        <td className="px-3 py-2 font-mono text-[11px] text-blue-200">
                          {proof.method ?? "—"}
                        </td>
                        <td className="px-3 py-2">{proof.status ?? "—"}</td>
                        <td className="px-3 py-2">
                          <span
                            className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] ${successBadge(
                              proof.success ?? null
                            )}`}
                          >
                            {proof.success === true && "✓ success"}
                            {proof.success === false && "× failed"}
                            {proof.success == null && "—"}
                          </span>
                        </td>
                        <td className="px-3 py-2">{proof.prover ?? "—"}</td>
                        <td className="px-3 py-2">
                          {proof.durationMs != null
                            ? `${nf.format(proof.durationMs)} ms`
                            : "—"}
                        </td>
                        <td className="px-3 py-2">
                          {proof.createdAt
                            ? timestampFormat.format(proof.createdAt)
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

interface GraphCanvasProps {
  detail: StatementDetail | null;
  onSelect(hash: string): void;
}

function GraphCanvas({ detail, onSelect }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<CytoscapeCore | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function init() {
      if (!containerRef.current || cyRef.current) {
        return;
      }
      const cytoscapeModule = await import("cytoscape");
      if (!isMounted || !containerRef.current) {
        return;
      }
      const cytoscape =
        (cytoscapeModule.default ??
          (cytoscapeModule as unknown)) as CytoscapeModule["default"];
      const cy = cytoscape({
        container: containerRef.current,
        wheelSensitivity: 0.2,
        minZoom: 0.25,
        maxZoom: 2.5,
        style: [
          {
            selector: "node",
            style: {
              "background-color": "#2563eb",
              "border-color": "#60a5fa",
              "border-width": 1,
              label: "data(label)",
              color: "#f8fafc",
              "font-size": 11,
              "font-family":
                "var(--font-geist-sans), ui-sans-serif, system-ui",
              "text-wrap": "wrap",
              "text-max-width": "180px",
              "text-valign": "center",
              "text-halign": "center",
              "padding-left": "8px",
              "padding-right": "8px",
              "padding-top": "6px",
              "padding-bottom": "6px",
              shape: "roundrectangle",
            },
          },
          {
            selector: "node[type = 'parent']",
            style: {
              "background-color": "#0ea5e9",
              "border-color": "#38bdf8",
            },
          },
          {
            selector: "node[type = 'focus']",
            style: {
              "background-color": "#f97316",
              "border-color": "#fb923c",
            },
          },
          {
            selector: "node:selected",
            style: {
              "border-width": 2,
              "border-color": "#fbbf24",
            },
          },
          {
            selector: "edge",
            style: {
              width: 2,
              "line-color": "#94a3b8",
              "target-arrow-color": "#94a3b8",
              "target-arrow-shape": "triangle",
              "curve-style": "straight",
            },
          },
        ],
      });
      cy.on("tap", "node", (event) => {
        const id = event.target.id();
        if (id) {
          onSelect(id);
        }
      });
      cyRef.current = cy;
    }

    init().catch(() => undefined);

    const handleResize = () => {
      cyRef.current?.resize();
      cyRef.current?.fit(undefined, 24);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      isMounted = false;
      window.removeEventListener("resize", handleResize);
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [onSelect]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) {
      return;
    }
    cy.elements().remove();
    if (!detail) {
      return;
    }

    const focusNode = {
      group: "nodes" as const,
      data: {
        id: detail.hash,
        label: truncateLabel(detail.display, 96),
        type: "focus",
      },
    };

    const parentNodes = detail.parents.map((parent) => ({
      group: "nodes" as const,
      data: {
        id: parent.hash,
        label: truncateLabel(parent.display ?? parent.hash, 84),
        type: "parent",
      },
    }));

    const edges = detail.parents.map((parent) => ({
      group: "edges" as const,
      data: {
        id: `${parent.hash}->${detail.hash}`,
        source: parent.hash,
        target: detail.hash,
      },
    }));

    const nodes = [focusNode, ...parentNodes];

    cy.add([...nodes, ...edges]);

    cy.layout({
      name: "breadthfirst",
      directed: true,
      padding: 25,
      spacingFactor: 1.2,
    }).run();

    cy.fit(undefined, 40);
  }, [detail]);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60">
      <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">
            Dependency Graph
          </h2>
          <p className="text-xs text-slate-500">
            Tap nodes to pivot across the proof DAG
          </p>
        </div>
      </div>
      <div ref={containerRef} className="h-72 w-full" />
    </div>
  );
}

