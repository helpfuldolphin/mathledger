const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");

type FetcherOptions = Omit<RequestInit, "headers"> & {
  headers?: Record<string, string>;
};

async function apiGet<T>(
  path: string,
  { headers, ...init }: FetcherOptions = {}
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    cache: "no-store",
    ...init,
    headers: {
      Accept: "application/json",
      ...headers,
    },
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`GET ${url} failed: ${response.status} ${body}`);
  }

  return response.json() as Promise<T>;
}

async function apiPost<T>(
  path: string,
  body: unknown,
  { headers, ...init }: FetcherOptions = {}
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    method: "POST",
    cache: "no-store",
    ...init,
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      ...headers,
    },
    body: JSON.stringify(body ?? {}),
  });

  if (!response.ok) {
    const bodyText = await response.text().catch(() => "");
    throw new Error(`POST ${url} failed: ${response.status} ${bodyText}`);
  }

  return response.json() as Promise<T>;
}

function parseDate(value: string | null | undefined): Date | null {
  if (!value) {
    return null;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

export interface Heartbeat {
  ok: boolean;
  ts: Date;
  proofs: {
    success: number;
    failure?: number;
    total?: number;
  };
  proofsPerHour: number;
  blocks: {
    height: number;
    latestMerkle?: string | null;
  };
  policyHash?: string | null;
  redisQueueLength: number;
}

interface RawHeartbeat {
  ok: boolean;
  ts: string;
  proofs: {
    success: number;
    failure?: number;
    total?: number;
  };
  proofs_per_sec: number;
  blocks: {
    height: number;
    latest: {
      merkle?: string | null;
    };
  };
  policy: {
    hash?: string | null;
  };
  redis: {
    ml_jobs_len: number;
  };
}

export async function fetchHeartbeat(): Promise<Heartbeat> {
  const raw = await apiGet<RawHeartbeat>("/heartbeat.json");
  return {
    ok: raw.ok,
    ts: new Date(raw.ts),
    proofs: raw.proofs,
    proofsPerHour: Math.max(0, raw.proofs_per_sec * 3600),
    blocks: {
      height: raw.blocks?.height ?? 0,
      latestMerkle: raw.blocks?.latest?.merkle ?? null,
    },
    policyHash: raw.policy?.hash ?? null,
    redisQueueLength: raw.redis?.ml_jobs_len ?? -1,
  };
}

export interface StatementSummary {
  hash: string;
  display: string;
  text?: string | null;
  normalizedText?: string | null;
}

interface RawRecentStatements {
  items: Array<{
    hash: string;
    display: string;
    text?: string | null;
    normalized_text?: string | null;
  }>;
}

export async function fetchRecentStatements(
  limit = 20
): Promise<StatementSummary[]> {
  const raw = await apiGet<RawRecentStatements>(
    `/ui/recent.json?limit=${encodeURIComponent(limit)}`
  );
  return raw.items.map((item) => ({
    hash: item.hash,
    display: item.display,
    text: item.text ?? null,
    normalizedText: item.normalized_text ?? null,
  }));
}

export interface ParentSummary {
  hash: string;
  display?: string | null;
}

export interface ProofSummary {
  method?: string | null;
  status?: string | null;
  success?: boolean | null;
  createdAt?: Date | null;
  prover?: string | null;
  durationMs?: number | null;
}

export interface StatementDetail {
  hash: string;
  display: string;
  text?: string | null;
  normalizedText?: string | null;
  proofs: ProofSummary[];
  parents: ParentSummary[];
}

interface RawStatementDetail {
  hash: string;
  display: string;
  text?: string | null;
  normalized_text?: string | null;
  proofs: Array<{
    method?: string | null;
    status?: string | null;
    success?: boolean | null;
    created_at?: string | null;
    prover?: string | null;
    duration_ms?: number | null;
  }>;
  parents: Array<{ hash: string; display?: string | null }>;
}

export async function fetchStatementDetail(
  hash: string
): Promise<StatementDetail> {
  const raw = await apiGet<RawStatementDetail>(
    `/ui/statement/${encodeURIComponent(hash)}.json`
  );
  return {
    hash: raw.hash,
    display: raw.display,
    text: raw.text ?? null,
    normalizedText: raw.normalized_text ?? null,
    proofs: raw.proofs.map((proof) => ({
      method: proof.method ?? null,
      status: proof.status ?? null,
      success: proof.success ?? null,
      createdAt: parseDate(proof.created_at),
      prover: proof.prover ?? null,
      durationMs:
        typeof proof.duration_ms === "number"
          ? proof.duration_ms
          : proof.duration_ms != null
          ? Number(proof.duration_ms)
          : null,
    })),
    parents: raw.parents.map((parent) => ({
      hash: parent.hash,
      display: parent.display ?? null,
    })),
  };
}

interface RawProofList {
  proofs: RawStatementDetail["proofs"];
}

export async function fetchProofs(hash: string): Promise<ProofSummary[]> {
  const raw = await apiGet<RawProofList>(
    `/ui/proofs/${encodeURIComponent(hash)}.json`
  );
  return raw.proofs.map((proof) => ({
    method: proof.method ?? null,
    status: proof.status ?? null,
    success: proof.success ?? null,
    createdAt: parseDate(proof.created_at),
    prover: proof.prover ?? null,
    durationMs:
      typeof proof.duration_ms === "number"
        ? proof.duration_ms
        : proof.duration_ms != null
        ? Number(proof.duration_ms)
        : null,
  }));
}

interface RawParentList {
  parents: Array<{ hash: string; display?: string | null }>;
}

export async function fetchParents(
  hash: string
): Promise<ParentSummary[]> {
  const raw = await apiGet<RawParentList>(
    `/ui/parents/${encodeURIComponent(hash)}.json`
  );
  return raw.parents.map((parent) => ({
    hash: parent.hash,
    display: parent.display ?? null,
  }));
}

export interface AttestationLeafMetadata {
  originalIndex: number;
  sortedIndex: number;
  canonicalValue: string;
  leafHash: string;
  merkleProof: Array<[string, boolean]>;
}

export interface AttestationSummary {
  blockNumber: number | null;
  blockHash: string | null;
  reasoningMerkleRoot: string;
  uiMerkleRoot: string;
  compositeAttestationRoot: string;
  reasoningLeaves: AttestationLeafMetadata[];
  uiLeaves: AttestationLeafMetadata[];
  metadata: Record<string, unknown>;
}

interface RawAttestationLeaf {
  original_index?: number;
  sorted_index?: number;
  canonical_value?: string;
  leaf_hash?: string;
  merkle_proof?: Array<[string, boolean]>;
}

interface RawAttestationSummary {
  block_number: number | null;
  block_hash?: string | null;
  reasoning_merkle_root: string;
  ui_merkle_root: string;
  composite_attestation_root: string;
  attestation_metadata: {
    reasoning_leaves?: RawAttestationLeaf[];
    ui_leaves?: RawAttestationLeaf[];
    [key: string]: unknown;
  };
}

function mapLeaf(raw: RawAttestationLeaf | undefined): AttestationLeafMetadata {
  const proofEntries = Array.isArray(raw?.merkle_proof)
    ? (raw?.merkle_proof ?? [])
    : [];
  const proof = proofEntries.map(
    (entry): [string, boolean] => [
      String(entry?.[0] ?? ""),
      Boolean(entry?.[1]),
    ]
  );
  return {
    originalIndex:
      raw?.original_index != null ? Number(raw.original_index) : 0,
    sortedIndex: raw?.sorted_index != null ? Number(raw.sorted_index) : 0,
    canonicalValue: raw?.canonical_value ?? "",
    leafHash: raw?.leaf_hash ?? "",
    merkleProof: proof,
  };
}

export async function fetchLatestAttestation(): Promise<AttestationSummary> {
  const raw = await apiGet<RawAttestationSummary>("/attestation/latest");
  const metadata = raw.attestation_metadata ?? {};
  const reasoningLeaves = Array.isArray(metadata.reasoning_leaves)
    ? metadata.reasoning_leaves.map(mapLeaf)
    : [];
  const uiLeaves = Array.isArray(metadata.ui_leaves)
    ? metadata.ui_leaves.map(mapLeaf)
    : [];

  const blockHash =
    raw.block_hash ??
    (typeof metadata.block_hash === "string" ? metadata.block_hash : null);

  return {
    blockNumber:
      raw.block_number != null ? Number(raw.block_number) : null,
    blockHash: blockHash ?? null,
    reasoningMerkleRoot: raw.reasoning_merkle_root,
    uiMerkleRoot: raw.ui_merkle_root,
    compositeAttestationRoot: raw.composite_attestation_root,
    reasoningLeaves,
    uiLeaves,
    metadata,
  };
}

interface RecordUiEventResponse {
  event_id: string;
  timestamp: number;
  leaf_hash: string;
}

export async function postUiEvent(
  event: Record<string, unknown>
): Promise<RecordUiEventResponse> {
  return apiPost<RecordUiEventResponse>("/attestation/ui-event", event);
}
