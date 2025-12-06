import Dashboard from "@/components/Dashboard";
import {
  fetchHeartbeat,
  fetchRecentStatements,
  fetchStatementDetail,
  type Heartbeat,
  type StatementDetail,
  type StatementSummary,
} from "@/lib/api";

export const dynamic = "force-dynamic";

async function loadHeartbeat(): Promise<Heartbeat | null> {
  try {
    return await fetchHeartbeat();
  } catch {
    return null;
  }
}

async function loadRecentStatements(): Promise<StatementSummary[]> {
  try {
    return await fetchRecentStatements(24);
  } catch {
    return [];
  }
}

async function loadStatementDetail(
  hash: string | null
): Promise<StatementDetail | null> {
  if (!hash) {
    return null;
  }
  try {
    return await fetchStatementDetail(hash);
  } catch {
    return null;
  }
}

export default async function HomePage() {
  const [heartbeat, statements] = await Promise.all([
    loadHeartbeat(),
    loadRecentStatements(),
  ]);

  const initialDetail = await loadStatementDetail(statements[0]?.hash ?? null);

  return (
    <Dashboard
      initialHeartbeat={heartbeat}
      initialStatements={statements}
      initialDetail={initialDetail}
    />
  );
}
