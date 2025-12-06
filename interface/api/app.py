"""
FastAPI orchestrator aligning UI, derivation, and attestation surfaces.

This module exposes the canonical HTTP schema for UI events, proof insights,
and composite attestation retrieval so every client interaction is contractually
bound to the MathLedger dual-root guarantees.

Reference: MathLedger Whitepaper §6.1 (Orchestrator API Surface).
"""

from substrate.repro.determinism import deterministic_timestamp

_GLOBAL_SEED = 0

import asyncio
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Sequence

import psycopg
import redis
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from substrate.security.runtime_env import (
    MissingEnvironmentVariable,
    get_allowed_origins,
    get_database_url,
)
from ledger.ui_events import capture_ui_event, snapshot_ui_events

from interface.api.schemas import (
    BlockLatestResponse,
    BlockTotals,
    HeartbeatBlocks,
    HeartbeatLatest,
    HeartbeatPolicy,
    HeartbeatProofs,
    HeartbeatRedis,
    HeartbeatResponse,
    HealthResponse,
    MetricsResponse,
    ParentListResponse,
    ParentSummary,
    ProofSummary,
    ProofTotals,
    ProofListResponse,
    RecentStatementItem,
    RecentStatementsResponse,
    StatementDetailResponse,
    StatementTotals,
    AttestationLatestResponse,
    UIEventResponse,
    UIEventListResponse,
    DerivationSimulationResponse,
    FirstOrganismMetrics,
)

HEX64 = re.compile(r'^[a-f0-9]{64}$')

FIRST_ORGANISM_LATENCY_BUCKETS = ["0.1", "0.5", "1.0", "2.0", "5.0", "10.0", "+Inf"]
FIRST_ORGANISM_ABSTENTION_BUCKETS = ["0.05", "0.1", "0.15", "0.2", "0.3", "0.5", "1.0", "+Inf"]

_MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", "1048576"))
_RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "120"))
_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


class RequestSizeLimiter(BaseHTTPMiddleware):
    """Reject requests whose bodies exceed `_MAX_REQUEST_BODY_BYTES`."""

    def __init__(self, app: FastAPI, max_body_bytes: int) -> None:
        super().__init__(app)
        self._max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return await call_next(request)

        body = await request.body()
        if len(body) > self._max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Request entity too large (> {self._max_body_bytes} bytes)."
                    )
                },
            )

        # Re-inject the body for downstream consumers (FastAPI caches by reference).
        request._body = body  # type: ignore[attr-defined]
        return await call_next(request)


class FixedWindowRateLimiter(BaseHTTPMiddleware):
    """Simple, deterministic fixed-window rate limiter keyed by client host."""

    def __init__(self, app: FastAPI, limit: int, window_seconds: int) -> None:
        super().__init__(app)
        self._limit = max(1, limit)
        self._window = max(1, window_seconds)
        self._buckets: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        client_host = request.client.host if request.client else "anonymous"
        now = time.monotonic()

        async with self._lock:
            bucket = self._buckets.setdefault(client_host, [])
            bucket[:] = [stamp for stamp in bucket if now - stamp < self._window]

            if len(bucket) >= self._limit:
                retry_after = self._window - (now - bucket[0])
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )
                response.headers["Retry-After"] = f"{int(max(1, retry_after))}"
                response.headers["X-RateLimit-Limit"] = str(self._limit)
                response.headers["X-RateLimit-Remaining"] = "0"
                return response

            bucket.append(now)

        return await call_next(request)


def _normalize_success_flag(value: Any) -> Optional[bool]:
    """Best-effort coercion of heterogeneous success indicators into booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "success", "ok", "passed", "pass"}:
            return True
        if lowered in {"false", "f", "0", "fail", "failed", "error"}:
            return False
    return None


def _coalesce_display(*candidates: Optional[str], fallback: str) -> str:
    for candidate in candidates:
        if candidate:
            trimmed = candidate.strip()
            if trimmed:
                return trimmed
    return fallback


def require_api_key(x_api_key: str = Header(None, alias='X-API-Key')) -> bool:
    """
    Validate API key from X-API-Key header.
    
    Raises:
        HTTPException: 401 if key is missing or invalid
        
    Returns:
        True if authentication successful
    """
    expected_key = os.getenv('LEDGER_API_KEY')
    if not expected_key:
        raise HTTPException(
            status_code=500, 
            detail="Server misconfiguration: LEDGER_API_KEY not set"
        )
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    if x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    return True


def _db_url() -> str:
    try:
        return get_database_url()
    except MissingEnvironmentVariable as exc:
        raise RuntimeError(str(exc)) from exc


def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """
    Dependency provider (tests may override this symbol).
    """
    try:
        url = _db_url()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if url.startswith("mock://"):
        raise HTTPException(status_code=503, detail="Mock DB in use; override get_db_connection in tests")
    conn = psycopg.connect(url)
    try:
        yield conn
    finally:
        conn.close()


def _get_redis():
    """Get Redis connection with authentication if configured."""
    try:
        from substrate.auth.redis_auth import get_redis_url_with_auth

        redis_url = get_redis_url_with_auth()
        return redis.from_url(redis_url, decode_responses=True)
    except Exception:
        return None


def _load_bucket_counts(redis_client: redis.Redis, prefix: str, labels: Sequence[str]) -> Dict[str, int]:
    """Read bucket counters from Redis for the given labels."""
    buckets: Dict[str, int] = {}
    for label in labels:
        value = redis_client.get(f"{prefix}:{label}")
        if not value:
            continue
        try:
            buckets[label] = int(float(value))
        except (TypeError, ValueError):
            continue
    return buckets

# --- Heartbeat UI & JSON (router-based, safe before app is defined) ---
heartbeat_router = APIRouter()
heartbeat_templates = Jinja2Templates(directory="backend/ui/templates")

# --- Minimal UI (Dashboard + Statement Detail) --------------------------------
ui_router = APIRouter()
templates = Jinja2Templates(directory="backend/ui/templates")

# --- Attestation API ---------------------------------------------------------
attestation_router = APIRouter(prefix="/attestation", tags=["attestation"])

def _proof_success_predicate(cur) -> tuple[str | None, dict]:
    cur.execute("""
      SELECT column_name, data_type
      FROM information_schema.columns
      WHERE table_name='proofs'
    """)
    cols = {r[0].lower(): r[1].lower() for r in cur.fetchall()}
    where_success = None
    def has(c): return c in cols
    def is_bool(c): return has(c) and 'boolean' in cols[c]
    def is_int(c):  return has(c) and ('int' in cols[c] or 'numeric' in cols[c])
    def is_txt(c):  return has(c) and ('char' in cols[c] or 'text' in cols[c])
    if is_bool('success'):
        where_success = "success = TRUE"
    elif is_bool('is_success'):
        where_success = "is_success = TRUE"
    elif is_int('success'):
        where_success = "success = 1"
    elif is_txt('status'):
        where_success = "LOWER(status) IN ('success','ok','passed')"
    elif is_txt('outcome'):
        where_success = "LOWER(outcome) IN ('success','ok','passed')"
    return where_success, cols

def _recent_statements(cur, limit=20):
    # tolerate schema variety: text/statement, normalized_text/normalized, hash/canonical_hash
    cur.execute("""
      SELECT column_name FROM information_schema.columns WHERE table_name='statements'
    """)
    sc = {r[0].lower() for r in cur.fetchall()}
    text_col  = 'text'            if 'text'            in sc else ('statement' if 'statement' in sc else None)
    norm_col  = 'normalized_text' if 'normalized_text' in sc else ('normalized' if 'normalized' in sc else None)
    hash_col  = 'hash'            if 'hash'            in sc else ('canonical_hash' if 'canonical_hash' in sc else None)
    created   = 'created_at'      if 'created_at'      in sc else None

    sel = []
    if text_col: sel.append(text_col)
    if norm_col: sel.append(norm_col)
    if hash_col: sel.append(hash_col)
    order = created or ( 'id' if 'id' in sc else None )
    if not (sel and hash_col):
        return []  # can't render without a hash
    sql = f"SELECT {', '.join(sel)} FROM statements"
    if order: sql += f" ORDER BY COALESCE({order}, now()) DESC"
    sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    rows = cur.fetchall()
    out=[]
    for r in rows:
        d = dict()
        # map tuple back to names we expect in templates
        idx=0
        if text_col: d['text'] = r[idx]; idx+=1
        if norm_col: d['normalized_text'] = r[idx]; idx+=1
        d['hash'] = r[idx]; idx+=1
        # prefer text, else normalized
        d['display'] = (d.get('text') or d.get('normalized_text') or '').strip() or d['hash']
        out.append(d)
    return out


def _resolve_statement_columns(cur) -> Dict[str, str | None]:
    """
    Inspect the statements table once per request and return canonical column names.
    """
    cur.execute("""
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name='statements'
    """)
    names = {row[0].lower(): row[0] for row in cur.fetchall()}
    return {
        "id": names.get("id") or names.get("statement_id"),
        "text": names.get("text") or names.get("statement"),
        "normalized": names.get("normalized_text") or names.get("normalized"),
        "hash": names.get("hash") or names.get("canonical_hash"),
        "created": names.get("created_at"),
    }


def _resolve_proof_columns(cur) -> Dict[str, str]:
    cur.execute("""
      SELECT column_name, data_type
      FROM information_schema.columns
      WHERE table_name='proofs'
    """)
    return {row[0].lower(): row[0] for row in cur.fetchall()}


def _build_select_clause(columns: list[tuple[str | None, str]]) -> tuple[str, list[str]]:
    parts: list[str] = []
    aliases: list[str] = []
    for source, alias in columns:
        if not source:
            continue
        if source == alias:
            parts.append(source)
        else:
            parts.append(f"{source} AS {alias}")
        aliases.append(alias)
    if not parts:
        raise ValueError("no columns available for select")
    return ", ".join(parts), aliases


def _lookup_statement_id(cur, hash_value: str) -> int | None:
    cols = _resolve_statement_columns(cur)
    hash_col = cols["hash"]
    id_col = cols["id"]
    if not (hash_col and id_col):
        return None
    cur.execute(f"""
      SELECT {id_col}
        FROM statements
       WHERE {hash_col} = %s
       LIMIT 1
    """, (hash_value,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def _fetch_statement_bundle(cur, hash_value: str, proof_limit: int = 100) -> Dict[str, Any]:
    cols = _resolve_statement_columns(cur)
    hash_col = cols["hash"]
    if not hash_col:
        raise LookupError("statements:missing-hash-column")

    select_columns = []
    if cols["id"]:
        select_columns.append((cols["id"], "id"))
    if cols["text"]:
        select_columns.append((cols["text"], "text"))
    if cols["normalized"]:
        select_columns.append((cols["normalized"], "normalized_text"))
    select_columns.append((hash_col, "hash"))
    if cols["created"]:
        select_columns.append((cols["created"], "created_at"))
    select_sql, aliases = _build_select_clause(select_columns)

    cur.execute(
        f"SELECT {select_sql} FROM statements WHERE {hash_col}=%s LIMIT 1",
        (hash_value,),
    )
    row = cur.fetchone()
    if not row:
        raise LookupError("statement:not-found")
    stmt = dict(zip(aliases, row))

    if "id" not in stmt or stmt["id"] is None:
        raise LookupError("statement:missing-id")
    try:
        stmt["id"] = int(stmt["id"])
    except (TypeError, ValueError):
        raise LookupError("statement:invalid-id")

    stmt["display"] = (
        (stmt.get("text") or stmt.get("normalized_text") or stmt.get("hash") or "")
        .strip()
        or stmt.get("hash")
    )

    proofs = _fetch_statement_proofs(cur, stmt["id"], limit=proof_limit)
    parents = _fetch_statement_parents(cur, stmt["hash"])

    return {"statement": stmt, "proofs": proofs, "parents": parents}


def _fetch_statement_proofs(cur, statement_id: int, limit: int = 100) -> list[Dict[str, Any]]:
    proof_cols = _resolve_proof_columns(cur)
    select_candidates: list[tuple[str | None, str]] = []
    for alias in ("id", "method", "prover", "status", "success", "duration_ms", "created_at"):
        select_candidates.append((proof_cols.get(alias), alias))

    select_candidates = [item for item in select_candidates if item[0]]
    if not select_candidates:
        return []

    select_sql, aliases = _build_select_clause(select_candidates)
    order_col = proof_cols.get("created_at") or proof_cols.get("id") or select_candidates[0][0]
    order_expr = (
        f"COALESCE({order_col}, now())"
        if proof_cols.get("created_at") and order_col == proof_cols.get("created_at")
        else order_col
    )

    cur.execute(
        f"""
        SELECT {select_sql}
          FROM proofs
         WHERE statement_id = %s
         ORDER BY {order_expr} DESC
         LIMIT %s
        """,
        (statement_id, limit),
    )
    rows = cur.fetchall()
    out: list[Dict[str, Any]] = []
    for row in rows:
        entry = dict(zip(aliases, row))
        out.append(entry)
    return out


def _fetch_statement_parents(cur, hash_value: str) -> list[Dict[str, Any]]:
    cur.execute("SELECT parent_hash FROM proof_parents WHERE child_hash=%s", (hash_value,))
    parent_hashes = [row[0] for row in cur.fetchall() if row and row[0]]
    if not parent_hashes:
        return []

    seen = set()
    ordered_hashes = []
    for h in parent_hashes:
        if h not in seen:
            seen.add(h)
            ordered_hashes.append(h)

    cols = _resolve_statement_columns(cur)
    hash_col = cols["hash"]
    if not hash_col:
        return [{"hash": h, "display": h} for h in ordered_hashes]

    select_columns = [(hash_col, "hash")]
    if cols["text"]:
        select_columns.append((cols["text"], "text"))
    if cols["normalized"]:
        select_columns.append((cols["normalized"], "normalized_text"))
    select_sql, aliases = _build_select_clause(select_columns)

    placeholders = ",".join(["%s"] * len(ordered_hashes))
    cur.execute(
        f"""
        SELECT {select_sql}
          FROM statements
         WHERE {hash_col} IN ({placeholders})
        """,
        ordered_hashes,
    )
    lookup = {}
    for row in cur.fetchall():
        data = dict(zip(aliases, row))
        lookup[data["hash"]] = data

    parents: list[Dict[str, Any]] = []
    for h in ordered_hashes:
        data = lookup.get(h)
        if data:
            display = (
                (data.get("text") or data.get("normalized_text") or data.get("hash") or "")
                .strip()
                or data["hash"]
            )
            parents.append({"hash": data["hash"], "display": display})
        else:
            parents.append({"hash": h, "display": h})
    return parents


def _proofs_to_summaries(items: Sequence[Dict[str, Any]]) -> List[ProofSummary]:
    summaries: List[ProofSummary] = []
    for payload in items:
        created = payload.get("created_at")
        duration_value: Optional[int] = None
        if payload.get("duration_ms") is not None:
            try:
                duration_value = int(payload.get("duration_ms"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                duration_value = None
        summaries.append(
            ProofSummary(
                method=payload.get("method"),
                status=payload.get("status"),
                success=_normalize_success_flag(payload.get("success")),
                created_at=created if isinstance(created, datetime) else created,
                prover=payload.get("prover"),
                duration_ms=duration_value,
            )
        )
    return summaries


def _parents_to_summaries(items: Sequence[Dict[str, Any]]) -> List[ParentSummary]:
    summaries: List[ParentSummary] = []
    for payload in items:
        hash_value = payload.get("hash")
        if not hash_value:
            continue
        summaries.append(
            ParentSummary(
                hash=hash_value,
                display=payload.get("display"),
            )
        )
    return summaries


@ui_router.get("/ui")
def ui_dashboard(request: Request):
    # Live counters + recent statements (no API key required)
    metrics = {"proofs_success": 0, "proofs_per_sec": 0.0, "blocks_height": 0, "merkle": None, "policy_hash": None}
    recent = []
    try:
        with psycopg.connect(_db_url(), connect_timeout=5) as conn, conn.cursor() as cur:
            # proofs.success & pps
            where_success, pcols = _proof_success_predicate(cur)
            succ = 0
            if where_success:
                cur.execute(f"SELECT COUNT(*) FROM proofs WHERE {where_success}")
                succ = int(cur.fetchone()[0] or 0)
            metrics["proofs_success"] = succ
            if where_success and ('created_at' in pcols):
                cur.execute(f"""
                  SELECT COUNT(*) FROM proofs
                  WHERE {where_success} AND created_at >= (now() - interval '300 seconds')
                """)
                last5 = int(cur.fetchone()[0] or 0)
                metrics["proofs_per_sec"] = last5/300.0

            # blocks & merkle
            cur.execute("SELECT COALESCE(MAX(block_number),0) FROM blocks")
            metrics["blocks_height"] = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT merkle_root FROM blocks ORDER BY block_number DESC, created_at DESC LIMIT 1")
            row = cur.fetchone()
            metrics["merkle"] = (row[0] if row else None)

            # policy hash (two styles)
            try:
                cur.execute("SELECT policy_hash FROM policy_settings ORDER BY id DESC LIMIT 1")
                r = cur.fetchone(); metrics["policy_hash"] = r[0] if r and r[0] else None
            except Exception:
                try:
                    cur.execute("SELECT value FROM policy_settings WHERE key='active_policy_hash' ORDER BY id DESC LIMIT 1")
                    r = cur.fetchone(); metrics["policy_hash"] = r[0] if r and r[0] else None
                except Exception:
                    pass

            # recent statements
            recent = _recent_statements(cur, limit=20)
    except Exception:
        pass

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "metrics": metrics,
        "recent": recent
    })

@ui_router.get("/ui/s/{hash}")
def ui_statement_detail(hash: str, request: Request):
    if not HEX64.match(hash or ""):
        return templates.TemplateResponse(
            "statement_detail.html",
            {"request": request, "not_found": True, "hash": hash},
        )

    bundle: Optional[Dict[str, Any]] = None
    try:
        with psycopg.connect(_db_url(), connect_timeout=5) as conn, conn.cursor() as cur:
            bundle = _fetch_statement_bundle(cur, hash, proof_limit=50)
    except LookupError:
        bundle = None
    except Exception:
        bundle = None

    if not bundle:
        return templates.TemplateResponse(
            "statement_detail.html",
            {"request": request, "not_found": True, "hash": hash},
        )

    stmt = {k: v for k, v in bundle["statement"].items() if k != "id"}

    return templates.TemplateResponse(
        "statement_detail.html",
        {
            "request": request,
            "hash": hash,
            "stmt": stmt,
            "proofs": bundle["proofs"],
            "parents": bundle["parents"],
        },
    )


@ui_router.get("/ui/statement/{hash}.json", response_model=StatementDetailResponse)
def ui_statement_json(hash: str) -> StatementDetailResponse:
    if not HEX64.match(hash or ""):
        raise HTTPException(status_code=400, detail="Invalid hash format")

    try:
        with psycopg.connect(_db_url(), connect_timeout=5) as conn, conn.cursor() as cur:
            bundle = _fetch_statement_bundle(cur, hash, proof_limit=100)
    except LookupError:
        raise HTTPException(status_code=404, detail="statement not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"db error: {type(exc).__name__}") from exc

    stmt = bundle["statement"]
    return StatementDetailResponse(
        hash=stmt["hash"],
        text=stmt.get("text"),
        normalized_text=stmt.get("normalized_text"),
        display=stmt["display"],
        proofs=_proofs_to_summaries(bundle["proofs"]),
        parents=_parents_to_summaries(bundle["parents"]),
    )


@ui_router.get("/ui/parents/{hash}.json", response_model=ParentListResponse)
def ui_parents_json(hash: str) -> ParentListResponse:
    if not HEX64.match(hash or ""):
        return ParentListResponse(parents=[])

    parents: List[Dict[str, Any]] = []
    try:
        with psycopg.connect(_db_url(), connect_timeout=5) as conn, conn.cursor() as cur:
            parents = _fetch_statement_parents(cur, hash)
    except Exception:
        parents = []

    return ParentListResponse(parents=_parents_to_summaries(parents))


@ui_router.get("/ui/proofs/{hash}.json", response_model=ProofListResponse)
def ui_proofs_json(hash: str) -> ProofListResponse:
    if not HEX64.match(hash or ""):
        return ProofListResponse(proofs=[])

    proofs: List[Dict[str, Any]] = []
    try:
        with psycopg.connect(_db_url(), connect_timeout=5) as conn, conn.cursor() as cur:
            statement_id = _lookup_statement_id(cur, hash)
            if statement_id is not None:
                proofs = _fetch_statement_proofs(cur, statement_id, limit=100)
    except Exception:
        proofs = []

    return ProofListResponse(proofs=_proofs_to_summaries(proofs))


@ui_router.get("/ui/recent.json", response_model=RecentStatementsResponse)
def ui_recent_json(limit: int = 20) -> RecentStatementsResponse:
    items: List[RecentStatementItem] = []
    try:
        with psycopg.connect(_db_url(), connect_timeout=5) as conn, conn.cursor() as cur:
            for entry in _recent_statements(cur, limit=limit):
                hash_value = entry.get("hash")
                if not hash_value:
                    continue
                items.append(
                    RecentStatementItem(
                        hash=hash_value,
                        display=_coalesce_display(
                            entry.get("display"),
                            entry.get("text"),
                            entry.get("normalized_text"),
                            fallback=hash_value,
                        ),
                        text=entry.get("text"),
                        normalized_text=entry.get("normalized_text"),
                    )
                )
    except Exception:
        pass
    return RecentStatementsResponse(items=items)

@heartbeat_router.get("/heartbeat")
def heartbeat_page(request: Request):
    return heartbeat_templates.TemplateResponse("heartbeat.html", {"request": request})

@heartbeat_router.get("/heartbeat.json", response_model=HeartbeatResponse)
def heartbeat_json() -> HeartbeatResponse:
    ok = True
    success_count = 0
    proofs_per_sec = 0.0
    block_height = 0
    latest_merkle: Optional[str] = None
    policy_hash: Optional[str] = None
    redis_depth = -1

    try:
        with psycopg.connect(_db_url(), connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM proofs")
            cur.fetchone()

            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name='proofs'
                """
            )
            cols = {r[0].lower(): r[1].lower() for r in cur.fetchall()}

            where_success = None

            def has(col: str) -> bool:
                return col in cols

            def is_bool(col: str) -> bool:
                return has(col) and "boolean" in cols[col]

            def is_int(col: str) -> bool:
                return has(col) and ("int" in cols[col] or "numeric" in cols[col])

            def is_txt(col: str) -> bool:
                return has(col) and ("char" in cols[col] or "text" in cols[col])

            if is_bool("success"):
                where_success = "success = TRUE"
            elif is_bool("is_success"):
                where_success = "is_success = TRUE"
            elif is_int("success"):
                where_success = "success = 1"
            elif is_txt("status"):
                where_success = "LOWER(status) IN ('success','ok','passed')"
            elif is_txt("outcome"):
                where_success = "LOWER(outcome) IN ('success','ok','passed')"

            if where_success:
                cur.execute(f"SELECT COUNT(*) FROM proofs WHERE {where_success}")
                success_count = int(cur.fetchone()[0] or 0)

                if has("created_at"):
                    cur.execute(
                        f"""
                        SELECT COUNT(*) FROM proofs
                        WHERE {where_success} AND created_at >= (now() - interval '300 seconds')
                        """
                    )
                    last5 = int(cur.fetchone()[0] or 0)
                    proofs_per_sec = last5 / 300.0

            cur.execute("SELECT COALESCE(MAX(block_number),0) FROM blocks")
            block_height = int(cur.fetchone()[0] or 0)

            cur.execute("SELECT merkle_root FROM blocks ORDER BY block_number DESC, created_at DESC LIMIT 1")
            row = cur.fetchone()
            latest_merkle = row[0] if row else None

            try:
                cur.execute("SELECT policy_hash FROM policy_settings ORDER BY id DESC LIMIT 1")
                r = cur.fetchone()
                policy_hash = r[0] if r and r[0] else None
            except Exception:
                try:
                    cur.execute(
                        "SELECT value FROM policy_settings WHERE key='active_policy_hash' ORDER BY id DESC LIMIT 1"
                    )
                    r = cur.fetchone()
                    policy_hash = r[0] if r and r[0] else None
                except Exception:
                    pass
    except Exception:
        ok = False

    try:
        redis_client = _get_redis()
        if redis_client:
            redis_depth = int(redis_client.llen("ml:jobs"))
    except Exception:
        redis_depth = -1

    return HeartbeatResponse(
        ok=ok,
        ts=deterministic_timestamp(_GLOBAL_SEED),
        proofs=HeartbeatProofs(success=success_count),
        proofs_per_sec=proofs_per_sec,
        blocks=HeartbeatBlocks(height=block_height, latest=HeartbeatLatest(merkle=latest_merkle)),
        policy=HeartbeatPolicy(hash=policy_hash),
        redis=HeartbeatRedis(ml_jobs_len=redis_depth),
    )


@attestation_router.post("/ui-event", response_model=UIEventResponse)
def record_ui_event(event: Dict[str, Any] = Body(..., embed=False)) -> UIEventResponse:
    """
    Capture a single UI event for inclusion in the next attestation.
    """
    if not isinstance(event, dict):
        raise HTTPException(status_code=400, detail="Event payload must be a JSON object")

    record = capture_ui_event(event)
    return UIEventResponse(
        event_id=record.event_id,
        timestamp=record.timestamp,
        leaf_hash=record.leaf_hash,
    )


@attestation_router.get("/ui-events", response_model=UIEventListResponse)
def list_ui_events() -> UIEventListResponse:
    """
    Return canonicalized UI events captured so far (deterministic ordering).
    """
    records = snapshot_ui_events()
    return UIEventListResponse(
        events=[
            {
                "event_id": record.event_id,
                "timestamp": record.timestamp,
                "leaf_hash": record.leaf_hash,
                "canonical_value": record.canonical_value,
                "metadata": record.metadata,
            }
            for record in records
        ]
    )


@attestation_router.get("/latest", response_model=AttestationLatestResponse)
def latest_attestation(conn: psycopg.Connection = Depends(get_db_connection)) -> AttestationLatestResponse:
    """
    Fetch the most recent dual-root attestation from the blocks table.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    block_number,
                    reasoning_merkle_root,
                    ui_merkle_root,
                    composite_attestation_root,
                    attestation_metadata
                FROM blocks
                WHERE reasoning_merkle_root IS NOT NULL
                ORDER BY block_number DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    except Exception as exc:  # pragma: no cover - defensive path
        raise HTTPException(status_code=500, detail=f"attestation query failed: {exc.__class__.__name__}") from exc

    if not row:
        raise HTTPException(status_code=404, detail="no attestation")

    block_number, r_t, u_t, h_t, metadata = row
    metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}

    return AttestationLatestResponse(
        block_number=int(block_number) if block_number is not None else None,
        reasoning_merkle_root=r_t,
        ui_merkle_root=u_t,
        composite_attestation_root=h_t,
        attestation_metadata=metadata_dict,
        block_hash=metadata_dict.get("block_hash"),
    )


@attestation_router.post("/simulate-derivation", response_model=DerivationSimulationResponse)
def simulate_derivation() -> DerivationSimulationResponse:
    """
    Stub endpoint to trigger or simulate a derivation run.
    """
    return DerivationSimulationResponse(
        triggered=True,
        job_id="simulated-job-id",
        status="queued"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if os.getenv("DISABLE_DB_STARTUP") != "1":
        url = _db_url()
        if not url.startswith("mock://"):
            try:
                with psycopg.connect(url) as c, c.cursor() as cur:
                    cur.execute("SELECT 1")
            except Exception:
                pass
    yield
    # Shutdown
    pass

app = FastAPI(lifespan=lifespan)

try:
    _allowed_origins = get_allowed_origins()
except MissingEnvironmentVariable:
    # Fallback for test/dev environments without CORS_ALLOWED_ORIGINS
    _allowed_origins = ["http://localhost", "http://127.0.0.1"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    expose_headers=["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    max_age=300,
)
app.add_middleware(RequestSizeLimiter, max_body_bytes=_MAX_REQUEST_BODY_BYTES)
app.add_middleware(
    FixedWindowRateLimiter,
    limit=_RATE_LIMIT_REQUESTS,
    window_seconds=_RATE_LIMIT_WINDOW_SECONDS,
)
app.include_router(heartbeat_router)
app.include_router(attestation_router)
app.include_router(ui_router)

templates = Jinja2Templates(directory="backend/ui/templates")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", timestamp=deterministic_timestamp(_GLOBAL_SEED))

@app.get("/metrics", response_model=MetricsResponse)
def metrics(conn: psycopg.Connection = Depends(get_db_connection)) -> MetricsResponse:
    """
    Deterministic system metrics aligned with tight JSON schema guarantees.
    """
    block_height = 0
    block_count = 0
    statement_count = 0
    proofs_total = 0
    proofs_success = 0
    max_depth = 0
    queue_length = -1

    try:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT COALESCE(MAX(block_number),0) FROM blocks")
                row = cur.fetchone()
                block_height = int(row[0]) if row and row[0] is not None else 0

                cur.execute("SELECT COUNT(*) FROM blocks")
                block_count = int(cur.fetchone()[0] or 0)
            except Exception:
                block_height = 0
                block_count = 0

            try:
                cur.execute("SELECT COUNT(*) FROM statements")
                statement_count = int(cur.fetchone()[0] or 0)
            except Exception:
                statement_count = 0

            try:
                cur.execute("SELECT COALESCE(MAX(derivation_depth),0) FROM statements")
                md = cur.fetchone()
                max_depth = int(md[0]) if md and md[0] is not None else 0
            except Exception:
                max_depth = 0

            try:
                cur.execute("SELECT COUNT(*) FROM proofs")
                proofs_total = int(cur.fetchone()[0] or 0)

                cur.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name='proofs'
                    """
                )
                cols = {r[0].lower(): r[1].lower() for r in cur.fetchall()}

                def has(col: str) -> bool:
                    return col in cols

                def is_bool(col: str) -> bool:
                    return has(col) and "boolean" in cols[col]

                def is_int(col: str) -> bool:
                    return has(col) and ("int" in cols[col] or "numeric" in cols[col])

                def is_txt(col: str) -> bool:
                    return has(col) and ("char" in cols[col] or "text" in cols[col])

                if is_bool("success"):
                    cur.execute("SELECT COUNT(*) FROM proofs WHERE success = TRUE")
                    proofs_success = int(cur.fetchone()[0] or 0)
                elif is_bool("is_success"):
                    cur.execute("SELECT COUNT(*) FROM proofs WHERE is_success = TRUE")
                    proofs_success = int(cur.fetchone()[0] or 0)
                elif is_int("success"):
                    cur.execute("SELECT COUNT(*) FROM proofs WHERE success = 1")
                    proofs_success = int(cur.fetchone()[0] or 0)
                elif is_txt("status"):
                    cur.execute("SELECT COUNT(*) FROM proofs WHERE LOWER(status) IN ('success','ok','passed')")
                    proofs_success = int(cur.fetchone()[0] or 0)
                elif is_txt("outcome"):
                    cur.execute("SELECT COUNT(*) FROM proofs WHERE LOWER(outcome) IN ('success','ok','passed')")
                    proofs_success = int(cur.fetchone()[0] or 0)
                elif is_txt("result"):
                    cur.execute("SELECT COUNT(*) FROM proofs WHERE LOWER(result) IN ('success','ok','passed')")
                    proofs_success = int(cur.fetchone()[0] or 0)
                else:
                    proofs_success = 0
            except Exception:
                proofs_total = 0
                proofs_success = 0
    except Exception:
        pass

    proofs_success = min(proofs_success, proofs_total)
    proofs_failure = max(0, proofs_total - proofs_success)
    success_rate = round(proofs_success / proofs_total, 6) if proofs_total else 0.0

    first_organism: Optional[FirstOrganismMetrics] = None
    try:
        redis_client = _get_redis()
        if redis_client:
            queue_length = int(redis_client.llen("ml:jobs"))

            # First Organism Telemetry
            fo_runs = redis_client.get("ml:metrics:first_organism:runs_total")
            if fo_runs:
                fo_ht = redis_client.get("ml:metrics:first_organism:last_ht")
                fo_lat = redis_client.get("ml:metrics:first_organism:latency_seconds")
                fo_completed = redis_client.get("ml:metrics:first_organism:runs_completed") or "0"
                fo_failed = redis_client.get("ml:metrics:first_organism:runs_failed") or "0"
                fo_abst_rate = redis_client.get("ml:metrics:first_organism:last_abstention_rate")
                latency_buckets = _load_bucket_counts(
                    redis_client,
                    "ml:metrics:first_organism:latency_bucket",
                    FIRST_ORGANISM_LATENCY_BUCKETS,
                )
                abstention_buckets = _load_bucket_counts(
                    redis_client,
                    "ml:metrics:first_organism:abstention_bucket",
                    FIRST_ORGANISM_ABSTENTION_BUCKETS,
                )
                first_organism = FirstOrganismMetrics(
                    runs_total=int(float(fo_runs)),
                    runs_completed=int(float(fo_completed)),
                    runs_failed=int(float(fo_failed)),
                    last_ht_hash=fo_ht,
                    last_abstention_rate=float(fo_abst_rate) if fo_abst_rate else 0.0,
                    latency_seconds=float(fo_lat) if fo_lat else 0.0,
                    latency_buckets=latency_buckets,
                    abstention_buckets=abstention_buckets,
                )
    except Exception:
        queue_length = -1

    return MetricsResponse(
        generated_at=deterministic_timestamp(_GLOBAL_SEED),
        proofs=ProofTotals(
            total=proofs_total,
            success=proofs_success,
            failure=proofs_failure,
            success_rate=success_rate,
        ),
        statements=StatementTotals(total=statement_count, max_depth=max_depth),
        blocks=BlockTotals(count=block_count, height=block_height),
        queue_length=queue_length,
        first_organism=first_organism,
    )

@app.get("/blocks/latest", response_model=BlockLatestResponse)
def blocks_latest(conn: psycopg.Connection = Depends(get_db_connection)) -> BlockLatestResponse:
    """
    404 with {"detail":"no blocks"} when empty.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT block_number, merkle_root, created_at, header
                FROM blocks
                ORDER BY block_number DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="no blocks")
            block_number, merkle_root, created_at, header = row
            return BlockLatestResponse(
                block_number=int(block_number) if block_number is not None else 0,
                merkle_root=merkle_root,
                created_at=created_at,
                header=header or {},
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="no blocks")

@app.get("/statements", response_model=StatementDetailResponse)
def statements_endpoint(
    hash: str = Query(..., description="sha256(normalized_text) 64-hex"),
    _: bool = Depends(require_api_key),
    conn: psycopg.Connection = Depends(get_db_connection),
) -> StatementDetailResponse:
    if not HEX64.match(hash or ""):
        raise HTTPException(status_code=400, detail="Invalid hash format")

    try:
        with conn.cursor() as cur:
            bundle = _fetch_statement_bundle(cur, hash, proof_limit=100)
            stmt = bundle["statement"]
            return StatementDetailResponse(
                hash=stmt["hash"],
                text=stmt.get("text"),
                normalized_text=stmt.get("normalized_text"),
                display=stmt["display"],
                proofs=_proofs_to_summaries(bundle["proofs"]),
                parents=_parents_to_summaries(bundle["parents"]),
            )
    except LookupError:
        raise HTTPException(status_code=404, detail="statement not found")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"db error: {type(exc).__name__}") from exc
from interface.api.routes.parents import parents_router
app.include_router(parents_router)
