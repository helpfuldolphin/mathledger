"""
Parent provenance routes for cataloguing dependent proofs.

# NOTE: Canonical interface module; backend.* imports are forbidden here.

Returns canonical parent summaries and proof references so the dual-root chain
can be audited across downstream consumers.

Reference: MathLedger Whitepaper §6.2 (Parent Provenance API).
"""

from __future__ import annotations

from typing import Any, List, Optional

import psycopg
from fastapi import APIRouter

from interface.api.schemas import ParentListResponse, ParentSummary, ProofListResponse, ProofSummary
from substrate.security.runtime_env import MissingEnvironmentVariable, get_database_url


parents_router = APIRouter()


def _db_url() -> str:
    try:
        return get_database_url()
    except MissingEnvironmentVariable as exc:
        raise RuntimeError(str(exc)) from exc


def _coalesce_display(*candidates: Optional[str], fallback: str) -> str:
    for candidate in candidates:
        if candidate:
            trimmed = candidate.strip()
            if trimmed:
                return trimmed
    return fallback


def _normalize_success_flag(value: Any) -> Optional[bool]:
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


@parents_router.get("/ui/parents/{hash}.json", response_model=ParentListResponse)
def ui_parents_json(hash: str) -> ParentListResponse:
    parents: List[ParentSummary] = []
    try:
        with psycopg.connect(_db_url(), connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("SELECT parent_hash FROM proof_parents WHERE child_hash=%s", (hash,))
            hs = [r[0] for r in cur.fetchall()]
            if hs:
                ph = ",".join(["%s"] * len(hs))
                # resolve display text for parents
                cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='statements'")
                sc = {r[0].lower() for r in cur.fetchall()}
                t = 'text' if 'text' in sc else ('statement' if 'statement' in sc else None)
                n = 'normalized_text' if 'normalized_text' in sc else ('normalized' if 'normalized' in sc else None)
                hcol = 'hash' if 'hash' in sc else ('canonical_hash' if 'canonical_hash' in sc else None)
                if hcol:
                    cur.execute(
                        f"SELECT {hcol},{t if t else 'NULL'},{n if n else 'NULL'} FROM statements WHERE {hcol} IN ({ph})",
                        tuple(hs),
                    )
                    for hh, tt, nn in cur.fetchall():
                        parents.append(
                            ParentSummary(
                                hash=hh,
                                display=_coalesce_display(tt, nn, fallback=hh),
                            )
                        )
    except Exception:
        pass
    return ParentListResponse(parents=parents)


@parents_router.get("/ui/proofs/{hash}.json", response_model=ProofListResponse)
def ui_proofs_json(hash: str) -> ProofListResponse:
    proofs: List[ProofSummary] = []
    try:
        with psycopg.connect(_db_url(), connect_timeout=5) as conn, conn.cursor() as cur:
            # find statement id
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='statements'")
            sc = {r[0].lower() for r in cur.fetchall()}
            hcol = 'hash' if 'hash' in sc else ('canonical_hash' if 'canonical_hash' in sc else None)
            if not hcol:
                return ProofListResponse(proofs=proofs)
            cur.execute(f"SELECT id FROM statements WHERE {hcol}=%s LIMIT 1", (hash,))
            r = cur.fetchone()
            if not r:
                return ProofListResponse(proofs=proofs)
            sid = int(r[0])
            # schema-tolerant proofs
            cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='proofs'")
            pc = {r[0].lower(): r[1].lower() for r in cur.fetchall()}
            cols = [c for c in ("method", "status", "success", "created_at") if c in pc]
            if cols:
                cur.execute(
                    f"SELECT {', '.join(cols)} FROM proofs WHERE statement_id=%s ORDER BY COALESCE(created_at, now()) DESC LIMIT 100",
                    (sid,),
                )
                for row in cur.fetchall():
                    payload: dict[str, Any] = {}
                    for idx, column in enumerate(cols):
                        value = row[idx]
                        if column == "success":
                            payload["success"] = _normalize_success_flag(value)
                        elif column == "created_at":
                            payload["created_at"] = value
                        else:
                            payload[column] = value
                    proofs.append(ProofSummary(**payload))
    except Exception:
        pass
    return ProofListResponse(proofs=proofs)
