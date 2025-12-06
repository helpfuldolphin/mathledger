"""
Utility functions for the axiom engine derivation system.

Provides database helpers, Redis connection, and diagnostic utilities.
"""

from __future__ import annotations

import os
from typing import Optional, Any, Set

try:
    import psycopg
    from psycopg import errors as pg_errors
except Exception:  # pragma: no cover
    psycopg = None
    pg_errors = None

try:
    import redis
except ImportError:
    redis = None

from backend.crypto.core import sha256_hex, DOMAIN_STMT
from backend.dag import ProofDagRepository
from normalization.canon import canonical_bytes


def sha256_statement(s: str) -> str:
    """
    Compute SHA-256 hash of statement with domain separation.
    
    Args:
        s: Statement text to hash
        
    Returns:
        64-character hex hash
    """
    return sha256_hex(canonical_bytes(s), domain=DOMAIN_STMT)


def print_diagnostic(prefix: str, e: BaseException, table: str) -> None:
    """
    Print best-effort psycopg diagnostics for database errors.
    
    Args:
        prefix: Error prefix
        e: Exception that was raised
        table: Table name where error occurred
    """
    col = tbl = None
    diag = getattr(e, "diag", None)
    if diag is not None:
        col = getattr(diag, "column_name", None)
        tbl = getattr(diag, "table_name", None)
    cls = type(e).__name__
    print(f"ERR={cls} SQL={table} COL={col} TABLE={tbl}", flush=True)


def get_table_columns(cur, table: str) -> Set[str]:
    """
    Get set of column names for a table.
    
    Args:
        cur: Database cursor
        table: Table name
        
    Returns:
        Set of lowercase column names
    """
    cur.execute(
        """
        SELECT column_name
          FROM information_schema.columns
         WHERE table_schema='public' AND table_name=%s
        """,
        (table,),
    )
    return {r[0].lower() for r in cur.fetchall()}


def ensure_redis() -> Optional[Any]:
    """
    Create Redis connection from REDIS_URL environment variable.
    
    Returns:
        Redis client or None if unavailable
    """
    url = os.getenv("REDIS_URL")
    if not url or not redis:
        return None
    try:
        return redis.from_url(
            url,
            socket_timeout=0.2,
            socket_connect_timeout=0.2,
            retry_on_timeout=False,
        )
    except Exception:
        return None


def record_proof_edge(
    cur,
    *,
    proof_id: int,
    child_statement_id: int,
    child_hash: str,
    parent_statement_id: int,
    parent_hash: str,
    edge_index: int = 0,
) -> None:
    """
    Record a parent-child relationship for a proof in proof_parents.
    
    Args:
        cur: Database cursor
        proof_id: Proof identifier
        child_statement_id: Statement proven by the proof
        child_hash: Canonical hash of the child statement
        parent_statement_id: Parent statement identifier
        parent_hash: Canonical hash of the parent statement
        edge_index: Stable ordering of the dependency within the proof
    """
    repo = ProofDagRepository(cur)
    repo.insert_edge(
        proof_id=proof_id,
        child_statement_id=child_statement_id,
        child_hash=child_hash,
        parent_statement_id=parent_statement_id,
        parent_hash=parent_hash,
        edge_index=edge_index,
    )


def get_or_create_system_id(cur, name: str = "pl") -> int:
    """
    Return an existing systems.id or create one with available columns.
    
    Args:
        cur: Database cursor
        name: System name
        
    Returns:
        System ID
    """
    from backend.repro.determinism import deterministic_timestamp
    
    cols = get_table_columns(cur, "systems")
    cur.execute("SELECT id FROM systems WHERE name=%s LIMIT 1", (name,))
    row = cur.fetchone()
    if row:
        return int(row[0])

    # Build dynamic insert
    insert_cols: list[str] = []
    params: list[Any] = []

    # name is required by our DDL
    if "name" in cols:
        insert_cols += ["name"]
        params += [name]

    # optional columns
    if "slug" in cols:
        insert_cols += ["slug"]
        params += [name]
    if "logic" in cols:
        insert_cols += ["logic"]
        params += ["PL"]
    if "version" in cols:
        insert_cols += ["version"]
        params += ["v1"]
    if "created_at" in cols:
        insert_cols += ["created_at"]
        params += [deterministic_timestamp(0)]

    if not insert_cols:
        raise RuntimeError("systems table has no compatible insertable columns")

    ph = ",".join(["%s"] * len(insert_cols))
    sql = f"INSERT INTO systems ({', '.join(insert_cols)}) VALUES ({ph}) RETURNING id"
    cur.execute(sql, params)
    return int(cur.fetchone()[0])
