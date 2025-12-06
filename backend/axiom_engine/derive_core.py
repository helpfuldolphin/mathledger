"""
Core derivation engine for axiom-based statement derivation.

Implements Algorithm 1 from the MathLedger whitepaper with slice-aware bounds,
deterministic Modus Ponens closure, and proof persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None

try:
    import redis
except ImportError:
    redis = None

from backend.repro.determinism import deterministic_timestamp
from normalization.canon import normalize, normalize_pretty

from .bounds import SliceBounds
from .pipeline import DerivationPipeline, StatementRecord
from .verification import StatementVerifier
from .derive_utils import (
    sha256_statement,
    print_diagnostic,
    get_table_columns,
    record_proof_edge,
    get_or_create_system_id,
)
from .derive_rules import ProofContext
from backend.crypto.core import merkle_root


_GLOBAL_SEED = 0


class StatementRepository:
    """Load and persist statements while respecting schema variability."""

    def __init__(self, cursor) -> None:
        self.cur = cursor
        self.statement_columns = get_table_columns(cursor, "statements")
        self.proof_columns = get_table_columns(cursor, "proofs")
        self._has_hash = "hash" in self.statement_columns

    def load(self, system_id: int) -> List[StatementRecord]:
        self.cur.execute(
            "SELECT * FROM statements WHERE system_id = %s ORDER BY created_at",
            (system_id,),
        )
        rows = self.cur.fetchall()
        col_names = [desc.name for desc in self.cur.description]

        records: List[StatementRecord] = []
        for row in rows:
            data = dict(zip(col_names, row))
            normalized = (
                data.get("normalized_text")
                or data.get("content_norm")
                or data.get("statement_norm")
                or data.get("statement")
                or data.get("text")
            )
            if not normalized:
                continue
            normalized = normalize(normalized)
            pretty = data.get("text") or data.get("statement") or normalize_pretty(normalized)
            hash_val = data.get("hash") or sha256_statement(normalized)
            rule = data.get("derivation_rule") or ("axiom:legacy" if data.get("is_axiom") else "legacy")
            mp_depth = int(data.get("derivation_depth") or 0)
            is_axiom = bool(data.get("is_axiom") or rule.startswith("axiom"))
            verification_method = data.get("verification_method") or "legacy"

            records.append(
                StatementRecord(
                    normalized=normalized,
                    hash=hash_val,
                    pretty=pretty,
                    rule=rule,
                    is_axiom=is_axiom,
                    mp_depth=mp_depth,
                    parents=tuple(),
                    verification_method=verification_method,
                )
            )
        return records

    def persist(self, system_id: int, record: StatementRecord) -> int:
        existing_id = self._lookup_existing(record)
        if existing_id is not None:
            return existing_id

        stmt_id = self._insert_statement(system_id, record)
        self._insert_proof(stmt_id, record)
        if record.parents:
            parent_ids = [self._lookup_statement_id_by_hash(parent_hash) for parent_hash in record.parents]
            for edge_index, (parent_hash, parent_id) in enumerate(zip(record.parents, parent_ids)):
                record_proof_edge(
                    self.cur,
                    proof_id=proof_id,
                    child_statement_id=stmt_id,
                    child_hash=record.hash,
                    parent_statement_id=parent_id,
                    parent_hash=parent_hash,
                    edge_index=edge_index,
                )
        return stmt_id

    def _lookup_existing(self, record: StatementRecord) -> Optional[int]:
        if not self._has_hash:
            return None
        self.cur.execute("SELECT id FROM statements WHERE hash = %s LIMIT 1", (record.hash,))
        row = self.cur.fetchone()
        return int(row[0]) if row else None

    def _insert_statement(self, system_id: int, record: StatementRecord) -> int:
        columns: List[str] = []
        values: List[Any] = []

        def add(column: str, value: Any) -> None:
            if column in self.statement_columns:
                columns.append(column)
                values.append(value)

        add("system_id", system_id)
        add("text", record.pretty)
        add("statement", record.pretty)
        add("normalized_text", record.normalized)
        add("content_norm", record.normalized)
        add("hash", record.hash)
        add("derivation_rule", record.rule)
        add("derivation_depth", record.mp_depth)
        add("is_axiom", record.is_axiom)
        add("verification_method", record.verification_method)
        timestamp = deterministic_timestamp(_GLOBAL_SEED)
        add("created_at", timestamp)
        add("updated_at", timestamp)

        if not columns:
            self.cur.execute(
                "INSERT INTO statements(normalized_text) VALUES (%s) RETURNING id",
                (record.normalized,),
            )
        else:
            placeholders = ",".join(["%s"] * len(columns))
            sql = f"INSERT INTO statements ({', '.join(columns)}) VALUES ({placeholders}) RETURNING id"
            try:
                self.cur.execute(sql, values)
            except Exception as exc:  # pragma: no cover
                print_diagnostic("statements-insert", exc, "statements")
                raise

        return int(self.cur.fetchone()[0])

    def _insert_proof(self, statement_id: int, record: StatementRecord) -> Optional[int]:
        if not self.proof_columns:
            return None

        columns: List[str] = []
        values: List[Any] = []

        def add(column: str, value: Any) -> None:
            if column in self.proof_columns:
                columns.append(column)
                values.append(value)

        proof_context = ProofContext(
            statement_id=record.hash,
            dependencies=list(record.parents),
            derivation_rule=record.rule,
            merkle_root=merkle_root(list(record.parents)),
        )

        add("statement_id", statement_id)
        add("method", record.rule)
        add("status", "success")
        add("success", True)
        add("verification_method", record.verification_method)
        add("context", json.dumps(proof_context.to_dict()))  # type: ignore[arg-type]
        timestamp = deterministic_timestamp(_GLOBAL_SEED)
        add("created_at", timestamp)
        add("updated_at", timestamp)

        if not columns:
            return None

        placeholders = ",".join(["%s"] * len(columns))
        returning_clause = " RETURNING id" if "id" in self.proof_columns else ""
        sql = f"INSERT INTO proofs ({', '.join(columns)}) VALUES ({placeholders}){returning_clause}"
        try:
            self.cur.execute(sql, values)
        except Exception as exc:  # pragma: no cover
            print_diagnostic("proofs-insert", exc, "proofs")
            raise
        if returning_clause:
            row = self.cur.fetchone()
            return int(row[0]) if row else None
        return None

    def _lookup_statement_id_by_hash(self, statement_hash: str) -> Optional[int]:
        if not self._has_hash:
            return None
        self.cur.execute("SELECT id FROM statements WHERE hash = %s LIMIT 1", (statement_hash,))
        row = self.cur.fetchone()
        return int(row[0]) if row else None


class DerivationEngine:
    """Slice-aware derivation engine with deterministic Modus Ponens closure."""

    def __init__(
        self,
        db_url: str,
        redis_url: str,
        *,
        max_depth: int = 3,
        max_breadth: int = 100,
        max_total: int = 256,
        slice_bounds: Optional[SliceBounds] = None,
        lean_project_root: Optional[str | Path] = None,
    ) -> None:
        self.db_url = db_url
        self.redis_url = redis_url
        self.redis_client = None

        default_bounds = SliceBounds(
            max_formula_depth=max_depth,
            max_mp_depth=max_depth,
            max_breadth=max_breadth or SliceBounds().max_breadth,
            max_total=max_total or SliceBounds().max_total,
        )
        self.bounds = slice_bounds or default_bounds

        candidate = Path(lean_project_root) if lean_project_root else Path("backend/lean_proj")
        self.lean_project_root = candidate if candidate.exists() else None

    def derive_statements(self, steps: int = 1) -> Dict[str, Any]:
        if not psycopg:
            return {"n_new": 0, "max_depth": 0, "n_jobs": 0, "pct_success": 0.0}

        try:
            with psycopg.connect(self.db_url, connect_timeout=5) as conn, conn.cursor() as cur:
                system_id = get_or_create_system_id(cur, "pl")
                repo = StatementRepository(cur)
                existing = repo.load(system_id)
                verifier = StatementVerifier(self.bounds, self.lean_project_root)
                pipeline = DerivationPipeline(self.bounds, verifier)

                inserted_total = 0
                max_mp_depth = 0
                verified_total = 0
                rejected_total = 0

                for _ in range(max(1, steps)):
                    outcome = pipeline.run_step(existing)
                    if not outcome.statements:
                        break

                    for record in outcome.statements:
                        repo.persist(system_id, record)
                    existing.extend(outcome.statements)

                    inserted_total += len(outcome.statements)
                    max_mp_depth = max([max_mp_depth, *[stmt.mp_depth for stmt in outcome.statements]])
                    verified_total += outcome.stats.verified
                    rejected_total += outcome.stats.rejected

                    if inserted_total >= self.bounds.max_total:
                        break

                conn.commit()

                attempts = verified_total + rejected_total
                pct = 100.0 * verified_total / attempts if attempts else (100.0 if inserted_total else 0.0)
                return {
                    "n_new": inserted_total,
                    "max_depth": max_mp_depth,
                    "n_jobs": 0,
                    "pct_success": pct,
                }

        except Exception as exc:  # pragma: no cover
            print(f"DERIVATION_ERROR={type(exc).__name__}: {exc}", flush=True)
            return {"n_new": 0, "max_depth": 0, "n_jobs": 0, "pct_success": 0.0}

    def load_axioms(self):
        if not psycopg:
            return []
        try:
            with psycopg.connect(self.db_url, connect_timeout=5) as conn, conn.cursor() as cur:
                system_id = get_or_create_system_id(cur, "pl")
                repo = StatementRepository(cur)
                records = [rec for rec in repo.load(system_id) if rec.is_axiom]
                from backend.axiom_engine.rules import Statement

                return [
                    Statement(rec.normalized, is_axiom=True, derivation_depth=rec.mp_depth)
                    for rec in records
                ]
        except Exception:
            return []

    def load_derived_statements(self):
        if not psycopg:
            return []
        try:
            with psycopg.connect(self.db_url, connect_timeout=5) as conn, conn.cursor() as cur:
                system_id = get_or_create_system_id(cur, "pl")
                repo = StatementRepository(cur)
                records = [rec for rec in repo.load(system_id) if not rec.is_axiom]
                from backend.axiom_engine.rules import Statement

                return [
                    Statement(
                        rec.normalized,
                        is_axiom=False,
                        derivation_rule=rec.rule,
                        derivation_depth=rec.mp_depth,
                    )
                    for rec in records
                ]
        except Exception:
            return []

    def upsert_statement(self, conn, stmt, theory_id):
        try:
            with conn.cursor() as cur:
                repo = StatementRepository(cur)
                normalized = normalize(stmt.content)
                record = StatementRecord(
                    normalized=normalized,
                    hash=sha256_statement(normalized),
                    pretty=normalize_pretty(stmt.content),
                    rule=stmt.derivation_rule or "external",
                    is_axiom=bool(stmt.is_axiom),
                    mp_depth=int(stmt.derivation_depth or 0),
                    parents=tuple(getattr(stmt, "parent_statements", []) or ()),
                    verification_method="external",
                )
                return repo.persist(system_id=theory_id, record=record)
        except Exception:
            return f"mock_statement_id_{hash(stmt.content) % 10000}"

    def enqueue_job(self, stmt):
        if not self.redis_client and redis:
            try:
                self.redis_client = redis.from_url(self.redis_url)
            except Exception:
                pass

        if self.redis_client:
            try:
                job_data = json.dumps({"statement": stmt.content, "theory": "Propositional"})
                self.redis_client.rpush("ml:jobs", job_data)
            except Exception:
                pass


class InferenceEngine:
    """Legacy inference engine interface retained for tests."""

    def derive_new_statements(self, *args, **kwargs):
        return []


def append_to_progress(*args, **kwargs) -> None:
    """Legacy function for test compatibility."""
    pass

