"""
Ledger Ingestion Module

Handles the atomic ingestion of statements, proofs, and the sealing of blocks.
Implements the core ledger logic for the Dual Root Attestation protocol.

See MathLedger whitepaper §4.2 (Dual Root Attestation) and §5.1 (Ledger Structure).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from psycopg.cursor import Cursor
from psycopg.types.json import Json

from attestation.dual_root import (
    compute_composite_root,
    compute_reasoning_root,
    compute_ui_root,
    generate_attestation_metadata,
)
from substrate.crypto.hashing import DOMAIN_ROOT, DOMAIN_STMT, hash_statement, sha256_hex
from normalization.canon import canonical_bytes, normalize
from normalization.proof import canonicalize_module_name, canonicalize_proof_text
from ledger.block_schema import seal_block_schema


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def verify_hash_contract(statement: str, observed_hash: str) -> bool:
    """
    Verify that a statement hash satisfies the canonical hash contract.
    
    The hash identity is: hash(s) = SHA256(DOMAIN_STMT || canonical_bytes(s))
    
    Args:
        statement: The statement text (raw or normalized)
        observed_hash: The hash to verify
        
    Returns:
        True if the hash matches the canonical computation
    """
    import hashlib
    canonical = canonical_bytes(statement)
    expected = hashlib.sha256(DOMAIN_STMT + canonical).hexdigest()
    return observed_hash == expected


def assert_hash_contract(statement: str, observed_hash: str, context: str = "") -> None:
    """
    Assert that a statement hash satisfies the canonical hash contract.
    
    Raises AssertionError if the hash contract is violated.
    
    Args:
        statement: The statement text
        observed_hash: The hash to verify
        context: Optional context for error messages
    """
    if not verify_hash_contract(statement, observed_hash):
        import hashlib
        canonical = canonical_bytes(statement)
        expected = hashlib.sha256(DOMAIN_STMT + canonical).hexdigest()
        raise AssertionError(
            f"Hash contract violation{' (' + context + ')' if context else ''}: "
            f"observed={observed_hash}, expected={expected}, "
            f"statement={statement!r}, normalized={normalize(statement)!r}"
        )


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.lower()).strip("-")
    return slug or "system"


def _canonical_json(data: Dict) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class StatementRecord:
    id: str
    hash: str
    normalized: str
    is_axiom: bool


@dataclass(frozen=True)
class ProofRecord:
    id: str
    hash: str
    statement: StatementRecord
    status: str


@dataclass(frozen=True)
class BlockRecord:
    id: int
    number: int
    reasoning_root: str
    ui_root: str
    composite_root: str
    block_hash: str


@dataclass(frozen=True)
class IngestOutcome:
    statement: StatementRecord
    proof: ProofRecord
    block: BlockRecord


class LedgerIngestor:
    """
    Deterministic ledger ingestion orchestrator.

    This class encapsulates all UPSERT, hashing, and block-sealing logic needed
    to persist statements, proofs, and sealed blocks with dual attestation.
    """

    def ingest_batch(
        self,
        cur: Cursor,
        *,
        theory_name: str,
        statements: Sequence[Tuple[str, str, str, str, int, bool]],  # (ascii, proof_text, status, rule, depth, is_axiom)
        prover: str,
        module_name: Optional[str] = None,
        sealed_by: str = "worker",
        ui_events: Optional[Sequence[str]] = None,
    ) -> BlockRecord:
        """
        Ingest a batch of statements/proofs and seal them into a single block.
        
        Args:
            statements: List of (ascii_statement, proof_text, status, rule, depth, is_axiom) tuples.
        """
        system_id, system_slug, run_id = self._ensure_system(cur, theory_name)
        
        proofs = []
        for ascii_stmt, proof_text, status, rule, depth, is_axiom in statements:
            stmt_record = self._upsert_statement(
                cur=cur,
                system_id=system_id,
                ascii_statement=ascii_stmt,
                status=status,
                derivation_rule=rule,
                derivation_depth=depth,
                truth_domain=None,
                is_axiom=is_axiom,
            )
            
            proof_record = self._upsert_proof(
                cur=cur,
                system_id=system_id,
                statement=stmt_record,
                proof_text=proof_text,
                prover=prover,
                status=status,
                module_name=module_name,
                stdout="",
                stderr="",
                derivation_rule=rule,
                derivation_depth=depth,
                method=None,
                duration_ms=None,
            )
            proofs.append(proof_record)
            
        if not proofs:
            raise ValueError("Cannot ingest empty batch")

        return self._seal_block(
            cur=cur,
            system_id=system_id,
            run_id=run_id,
            proofs=proofs,
            ui_events=ui_events or (),
            sealed_by=sealed_by,
        )

    def ingest(
        self,
        cur: Cursor,
        *,
        theory_name: str,
        ascii_statement: str,
        proof_text: str,
        prover: str,
        status: str,
        module_name: Optional[str],
        stdout: Optional[str],
        stderr: Optional[str],
        derivation_rule: Optional[str],
        derivation_depth: Optional[int],
        method: Optional[str] = None,
        duration_ms: Optional[int] = None,
        truth_domain: Optional[str] = None,
        is_axiom: bool = False,
        ui_events: Optional[Sequence[str]] = None,
        sealed_by: str = "worker",
    ) -> IngestOutcome:
        system_id, system_slug, run_id = self._ensure_system(cur, theory_name)

        statement = self._upsert_statement(
            cur=cur,
            system_id=system_id,
            ascii_statement=ascii_statement,
            status=status,
            derivation_rule=derivation_rule,
            derivation_depth=derivation_depth,
            truth_domain=truth_domain,
            is_axiom=is_axiom,
        )

        proof = self._upsert_proof(
            cur=cur,
            system_id=system_id,
            statement=statement,
            proof_text=proof_text,
            prover=prover,
            status=status,
            module_name=module_name,
            stdout=stdout,
            stderr=stderr,
            derivation_rule=derivation_rule,
            derivation_depth=derivation_depth,
            method=method,
            duration_ms=duration_ms,
        )

        block = self._seal_block(
            cur=cur,
            system_id=system_id,
            run_id=run_id,
            proofs=[proof],
            ui_events=ui_events or (),
            sealed_by=sealed_by,
        )

        return IngestOutcome(statement=statement, proof=proof, block=block)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_system(self, cur: Cursor, theory_name: str) -> Tuple[str, str, int]:
        slug = _slugify(theory_name)
        cur.execute(
            """
            INSERT INTO theories (name, slug)
            VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE
            SET slug = COALESCE(theories.slug, EXCLUDED.slug)
            RETURNING id, slug
            """,
            (theory_name, slug),
        )
        system_id, stored_slug = cur.fetchone()
        if not stored_slug:
            cur.execute(
                "UPDATE theories SET slug = %s WHERE id = %s RETURNING slug",
                (slug, system_id),
            )
            stored_slug = cur.fetchone()[0]

        run_id = self._ensure_run(cur, system_id, stored_slug)
        self._ensure_sequence(cur, system_id, run_id)
        return str(system_id), stored_slug, run_id

    def _ensure_run(self, cur: Cursor, system_id: str, slug: str) -> int:
        run_name = f"ledger::{slug}"
        cur.execute(
            """
            SELECT id
            FROM runs
            WHERE system_id = %s AND name = %s
            ORDER BY id
            LIMIT 1
            """,
            (system_id, run_name),
        )
        row = cur.fetchone()
        if row:
            return row[0]

        from substrate.repro.determinism import deterministic_timestamp_from_content
        started_at = deterministic_timestamp_from_content(run_name)
        cur.execute(
            """
            INSERT INTO runs (name, system_id, status, started_at)
            VALUES (%s, %s, 'running', %s)
            RETURNING id
            """,
            (run_name, system_id, started_at),
        )
        return cur.fetchone()[0]

    def _ensure_sequence(self, cur: Cursor, system_id: str, run_id: int) -> None:
        cur.execute(
            """
            INSERT INTO ledger_sequences (system_id, run_id)
            VALUES (%s, %s)
            ON CONFLICT (system_id) DO UPDATE
            SET run_id = COALESCE(ledger_sequences.run_id, EXCLUDED.run_id)
            """,
            (system_id, run_id),
        )

    def _upsert_statement(
        self,
        *,
        cur: Cursor,
        system_id: str,
        ascii_statement: str,
        status: str,
        derivation_rule: Optional[str],
        derivation_depth: Optional[int],
        truth_domain: Optional[str],
        is_axiom: bool,
    ) -> StatementRecord:
        normalized = normalize(ascii_statement)
        if not normalized:
            raise ValueError("Cannot ingest empty statement")

        # Hash contract: hash(s) = SHA256(DOMAIN_STMT || canonical_bytes(s))
        # This is enforced by hash_statement which calls canonical_bytes internally.
        statement_hash = hash_statement(normalized)
        
        # Defensive assertion: verify the hash is a valid 64-char hex string
        assert len(statement_hash) == 64, f"Invalid hash length: {len(statement_hash)}"
        assert all(c in "0123456789abcdef" for c in statement_hash), "Invalid hash chars"
        statement_status = "proven" if status == "success" or is_axiom else "unknown"

        cur.execute(
            """
            INSERT INTO statements (
                theory_id,
                system_id,
                hash,
                content,
                content_norm,
                normalized_text,
                status,
                derivation_rule,
                derivation_depth,
                truth_domain,
                is_axiom
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (hash) DO UPDATE
            SET content = COALESCE(statements.content, EXCLUDED.content),
                content_norm = COALESCE(statements.content_norm, EXCLUDED.content_norm),
                normalized_text = COALESCE(statements.normalized_text, EXCLUDED.normalized_text),
                system_id = EXCLUDED.system_id,
                theory_id = EXCLUDED.theory_id,
                status = CASE
                    WHEN statements.status = 'unknown' AND EXCLUDED.status <> 'unknown'
                        THEN EXCLUDED.status
                    ELSE statements.status
                END,
                derivation_rule = COALESCE(statements.derivation_rule, EXCLUDED.derivation_rule),
                derivation_depth = CASE
                    WHEN statements.derivation_depth IS NULL THEN EXCLUDED.derivation_depth
                    WHEN EXCLUDED.derivation_depth IS NULL THEN statements.derivation_depth
                    ELSE LEAST(statements.derivation_depth, EXCLUDED.derivation_depth)
                END,
                truth_domain = COALESCE(statements.truth_domain, EXCLUDED.truth_domain),
                is_axiom = statements.is_axiom OR EXCLUDED.is_axiom
            RETURNING id, is_axiom
            """,
            (
                system_id,
                system_id,
                statement_hash,
                ascii_statement,
                normalized,
                ascii_statement,
                statement_status,
                derivation_rule,
                derivation_depth,
                truth_domain,
                is_axiom,
            ),
        )
        row_id, resolved_axiom = cur.fetchone()
        return StatementRecord(
            id=str(row_id),
            hash=statement_hash,
            normalized=normalized,
            is_axiom=bool(resolved_axiom),
        )

    def _upsert_proof(
        self,
        *,
        cur: Cursor,
        system_id: str,
        statement: StatementRecord,
        proof_text: str,
        prover: str,
        status: str,
        module_name: Optional[str],
        stdout: Optional[str],
        stderr: Optional[str],
        derivation_rule: Optional[str],
        derivation_depth: Optional[int],
        method: Optional[str],
        duration_ms: Optional[int],
    ) -> ProofRecord:
        normalized_proof_text = proof_text or ""
        canonical_proof_text = canonicalize_proof_text(normalized_proof_text)
        payload = {
            "statement_hash": statement.hash,
            "prover": prover or "",
            "status": status or "",
            "proof_text": canonical_proof_text,
            "module": canonicalize_module_name(module_name),
            "derivation_rule": derivation_rule or "",
        }
        payload_json = _canonical_json(payload)
        proof_hash = sha256_hex(payload_json, domain=DOMAIN_ROOT)
        success = status == "success"

        cur.execute(
            """
            INSERT INTO proofs (
                statement_id,
                system_id,
                prover,
                method,
                status,
                proof_text,
                module_name,
                stdout,
                stderr,
                proof_hash,
                success,
                derivation_rule,
                derivation_depth,
                duration_ms
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (statement_id, prover, proof_hash) DO UPDATE
            SET status = EXCLUDED.status,
                proof_text = CASE
                    WHEN EXCLUDED.status = 'success' THEN EXCLUDED.proof_text
                    ELSE proofs.proof_text
                END,
                module_name = COALESCE(EXCLUDED.module_name, proofs.module_name),
                stdout = COALESCE(EXCLUDED.stdout, proofs.stdout),
                stderr = COALESCE(EXCLUDED.stderr, proofs.stderr),
                success = EXCLUDED.success,
                derivation_rule = COALESCE(proofs.derivation_rule, EXCLUDED.derivation_rule),
                derivation_depth = CASE
                    WHEN proofs.derivation_depth IS NULL THEN EXCLUDED.derivation_depth
                    WHEN EXCLUDED.derivation_depth IS NULL THEN proofs.derivation_depth
                    ELSE LEAST(proofs.derivation_depth, EXCLUDED.derivation_depth)
                END,
                duration_ms = COALESCE(EXCLUDED.duration_ms, proofs.duration_ms)
            RETURNING id
            """,
            (
                statement.id,
                system_id,
                prover,
                method,
                status,
                normalized_proof_text,
                module_name,
                stdout,
                stderr,
                proof_hash,
                success,
                derivation_rule,
                derivation_depth,
                duration_ms,
            ),
        )
        proof_id = cur.fetchone()[0]
        return ProofRecord(
            id=str(proof_id),
            hash=proof_hash,
            statement=statement,
            status=status,
        )

    def _seal_block(
        self,
        *,
        cur: Cursor,
        system_id: str,
        run_id: int,
        proofs: Sequence[ProofRecord],
        ui_events: Sequence[str],
        sealed_by: str,
    ) -> BlockRecord:
        if not proofs:
            raise ValueError("Cannot seal a block without proofs")

        cur.execute(
            """
            SELECT system_id, height, prev_block_id, prev_block_hash, prev_composite_root, run_id
            FROM ledger_sequences
            WHERE system_id = %s
            FOR UPDATE
            """,
            (system_id,),
        )
        seq_row = cur.fetchone()
        if not seq_row:
            cur.execute(
                """
                INSERT INTO ledger_sequences (system_id, run_id, height, prev_block_id, prev_block_hash, prev_composite_root)
                VALUES (%s, %s, 0, NULL, NULL, NULL)
                RETURNING system_id, height, prev_block_id, prev_block_hash, prev_composite_root, run_id
                """,
                (system_id, run_id),
            )
            seq_row = cur.fetchone()

        _, height, prev_block_id, prev_block_hash, prev_composite_root, stored_run_id = seq_row
        if stored_run_id is None:
            stored_run_id = run_id

        block_number = (height or 0) + 1
        prev_hash = prev_composite_root or prev_block_hash

        statement_map: Dict[str, str] = {}
        for proof in proofs:
            statement_map.setdefault(proof.statement.hash, proof.statement.id)
        statement_hashes = sorted(statement_map.keys())

        proof_entries = sorted(proofs, key=lambda p: (p.hash, p.statement.hash))
        reasoning_inputs = [p.hash for p in proof_entries]
        reasoning_root = compute_reasoning_root(reasoning_inputs)

        ui_inputs = sorted(set(str(ev) for ev in ui_events))
        ui_root = compute_ui_root(ui_inputs)
        composite_root = compute_composite_root(reasoning_root, ui_root)

        sealed_schema = seal_block_schema(
            system_id=system_id,
            block_number=block_number,
            reasoning_root=reasoning_root,
            ui_root=ui_root,
            composite_root=composite_root,
            statements=statement_hashes,
            proofs=[
                {
                    "hash": proof.hash,
                    "statement_hash": proof.statement.hash,
                    "status": proof.status,
                }
                for proof in proof_entries
            ],
            prev_hash=prev_hash,
        )
        statement_hashes = sealed_schema.statements
        sealed_at = sealed_schema.sealed_at
        payload_hash = sealed_schema.payload_hash
        block_hash = sealed_schema.block_hash

        attestation_metadata = generate_attestation_metadata(
            r_t=reasoning_root,
            u_t=ui_root,
            h_t=composite_root,
            reasoning_event_count=len(proof_entries),
            ui_event_count=len(ui_inputs),
            extra={
                "sealed_by": sealed_by,
                "payload_hash": payload_hash,
            },
        )

        cur.execute(
            """
            INSERT INTO blocks (
                run_id,
                system_id,
                block_number,
                prev_hash,
                prev_block_id,
                root_hash,
                merkle_root,
                header,
                statements,
                canonical_statements,
                canonical_proofs,
                statement_count,
                proof_count,
                reasoning_merkle_root,
                ui_merkle_root,
                composite_attestation_root,
                attestation_metadata,
                sealed_at,
                sealed_by,
                payload_hash,
                block_hash
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id
            """,
            (
                stored_run_id,
                system_id,
                block_number,
                prev_hash,
                prev_block_id,
                reasoning_root,
                reasoning_root,
                Json(sealed_schema.header),
                Json(sealed_schema.statements),
                Json(sealed_schema.canonical_statements),
                Json(sealed_schema.canonical_proofs),
                sealed_schema.header["statement_count"],
                sealed_schema.header["proof_count"],
                reasoning_root,
                ui_root,
                composite_root,
                Json(attestation_metadata),
                sealed_at,
                sealed_by,
                payload_hash,
                block_hash,
            ),
        )
        block_id = cur.fetchone()[0]

        self._persist_block_links(
            cur=cur,
            block_id=block_id,
            statement_hashes=statement_hashes,
            statement_map=statement_map,
            proofs=proof_entries,
        )

        # DETERMINISM: Use the same sealed_at timestamp for updated_at
        # instead of SQL NOW() to ensure reproducibility
        cur.execute(
            """
            UPDATE ledger_sequences
            SET height = %s,
                prev_block_id = %s,
                prev_block_hash = %s,
                prev_composite_root = %s,
                run_id = COALESCE(run_id, %s),
                updated_at = %s
            WHERE system_id = %s
            """,
            (
                block_number,
                block_id,
                block_hash,
                composite_root,
                stored_run_id,
                sealed_at,
                system_id,
            ),
        )

        return BlockRecord(
            id=block_id,
            number=block_number,
            reasoning_root=reasoning_root,
            ui_root=ui_root,
            composite_root=composite_root,
            block_hash=block_hash,
        )

    def _persist_block_links(
        self,
        *,
        cur: Cursor,
        block_id: int,
        statement_hashes: Sequence[str],
        statement_map: Dict[str, str],
        proofs: Sequence[ProofRecord],
    ) -> None:
        for position, statement_hash in enumerate(statement_hashes):
            statement_id = statement_map[statement_hash]
            cur.execute(
                """
                INSERT INTO block_statements (block_id, position, statement_id, statement_hash)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (block_id, statement_id) DO NOTHING
                """,
                (block_id, position, statement_id, statement_hash),
            )

        for position, proof in enumerate(proofs):
            cur.execute(
                """
                INSERT INTO block_proofs (block_id, position, proof_id, proof_hash, statement_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (block_id, proof_id) DO NOTHING
                """,
                (block_id, position, proof.id, proof.hash, proof.statement.id),
            )

