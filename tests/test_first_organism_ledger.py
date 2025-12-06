import os
import subprocess

import psycopg
import pytest

from attestation.dual_root import (
    compute_reasoning_root,
    compute_ui_root,
    compute_composite_root,
)
from backend.security.runtime_env import MissingEnvironmentVariable
from ledger.first_organism import ingest_and_seal_for_first_organism
from tests.conftest import _run_migrations_once


def _resolve_database_url() -> str | None:
    return os.environ.get("DATABASE_URL_TEST") or os.environ.get("DATABASE_URL")


@pytest.fixture(scope="module")
def first_organism_cursor():
    db_url = _resolve_database_url()
    if not db_url:
        pytest.skip("Set DATABASE_URL or DATABASE_URL_TEST to run the First Organism ledger contract test.")

    try:
        connectivity_conn = psycopg.connect(db_url, connect_timeout=2)
    except psycopg.Error as exc:
        pytest.skip(f"Postgres not reachable at {db_url}: {exc}")
    finally:
        if "connectivity_conn" in locals():
            connectivity_conn.close()

    try:
        _run_migrations_once()
    except (subprocess.CalledProcessError, MissingEnvironmentVariable) as exc:
        pytest.skip(f"Unable to prepare ledger schema ({exc}); skipping First Organism contract test.")

    conn = psycopg.connect(db_url, connect_timeout=2)
    cur = conn.cursor()
    try:
        yield cur
    finally:
        conn.rollback()
        cur.close()
        conn.close()


@pytest.mark.integration
def test_first_organism_ingestion_contract(first_organism_cursor):
    """
    Verifies the First Organism ingestion contract:
    1. Ingests a proof + UI events.
    2. Computes R_t, U_t, and H_t from the stored block.
    3. Validates the composite root binding H_t = SHA256(R_t || U_t).
    """
    proof_payload = {
        "theory": "first_organism_test",
        "statement": "forall x, x = x",
        "proof": "reflexivity",
        "prover": "lean_checker",
        "status": "success",
    }
    ui_events = ["event_1", "event_2"]

    sealed_block = ingest_and_seal_for_first_organism(first_organism_cursor, proof_payload, ui_events)

    first_organism_cursor.execute(
        "SELECT proof_hash FROM block_proofs WHERE block_id = %s ORDER BY proof_hash",
        (sealed_block.block_id,),
    )
    proof_rows = first_organism_cursor.fetchall()
    proof_hashes = [row[0] for row in proof_rows]
    assert proof_hashes, "Sealed block must record at least one proof."

    computed_reasoning_root = compute_reasoning_root(proof_hashes)
    assert sealed_block.reasoning_root == computed_reasoning_root

    canonicalized_ui_events = sorted({str(ev) for ev in ui_events})
    computed_ui_root = compute_ui_root(canonicalized_ui_events)
    assert sealed_block.ui_root == computed_ui_root

    computed_composite_root = compute_composite_root(computed_reasoning_root, computed_ui_root)
    assert sealed_block.composite_root == computed_composite_root

    assert sealed_block.block_id > 0
    assert sealed_block.sequence > 0
    assert sealed_block.timestamp

