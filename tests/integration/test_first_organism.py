"""
First Organism Integration Test - Definitive Implementation
============================================================

This test proves the First Organism closed loop as described in the whitepaper:

    UI Event → Curriculum Gate → Derivation → Lean Verify (abstention) →
    Dual-Attest seal H_t → RFL runner metabolism.

Test Decomposition (5 phases):
    1. test_first_organism_ui_event_capture - UI event capture and U_t computation
    2. test_first_organism_curriculum_gate - Curriculum gate evaluation
    3. test_first_organism_derivation_and_abstention - Derivation pipeline with abstention
    4. test_first_organism_dual_attestation_seal - Dual-root attestation (R_t, U_t, H_t)
    5. test_first_organism_rfl_metabolism - RFL runner metabolism verification

MDAP Compliance:
    - All randomness sourced from deterministic helpers
    - All timestamps derived from content hashes
    - No direct calls to datetime.now(), time.time(), or uuid.uuid4()

Certification:
    - Emits "[PASS] FIRST ORGANISM ALIVE H_t=<short_hash>" on success
    - Writes artifacts/first_organism/attestation.json for Cursor P

Test Gating (SPARK):
    - SPARK tests = First Organism tests (marked with @pytest.mark.first_organism)
    - Enable with: FIRST_ORGANISM_TESTS=true, SPARK_RUN=1, or .spark_run_enable file
    - Hermetic tests (standalone, determinism) run even without DB
    - DB-dependent tests (happy_path) skip gracefully when Postgres/Redis are down

Usage:
    FIRST_ORGANISM_TESTS=true pytest tests/integration/test_first_organism.py -v
    FIRST_ORGANISM_TESTS=true pytest tests/integration/test_first_organism.py -k ui_event
    SPARK_RUN=1 pytest -m first_organism -v
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence

import sys

import pytest

# Import from canonical attestation module (single source of truth)
from attestation.dual_root import (
    compute_composite_root,
    compute_reasoning_root,
    compute_ui_root,
    verify_composite_integrity,
)

# Canonical hash pipeline imports for hash contract assertions
# Binding: This test is bound to the hash contract in tests/test_canon.py
# The identity `hash(s) = SHA256(DOMAIN_STMT || canonical_bytes(s))` must hold.
from normalization.canon import canonical_bytes, normalize
from backend.crypto.hashing import DOMAIN_STMT, hash_statement

# Import from canonical ledger modules
from ledger.blocking import seal_block_with_dual_roots
from ledger.ingest import LedgerIngestor
from ledger.ui_events import (
    capture_ui_event,
    consume_ui_artifacts,
    materialize_ui_artifacts,
    snapshot_ui_events,
    ui_event_store,
)

# Backend modules
from derivation.bounds import SliceBounds
from derivation.pipeline import DerivationPipeline
from derivation.verification import StatementVerifier
from curriculum.gates import (
    CurriculumSystem,
    GateEvaluator,
    NormalizedMetrics,
    make_first_organism_slice,
    should_ratchet,
)
from rfl.bootstrap_stats import BootstrapResult, verify_metabolism
from rfl.config import RFLConfig
from rfl.experiment import ExperimentResult
from rfl.runner import RFLRunner, RunLedgerEntry

# Derivation/RFL modules
from derivation.pipeline import (
    make_first_organism_derivation_config,
    make_first_organism_derivation_slice,
    make_first_organism_seed_statements,
    run_slice_for_test,
)
from rfl.config import CurriculumSlice as RFLCurriculumSlice
from rfl.runner import RFLRunner as RFLRunnerCanonical

# Substrate: Determinism helpers
from substrate.repro.determinism import (
    deterministic_hash,
    deterministic_seed_from_content,
)
from substrate.bridge.context import AttestedRunContext

# Local fixtures and helpers
from tests.integration.conftest import (
    FirstOrganismAttestation,
    MDAP_EPOCH_SEED,
    build_first_organism_attestation,
    log_first_organism_pass,
    log_first_organism_skip,
    mdap_deterministic_id,
    mdap_deterministic_timestamp,
)


pytestmark = [pytest.mark.integration, pytest.mark.first_organism]


# ---------------------------------------------------------------------------
# Harness Printer (Green Bar)
# ---------------------------------------------------------------------------

class HarnessPrinter:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    
    # Check if stdout is a TTY for color support
    USE_COLOR = sys.stdout.isatty() or os.environ.get("PYTEST_COLOR") == "yes"

    @classmethod
    def print_status(cls, phase: str, message: str, status: str = "INFO"):
        """
        Print a status message with optional color coding.
        
        Args:
            phase: The phase of the pipeline (e.g., "GATE", "ATTEST").
            message: The message to print.
            status: "PASS", "FAIL", "WARN", or "INFO".
        """
        color = ""
        if cls.USE_COLOR:
            if status == "PASS":
                color = cls.GREEN
            elif status == "FAIL":
                color = cls.RED
            elif status == "WARN":
                color = cls.YELLOW

        reset = cls.RESET if cls.USE_COLOR else ""
        # Format: [PHASE] Message
        sys.stdout.write(f"{color}[{phase}] {message}{reset}\n")
        sys.stdout.flush()

def harness_print_status(phase: str, message: str, status: str = "INFO"):
    """Wrapper for HarnessPrinter for backward compatibility."""
    HarnessPrinter.print_status(phase, message, status)


# ---------------------------------------------------------------------------
# Shared Test State (module-scoped for phase reuse)
# ---------------------------------------------------------------------------
class FirstOrganismTestState:
    """Shared state across test phases."""

    def __init__(self):
        self.ui_event = None
        self.ui_event_id = None
        self.ui_snapshot = None
        self.slice_cfg = None
        self.gate_statuses = None
        self.derivation_result = None
        self.candidate = None
        self.block = None
        self.attestation = None
        self.rfl_runner = None
        self.rfl_result = None

    def reset(self):
        """Reset state for fresh test run."""
        self.__init__()

    def log_phase(self, phase: str, message: str, status: str = "INFO"):
        """Emit investor-grade phase status for the current phase."""
        harness_print_status(phase, message, status)


# Module-level state instance
_state = FirstOrganismTestState()


@pytest.fixture(scope="module", autouse=True)
def reset_module_state():
    """Reset state at module start."""
    _state.reset()
    ui_event_store.clear()
    yield
    ui_event_store.clear()


# ---------------------------------------------------------------------------
# Hash Contract Assertion (bound to tests/test_canon.py)
# ---------------------------------------------------------------------------


def assert_hash_contract(statement: str, observed_hash: str, context: str = "") -> None:
    """
    Assert that the observed hash matches the canonical hash contract.
    
    The hash identity `hash(s) = SHA256(DOMAIN_STMT || canonical_bytes(s))` must hold.
    This function is the integration test's binding to the unit test contract in
    tests/test_canon.py::TestCanonicalHashing.
    
    Args:
        statement: The statement text (may be raw or normalized)
        observed_hash: The hash value observed in the ledger/attestation
        context: Optional context string for error messages
    
    Raises:
        AssertionError: If the hash contract is violated
    """
    import hashlib
    
    # Compute expected hash using the canonical pipeline
    canonical = canonical_bytes(statement)
    expected_hash = hashlib.sha256(DOMAIN_STMT + canonical).hexdigest()
    
    # Also verify via hash_statement for consistency
    hash_via_helper = hash_statement(statement)
    
    assert expected_hash == hash_via_helper, (
        f"Hash helper inconsistency: expected={expected_hash}, hash_statement={hash_via_helper}"
    )
    
    assert observed_hash == expected_hash, (
        f"Hash contract violation{' (' + context + ')' if context else ''}: "
        f"observed={observed_hash}, expected={expected_hash}, "
        f"statement={statement!r}, normalized={normalize(statement)!r}"
    )


def get_statement_hash_for_verification(statement: str) -> str:
    """
    Compute the canonical hash for a statement.
    
    This is a test-friendly helper that exposes the canonical hash computation
    for verification purposes.
    
    Args:
        statement: The statement text
        
    Returns:
        64-character hex hash
    """
    return hash_statement(statement)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _write_attestation_artifact(
    attestation: FirstOrganismAttestation,
    extra_metadata: Dict = None,
) -> Path:
    """
    Write canonical attestation artifact for external auditing.

    This is the single JSON file that Cursor P relies on to certify Wave-1.

    Args:
        attestation: FirstOrganismAttestation instance
        extra_metadata: Optional additional fields to include

    Returns:
        Path to written artifact
    """
    artifact_dir = Path("artifacts/first_organism")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "attestation.json"

    payload = attestation.to_dict()
    if extra_metadata:
        payload["extra"] = extra_metadata

    # Add component versions for auditing
    payload["components"] = {
        "derivation": "axiom_engine",
        "ledger": "LedgerIngestor",
        "attestation": "attestation.dual_root",
        "rfl": "RFLRunner",
    }

    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path


def _assert_gate_statuses(gate_statuses):
    """Assert all curriculum gates passed."""
    failed = [status for status in gate_statuses if not status.passed]
    assert not failed, f"Curriculum gate rejected the slice: {[status.gate for status in failed]}"


def _assert_block_linkages(db_conn, block_id, statement_hash):
    """Assert block linkages are correctly established in the database."""
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT statement_hash FROM block_statements WHERE block_id = %s ORDER BY position",
            (block_id,),
        )
        block_statements = [row[0] for row in cur.fetchall()]
        cur.execute(
            "SELECT proof_hash, status FROM block_proofs WHERE block_id = %s",
            (block_id,),
        )
        block_proofs = cur.fetchall()
    assert block_statements == [statement_hash]
    assert block_proofs and block_proofs[0][1] in {"failure", "abstain", "timeout"}


def _assert_composite_root_recomputable(block):
    """
    Assert that H_t can be recomputed from stored R_t and U_t.

    This is the H_t Invariant Warden check: the canonical compute_composite_root
    function must produce the same result as the stored composite_root.
    """
    assert block.reasoning_root, "R_t must be non-empty"
    assert block.ui_merkle_root, "U_t must be non-empty"
    assert block.composite_root, "H_t must be non-empty"

    # Recompute using canonical function from attestation.dual_root
    recomputed = compute_composite_root(block.reasoning_root, block.ui_merkle_root)
    assert recomputed == block.composite_root, (
        f"H_t Invariant violated: recomputed={recomputed}, stored={block.composite_root}"
    )

    # Also verify using the integrity check helper
    assert verify_composite_integrity(
        block.reasoning_root, block.ui_merkle_root, block.composite_root
    ), "verify_composite_integrity failed"


def _seal_single_block(
    proof_payload: dict,
    *,
    ui_artifacts: Optional[Sequence[str]] = None,
) -> dict:
    """
    Seal a single-proof block while explicitly managing the UI event stream.

    Artifacts can be supplied directly or consumed from the live store. After
    sealing, the UI event store is cleared to enforce the no-residue invariant.

    Reference: MathLedger Whitepaper §4.2 (Dual Attestation Block Sealing).
    """
    artifacts = list(ui_artifacts) if ui_artifacts is not None else consume_ui_artifacts()
    block = seal_block_with_dual_roots("pl", [proof_payload], ui_events=artifacts)
    ui_event_store.clear()
    return block


def _run_rfl_metabolism(block, candidate_hash, slice_name, monkeypatch):
    """
    Run RFL metabolism step with synthetic experiment result.

    Returns:
        Tuple of (runner, result) for assertion
    """
    config = RFLConfig(
        experiment_id="first-organism-test",
        num_runs=2,
        random_seed=42,
        system_id=1,
        derive_steps=1,
        max_breadth=1,
        max_total=1,
        depth_max=1,
        bootstrap_replicates=1000,
        coverage_threshold=0.0,
        uplift_threshold=0.0,
        dual_attestation=False,
        curriculum=[
            RFLCurriculumSlice(
                name=slice_name,
                start_run=1,
                end_run=2,
                derive_steps=1,
                max_breadth=1,
                max_total=1,
                depth_max=1,
            )
        ],
    )
    monkeypatch.setattr("rfl.runner.load_baseline_from_db", lambda *args, **kwargs: [])
    runner = RFLRunner(config)
    result = ExperimentResult(
        run_id=f"rfl-{slice_name}-{block.id}",
        system_id=1,
        start_time="2025-01-01T00:00:00Z",
        end_time="2025-01-01T00:01:00Z",
        duration_seconds=60.0,
        total_statements=1,
        successful_proofs=0,
        failed_proofs=0,
        abstentions=1,
        throughput_proofs_per_hour=0.0,
        mean_depth=0.0,
        max_depth=0,
        statement_hashes=[candidate_hash],
        derive_steps=1,
        max_breadth=1,
        max_total=1,
        policy_context={
            "slice": slice_name,
            "attestation_root": block.composite_root,
            "reasoning_root": block.reasoning_root,
            "ui_root": block.ui_merkle_root,
            "block_id": block.id,
        },
        abstention_breakdown={"lean_failure": 1},
        status="failed",
        error_message="synthetic abstention",
    )
    runner.run_results.append(result)
    runner._merge_abstention_breakdown(result.abstention_breakdown)
    runner._record_policy_entry(result, config.curriculum[0])
    return runner, result


# ---------------------------------------------------------------------------
# Tests using fixtures (first_organism_db, etc.)
# ---------------------------------------------------------------------------


@pytest.mark.first_organism
@pytest.mark.requires_db  # Requires database connection
def test_first_organism_chain_integrity(
    first_organism_db,
    first_organism_env,
    first_organism_attestation_context,
):
    """
    Test that block chain integrity is maintained with H_t recomputability.

    This test validates:
    1. Curriculum gates pass for the configured slice
    2. Block linkages are correctly established
    3. H_t can be recomputed from R_t and U_t

    Whitepaper chain mapping:
    - Derivation → LedgerIngestor → DualAttest(H_t) → Integrity check
    """
    context = first_organism_attestation_context
    attestation: FirstOrganismAttestation = context["attestation"]

    # Step 1: Curriculum gate verification
    _assert_gate_statuses(context["gate_statuses"])
    harness_print_status("GATE", f"Curriculum gates passed for {context['curriculum_slice_name']}", "PASS")

    # Step 2: Block integrity verification
    block = context["block"]
    candidate_hash = context["candidate_hash"]
    _assert_composite_root_recomputable(block)
    harness_print_status("ATTEST", f"H_t recomputable: {attestation.short_h_t()}", "PASS")

    # Step 3: Database linkage verification
    _assert_block_linkages(first_organism_db, block.id, candidate_hash)
    harness_print_status("LEDGER", f"Block {block.id} linkages verified", "PASS")

    # Write attestation artifact
    artifact_path = _write_attestation_artifact(
        attestation,
        extra_metadata={"test": "chain_integrity"},
    )
    harness_print_status("ARTIFACT", f"Written to {artifact_path}", "INFO")

    # Emit canonical PASS message
    log_first_organism_pass(attestation.composite_root)


@pytest.mark.first_organism
@pytest.mark.requires_db  # Requires database connection
def test_first_organism_closed_loop_happy_path(
    first_organism_db,
    first_organism_env,
    first_organism_attestation_context,
    monkeypatch,
):
    """
    Test the full closed loop with RFL metabolism verification.

    This test validates the complete whitepaper chain:
    1. UI Event (U_t) → captured and hashed
    2. Curriculum Gate → slice allowed
    3. Derivation → candidate produced
    4. Lean Verify → abstention/failure
    5. LedgerIngestor → block sealed with R_t, U_t, H_t
    6. RFL Runner → H_t consumed, policy updated

    Certification:
    - Emits "[PASS] FIRST ORGANISM ALIVE H_t=<short_hash>"
    - Writes artifacts/first_organism/attestation.json

    Skip Reason Visibility:
    If this test is SKIPPED, search for `[SKIP][FO]` in SPARK_run_log.txt (or pytest output)
    for the exact reason and remediation. All skip messages follow the format:
    [SKIP][FO] <precise reason> (mode=<mode>, db_url=<trimmed-url>)
    
    Common skip reasons:
    - FIRST_ORGANISM_TESTS not enabled: Set FIRST_ORGANISM_TESTS=true or SPARK_RUN=1
    - Postgres unreachable: Run scripts/start_first_organism_infra.ps1
    - Migration missing: Ensure migrations/016_monotone_ledger.sql exists

    Debug Mode:
    Set FIRST_ORGANISM_DEBUG=1 to print environment diagnostics before test execution.
    """
    # Debug mode: Print environment diagnostics
    if os.getenv("FIRST_ORGANISM_DEBUG", "").strip() == "1":
        from tests.integration.conftest import detect_environment_mode, _mask_password
        
        print()
        print("=" * 70)
        print("[DEBUG] First Organism Test Environment Diagnostics")
        print("=" * 70)
        
        # Resolved DATABASE_URL
        db_url = os.getenv(
            "DATABASE_URL_TEST",
            os.getenv(
                "DATABASE_URL",
                "postgresql://ml:mlpass@127.0.0.1:5433/mathledger?connect_timeout=5",
            ),
        )
        db_url_masked = _mask_password(db_url)
        print(f"  DATABASE_URL (resolved): {db_url_masked}")
        
        # EnvironmentMode
        env_mode = detect_environment_mode()
        print(f"  EnvironmentMode.chain_status: {env_mode.chain_status.value}")
        print(f"  EnvironmentMode.db_status: {env_mode.db_status.value}")
        print(f"  EnvironmentMode.redis_status: {env_mode.redis_status.value}")
        if env_mode.skip_reason:
            print(f"  Skip reason: {env_mode.skip_reason}")
        if env_mode.abstain_reason:
            print(f"  Abstain reason: {env_mode.abstain_reason}")
        
        # FIRST_ORGANISM_TESTS status
        first_org_tests = os.getenv("FIRST_ORGANISM_TESTS", "")
        spark_run = os.getenv("SPARK_RUN", "")
        spark_file = Path(".spark_run_enable").exists()
        print(f"  FIRST_ORGANISM_TESTS: {first_org_tests or '(not set)'}")
        print(f"  SPARK_RUN: {spark_run or '(not set)'}")
        print(f"  .spark_run_enable file exists: {spark_file}")
        
        print("=" * 70)
        print()
    
    context = first_organism_attestation_context
    attestation: FirstOrganismAttestation = context["attestation"]

    # Step 1: Validate curriculum gates
    _assert_gate_statuses(context["gate_statuses"])
    harness_print_status("GATE", f"Slice {context['curriculum_slice_name']} allowed", "PASS")

    # Step 2: Extract block and attestation data
    block = context["block"]
    candidate_hash = context["candidate_hash"]
    ui_event_id = context["ui_event_id"]
    slice_name = context["curriculum_slice_name"]

    harness_print_status("DERIVE", f"Candidate {candidate_hash[:12]}...", "INFO")
    harness_print_status("ATTEST", f"R_t={attestation.reasoning_root[:12]}...", "INFO")
    harness_print_status("ATTEST", f"U_t={attestation.ui_root[:12]}...", "INFO")
    harness_print_status("ATTEST", f"H_t={attestation.short_h_t()}", "PASS")

    # Step 3: RFL metabolism - use run_with_attestation for full flow
    monkeypatch.setattr("rfl.runner.load_baseline_from_db", lambda *args, **kwargs: [])
    from rfl.config import RFLConfig as RFLConfigCanonical
    rfl_config = RFLConfigCanonical(
        experiment_id="first-organism-closed-loop",
        num_runs=1,
        random_seed=42,
        system_id=1,
        derive_steps=1,
        max_breadth=1,
        max_total=1,
        depth_max=1,
        bootstrap_replicates=1000,
        coverage_threshold=0.0,
        uplift_threshold=0.0,
        dual_attestation=False,
        curriculum=[
            RFLCurriculumSlice(
                name=slice_name,
                start_run=1,
                end_run=1,
                derive_steps=1,
                max_breadth=1,
                max_total=1,
                depth_max=1,
            )
        ],
    )
    runner = RFLRunnerCanonical(rfl_config)
    
    # Create AttestedRunContext from block
    attestation_context = AttestedRunContext(
        slice_id=slice_name,
        statement_hash=candidate_hash,
        proof_status="failure",  # Abstained proof
        block_id=block.id,
        composite_root=block.composite_root,
        reasoning_root=block.reasoning_root,
        ui_root=block.ui_merkle_root,
        abstention_metrics={
            "rate": 1.0,
            "mass": 1.0,
            "counts": {"verified": 0, "rejected": 1, "considered": 1},
            "reasons": {"lean_failure": 1},
        },
        policy_id="first-organism-policy",
        metadata={
            "attempt_mass": 1.0,
            "abstention_breakdown": {"lean_failure": 1},
            "first_organism_abstentions": 1,
        },
    )
    
    # Call run_with_attestation - this is the critical path
    rfl_result = runner.run_with_attestation(attestation_context)
    
    # Verify RFL result
    assert rfl_result.source_root == block.composite_root, "RFL must reference H_t"
    assert rfl_result.policy_update_applied is True, "Policy should be updated"
    
    # Verify ledger entry
    assert len(runner.policy_ledger) > 0, "Policy ledger should have entries"
    ledger_entry = runner.policy_ledger[-1]
    assert ledger_entry.status == "attestation", f"Expected 'attestation', got {ledger_entry.status}"
    assert runner.abstention_histogram.get("lean_failure", 0) >= 1, "Abstention histogram should record lean failure"
    
    # Requested output: RFL summary
    summary_msg = (
        f"Metabolism complete, H_t consumed: {attestation.short_h_t()}\n"
        f"       Policy Update: {'Yes' if rfl_result.policy_update_applied else 'No'}\n"
        f"       Abstentions: {runner.abstention_histogram}\n"
        f"       Symbolic Descent: {ledger_entry.symbolic_descent:.4f}"
    )
    harness_print_status("RFL", summary_msg, "PASS")

    # Step 4: Write attestation artifact
    artifact_path = _write_attestation_artifact(
        attestation,
        extra_metadata={
            "test": "closed_loop_happy_path",
            "rfl_status": ledger_entry.status,
            "abstention_histogram": runner.abstention_histogram,
        },
    )
    assert artifact_path.exists(), f"Attestation artifact not written: {artifact_path}"
    harness_print_status("ARTIFACT", f"Written to {artifact_path}", "INFO")

    # Step 5: Emit canonical PASS message
    # This is the single, unmistakable line that Cursor P looks for
    log_first_organism_pass(attestation.composite_root)


@pytest.mark.first_organism
@pytest.mark.first_organism_smoke
@pytest.mark.hermetic  # This test does NOT require DB/Redis
def test_first_organism_closed_loop_smoke(monkeypatch):
    """
    DB-optional smoke test for First Organism closed loop.
    
    This test verifies the logical FO + RFL pipeline wiring without requiring
    a live database. It's useful when DB migrations are broken but you still
    need to verify that the code path is intact.
    
    This test:
    1. Runs derivation pipeline (no DB needed)
    2. Seals block with dual roots (no DB needed)
    3. Mocks LedgerIngestor.ingest to skip DB writes
    4. Builds AttestedRunContext from sealed block
    5. Calls run_with_attestation() and verifies H_t and status
    
    Usage:
        # If DB migrations are still broken, run:
        uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_smoke -v -s
        
        # To verify that the logical FO + RFL pipeline is intact.
    
    Differences from happy_path:
        - No first_organism_db fixture (DB-free)
        - LedgerIngestor.ingest is mocked to no-op
        - Block is sealed in-memory, not persisted
        - AttestedRunContext is built from sealed block dict, not DB record
    """
    # Clear UI event store
    ui_event_store.clear()
    
    # Step 1: UI Event Capture
    seed_content = "smoke-test-ui-event"
    event_id = mdap_deterministic_id("ui-event", seed_content)
    event_timestamp = mdap_deterministic_timestamp(seed_content)
    ui_event = {
        "event_id": event_id,
        "event_type": "select_statement",
        "actor": "smoke-test",
        "statement_hash": deterministic_hash("p -> p"),
        "action": "toggle_abstain",
        "meta": {"origin": "smoke-test"},
        "timestamp": event_timestamp,
    }
    capture_ui_event(ui_event)
    harness_print_status("SMOKE", "UI event captured", "INFO")
    
    # Step 2: Curriculum Gate (passing metrics)
    slice_cfg = make_first_organism_slice()
    metrics_raw = {
        "metrics": {
            "coverage": {"ci_lower": 0.95, "sample_size": 24},
            "proofs": {"abstention_rate": 12.0, "attempt_mass": 3200},
            "curriculum": {
                "active_slice": {"wallclock_minutes": 45.0, "proof_velocity_cv": 0.05}
            },
            "throughput": {
                "proofs_per_hour": 240.0,
                "coefficient_of_variation": 0.04,
                "window_minutes": 60,
            },
            "queue": {"backlog_fraction": 0.12},
        },
        "provenance": {"attestation_hash": "smoke-test-attestation"},
    }
    normalized = NormalizedMetrics.from_raw(metrics_raw)
    gate_statuses = GateEvaluator(normalized, slice_cfg).evaluate()
    gates_passed = all(s.passed for s in gate_statuses)
    assert gates_passed, "Curriculum gates must pass for smoke test"
    harness_print_status("SMOKE", f"Gates passed for {slice_cfg.name}", "PASS")
    
    # Step 3: Derivation Pipeline
    fo_slice = make_first_organism_derivation_slice()
    fo_seeds = make_first_organism_seed_statements()
    
    derivation_result = run_slice_for_test(
        fo_slice,
        existing=list(fo_seeds),
        limit=1,
    )
    
    assert derivation_result.n_candidates > 0, "Derivation must produce candidates"
    assert derivation_result.n_abstained >= 1, "Must have at least one abstention"
    
    candidate = derivation_result.abstained_candidates[0]
    candidate_hash = candidate.hash
    harness_print_status("SMOKE", f"Derived candidate {candidate_hash[:12]}...", "INFO")
    
    # Step 4: Seal Block (in-memory, no DB)
    proof_payload = {
        "statement": candidate.pretty or candidate.normalized,
        "statement_hash": candidate_hash,
        "status": "abstain",
        "prover": "lean-interface",
        "verification_method": candidate.verification_method,
        "reason": "smoke-test-abstention",
    }
    
    # Seal block with dual roots (no DB writes)
    block = seal_block_with_dual_roots("pl", [proof_payload])
    
    r_t = block["reasoning_merkle_root"]
    u_t = block["ui_merkle_root"]
    h_t = block["composite_attestation_root"]
    
    assert r_t and len(r_t) == 64, "R_t must be 64-char hex"
    assert u_t and len(u_t) == 64, "U_t must be 64-char hex"
    assert h_t and len(h_t) == 64, "H_t must be 64-char hex"
    
    # Verify H_t formula
    expected_h_t = compute_composite_root(r_t, u_t)
    assert h_t == expected_h_t, f"H_t formula violated: stored={h_t}, expected={expected_h_t}"
    
    harness_print_status("SMOKE", f"R_t={r_t[:12]}... U_t={u_t[:12]}... H_t={h_t[:12]}...", "PASS")
    
    # Step 5: Mock LedgerIngestor.ingest to be a no-op (skip DB writes)
    def mock_ingest(*args, **kwargs):
        """No-op mock for LedgerIngestor.ingest - skips DB writes."""
        # Return a minimal mock result that won't be used
        from ledger.ingest import IngestOutcome, StatementRecord, ProofRecord, BlockRecord
        mock_stmt = StatementRecord(
            id="mock-stmt-id",
            hash=candidate_hash,
            normalized=candidate.normalized,
            is_axiom=False,
        )
        mock_proof = ProofRecord(
            id="mock-proof-id",
            hash="mock-proof-hash",
            statement=mock_stmt,
            status="abstain",
        )
        mock_block = BlockRecord(
            id=1,
            number=1,
            reasoning_root=r_t,
            ui_root=u_t,
            composite_root=h_t,
            block_hash="mock-block-hash",
        )
        return IngestOutcome(
            statement=mock_stmt,
            proof=mock_proof,
            block=mock_block,
        )
    
    monkeypatch.setattr("ledger.ingest.LedgerIngestor.ingest", mock_ingest)
    monkeypatch.setattr("rfl.runner.load_baseline_from_db", lambda *args, **kwargs: [])
    
    # Step 6: Build AttestedRunContext from sealed block (no DB record)
    slice_name = slice_cfg.name
    attestation_context = AttestedRunContext(
        slice_id=slice_name,
        statement_hash=candidate_hash,
        proof_status="failure",  # Abstained proof
        block_id=1,  # Synthetic block ID
        composite_root=h_t,
        reasoning_root=r_t,
        ui_root=u_t,
        abstention_metrics={
            "rate": 1.0,
            "mass": 1.0,
            "counts": {"verified": 0, "rejected": 1, "considered": 1},
            "reasons": {"lean_failure": 1},
        },
        policy_id="smoke-test-policy",
        metadata={
            "attempt_mass": 1.0,
            "abstention_breakdown": {"lean_failure": 1},
            "first_organism_abstentions": 1,
        },
    )
    
    # Step 7: RFL Metabolism - call run_with_attestation
    from rfl.config import RFLConfig as RFLConfigCanonical
    rfl_config = RFLConfigCanonical(
        experiment_id="first-organism-smoke",
        num_runs=1,
        random_seed=42,
        system_id=1,
        derive_steps=1,
        max_breadth=1,
        max_total=1,
        depth_max=1,
        bootstrap_replicates=1000,
        coverage_threshold=0.0,
        uplift_threshold=0.0,
        dual_attestation=False,
        curriculum=[
            RFLCurriculumSlice(
                name=slice_name,
                start_run=1,
                end_run=1,
                derive_steps=1,
                max_breadth=1,
                max_total=1,
                depth_max=1,
            )
        ],
    )
    runner = RFLRunnerCanonical(rfl_config)
    
    # Call run_with_attestation - this is the critical path
    rfl_result = runner.run_with_attestation(attestation_context)
    
    # Step 8: Verify RFL result
    assert rfl_result.source_root == h_t, f"RFL must reference H_t: got {rfl_result.source_root}, expected {h_t}"
    assert rfl_result.policy_update_applied is True, "Policy should be updated"
    
    # Verify ledger entry
    assert len(runner.policy_ledger) > 0, "Policy ledger should have entries"
    ledger_entry = runner.policy_ledger[-1]
    assert ledger_entry.status == "attestation", f"Expected 'attestation', got {ledger_entry.status}"
    assert runner.abstention_histogram.get("lean_failure", 0) >= 1, "Abstention histogram should record lean failure"
    
    # Verify H_t is correctly stored in result
    assert hasattr(rfl_result, "source_root"), "RflResult must have source_root"
    assert rfl_result.source_root == h_t, "source_root must match H_t"
    
    # Summary output
    summary_msg = (
        f"Smoke test complete, H_t consumed: {h_t[:12]}...\n"
        f"       Policy Update: {'Yes' if rfl_result.policy_update_applied else 'No'}\n"
        f"       Abstentions: {runner.abstention_histogram}\n"
        f"       Symbolic Descent: {ledger_entry.symbolic_descent:.4f}"
    )
    harness_print_status("SMOKE", summary_msg, "PASS")
    
    # Verify no DB calls were made (LedgerIngestor.ingest was mocked)
    # This is implicit - if we got here, the mock worked
    
    harness_print_status("SMOKE", "DB-optional smoke test passed - logical pipeline intact", "PASS")


# ---------------------------------------------------------------------------
# Standalone integration tests (Failure paths & partial failures)
# ---------------------------------------------------------------------------

@pytest.mark.first_organism
@pytest.mark.integration
def test_first_organism_gate_failure():
    """
    Test failure path: Curriculum gate rejects the slice.

    Verifies:
    1. Gate failure produces explicit reason.
    2. Pipeline halts (mocked by not proceeding to derivation).
    """
    harness_print_status("TEST", "Starting Gate Failure Test", "INFO")

    # 1. Setup Failing Metrics (Coverage < 0.915 required by First Organism slice)
    slice_cfg = make_first_organism_slice()
    curriculum_metrics = {
        "metrics": {
            # Coverage 0.10 should fail the CI_LOWER_MIN=0.915 gate
            "rfl": {
                "coverage": {"ci_lower": 0.10, "sample_size": 24},
            },
            "success_rates": {"abstention_rate": 12.0},
            "curriculum": {
                "active_slice": {
                    "attempt_mass": 3200,
                    "wallclock_minutes": 45.0,
                    "proof_velocity_cv": 0.05,
                }
            },
            "throughput": {
                "proofs_per_hour": 240.0,
                "coefficient_of_variation": 0.04,
                "window_minutes": 60,
            },
            "frontier": {"queue_backlog": 0.12},
        },
        "provenance": {"merkle_hash": "attn-stub" * 8},
    }
    normalized = NormalizedMetrics.from_raw(curriculum_metrics)
    
    # 2. Evaluate Gates
    gate_evaluator = GateEvaluator(normalized, slice_cfg)
    gate_statuses = gate_evaluator.evaluate()
    
    # 3. Verify Failure
    failed_gates = [s for s in gate_statuses if not s.passed]
    assert len(failed_gates) > 0, "Expected at least one gate to fail"
    
    failure = failed_gates[0]
    harness_print_status("GATE", f"Gate '{failure.gate}' rejected: {failure.message}", "PASS") # Green because we expect failure here!
    
    # Ensure we have a meaningful error message
    assert failure.message, "Gate failure must have a message"
    assert not all(s.passed for s in gate_statuses)

    # 4. Verify we would NOT proceed (simulated)
    # In a real pipeline, this check prevents the runner from starting.
    # We assert that the system state implies 'Blocked'.
    assert failure.gate == 'coverage', f"Expected coverage gate to fail, got {failure.gate}"


@pytest.mark.first_organism
@pytest.mark.integration
@pytest.mark.hermetic  # This test does NOT require DB/Redis
def test_first_organism_closed_loop_standalone():
    """
    Standalone test of the First Organism closed loop.

    This test verifies:
    1. UI event capture and U_t computation
    2. Curriculum gate evaluation
    3. Derivation pipeline produces abstained candidates
    4. Dual attestation seal computes R_t, U_t, H_t
    5. H_t is recomputable from canonical leaves
    6. RFL runner metabolism verification

    Does NOT require Postgres/Redis - uses in-memory structures.
    """
    # --- UI Event → U_t ---
    ui_event_store.clear()
    ui_event = {
        "event_type": "select_statement",
        "statement_hash": "mock-statement-hash",
        "action": "toggle_abstain",
    }
    capture_ui_event(ui_event)
    events_snapshot = snapshot_ui_events()
    assert len(events_snapshot) == 1
    harness_print_status("UI", "Event captured", "PASS")
    ui_artifacts = consume_ui_artifacts()
    assert len(ui_artifacts) == 1, "UI artifacts should contain the captured event"

    # --- Curriculum Gate ---
    slice_cfg = make_first_organism_slice()
    curriculum_metrics = {
        "metrics": {
            "rfl": {
                "coverage": {"ci_lower": 0.95, "sample_size": 24},
            },
            "success_rates": {"abstention_rate": 12.0},
            "curriculum": {
                "active_slice": {
                    "attempt_mass": 3200,
                    "wallclock_minutes": 45.0,
                    "proof_velocity_cv": 0.05,
                }
            },
            "throughput": {
                "proofs_per_hour": 240.0,
                "coefficient_of_variation": 0.04,
                "window_minutes": 60,
            },
            "frontier": {"queue_backlog": 0.12},
        },
        "provenance": {"merkle_hash": "attn-stub" * 8},
    }
    normalized = NormalizedMetrics.from_raw(curriculum_metrics)
    gate_evaluator = GateEvaluator(normalized, slice_cfg)
    gate_statuses = gate_evaluator.evaluate()
    assert len(gate_statuses) == 4
    assert all(status.passed for status in gate_statuses)
    harness_print_status("GATE", f"All {len(gate_statuses)} gates passed", "PASS")

    # --- Derivation Pipeline (use First Organism slice helper) ---
    # make_first_organism_derivation_slice() returns a CurriculumSlice with:
    # - axiom_instances = 0: No axiom seeding, use seeds only
    # - atoms = 2, depth_max = 2, mp_depth = 1: Minimal search space
    # - lean_timeout_s = 0.001: Effectively disables Lean fallback
    fo_slice = make_first_organism_derivation_slice()
    fo_seeds = make_first_organism_seed_statements()

    derivation_result = run_slice_for_test(
        fo_slice,
        existing=list(fo_seeds),
        limit=1,
    )

    # Assert abstention metrics from DerivationResult
    assert derivation_result.n_candidates > 0, "Pipeline must consider candidates"
    assert derivation_result.n_abstained >= 1, "First Organism slice must produce at least one abstention"
    assert derivation_result.abstained_candidates, "abstained_candidates must be populated"

    # Pick the abstained statement for the ledger + dual-root step
    candidate = derivation_result.abstained_candidates[0]
    ascii_statement = candidate.pretty or candidate.normalized

    harness_print_status(
        "DERIVE",
        f"Abstained: {ascii_statement[:30]}... hash={candidate.hash[:12]}... method={candidate.verification_method}",
        "INFO"
    )

    # --- Hash Contract Assertion (bound to tests/test_canon.py) ---
    # Verify that the candidate hash matches the canonical hash contract:
    # hash(s) = SHA256(DOMAIN_STMT || canonical_bytes(s))
    assert_hash_contract(
        candidate.normalized,
        candidate.hash,
        context="derivation pipeline candidate",
    )
    harness_print_status("HASH", f"Hash contract verified for {candidate.hash[:12]}...", "PASS")

    # --- Dual Attestation Seal (R_t, U_t, H_t) ---
    proof_payload = {
        "statement": ascii_statement,
        "statement_hash": candidate.hash,
        "status": "abstain",
        "prover": "lean-interface",
        "verification_method": candidate.verification_method,
        "reason": "non-tautology",
    }
    block = _seal_single_block(proof_payload, ui_artifacts=ui_artifacts)

    # Verify block has all required attestation fields
    assert block["reasoning_merkle_root"], "R_t must be computed"
    assert block["ui_merkle_root"], "U_t must be computed"
    assert block["composite_attestation_root"], "H_t must be computed"
    assert block["ui_event_count"] == 1
    assert block["reasoning_leaves"]
    assert block["ui_leaves"]

    r_t = block["reasoning_merkle_root"]
    u_t = block["ui_merkle_root"]
    h_t = block["composite_attestation_root"]
    harness_print_status("ATTEST", f"R_t={r_t[:12]}... U_t={u_t[:12]}... H_t={h_t[:12]}...", "PASS")

    # --- Hash Contract Assertion for Sealed Block ---
    # Verify the statement_hash in the proof payload matches canonical computation
    assert_hash_contract(
        ascii_statement,
        proof_payload["statement_hash"],
        context="sealed block proof payload",
    )

    # --- H_t Recomputability Verification (Core Invariant) ---
    # Recompute R_t from canonical leaves using attestation.dual_root
    recomputed_reasoning_root = compute_reasoning_root(
        [leaf["canonical_value"] for leaf in block["reasoning_leaves"]]
    )
    # Recompute U_t from canonical leaves using attestation.dual_root
    recomputed_ui_root = compute_ui_root(
        [leaf["canonical_value"] for leaf in block["ui_leaves"]]
    )
    # Recompute H_t using attestation.dual_root
    recomputed_h_t = compute_composite_root(
        recomputed_reasoning_root,
        recomputed_ui_root,
    )

    # Assert all recomputed values match stored values
    assert recomputed_reasoning_root == block["reasoning_merkle_root"], (
        f"R_t mismatch: recomputed={recomputed_reasoning_root}, "
        f"stored={block['reasoning_merkle_root']}"
    )
    assert recomputed_ui_root == block["ui_merkle_root"], (
        f"U_t mismatch: recomputed={recomputed_ui_root}, "
        f"stored={block['ui_merkle_root']}"
    )
    assert recomputed_h_t == block["composite_attestation_root"], (
        f"H_t Invariant violated: recomputed={recomputed_h_t}, "
        f"stored={block['composite_attestation_root']}"
    )

    # Also verify using the integrity check helper
    assert verify_composite_integrity(
        block["reasoning_merkle_root"],
        block["ui_merkle_root"],
        block["composite_attestation_root"],
    ), "verify_composite_integrity failed for sealed block"

    # Verify attestation metadata
    metadata = block["attestation_metadata"]
    assert metadata["attestation_version"] == "v2"
    assert metadata["reasoning_event_count"] == 1
    assert metadata["ui_event_count"] == 1
    assert metadata["composite_attestation_root"] == block["composite_attestation_root"]
    assert metadata["composite_formula"] == "SHA256(R_t || U_t)"

    # Draining store ensures no residual events
    assert not materialize_ui_artifacts()

    # --- RFL Runner Metabolism (symbolic ledger + metabolism check) ---
    # Use fo_slice (First Organism derivation slice) for the RFL ledger entry
    ledger_entry = RunLedgerEntry(
        run_id="first_organism_run_01",
        slice_name=fo_slice.name,
        status="abstain",
        coverage_rate=0.0,
        novelty_rate=0.0,
        throughput=0.0,
        success_rate=0.0,
        abstention_fraction=1.0,
        policy_reward=0.0,
        symbolic_descent=-1.0,
        budget_spent=int(fo_slice.params["total_max"]),
        derive_steps=1,
        max_breadth=int(fo_slice.params["breadth_max"]),
        max_total=int(fo_slice.params["total_max"]),
        abstention_breakdown={"lean_failure": 1},
    )
    ledger_record = asdict(ledger_entry)
    assert ledger_record["status"] == "abstain"
    assert math.isclose(ledger_record["abstention_fraction"], 1.0)

    coverage_result = BootstrapResult(
        point_estimate=0.95,
        ci_lower=0.93,
        ci_upper=0.97,
        std_error=0.01,
        num_replicates=256,
        method="BCa_95%",
    )
    uplift_result = BootstrapResult(
        point_estimate=1.12,
        ci_lower=1.05,
        ci_upper=1.20,
        std_error=0.02,
        num_replicates=256,
        method="BCa_95%",
    )
    metabolism_passed, metabolism_message = verify_metabolism(
        coverage_result,
        uplift_result,
        coverage_threshold=0.92,
        uplift_threshold=1.0,
    )
    assert metabolism_passed
    assert "Reflexive Metabolism Verified" in metabolism_message
    harness_print_status("RFL", "Metabolism verified", "PASS")

    # Final assert tying sealed attestation to RFL ledger context.
    assert block["composite_attestation_root"] == metadata["composite_attestation_root"]
    assert block["composite_attestation_root"] not in {None, ""}

    # Build and write attestation artifact
    attestation = build_first_organism_attestation(
        statement_hash=candidate.hash,
        reasoning_root=r_t,
        ui_root=u_t,
        composite_root=h_t,
        environment_mode="standalone",
        slice_name=fo_slice.name,
    )
    artifact_path = _write_attestation_artifact(
        attestation,
        extra_metadata={"test": "standalone"},
    )
    harness_print_status("ARTIFACT", f"Written to {artifact_path}", "INFO")

    # --- Attestation Artifact Verification ---
    assert artifact_path.exists(), f"Attestation artifact not found: {artifact_path}"
    artifact_data = json.loads(artifact_path.read_text(encoding="utf-8"))
    
    # Verify required roots exist
    reasoning_root = artifact_data.get("reasoning_root") or artifact_data.get("R_t")
    ui_root = artifact_data.get("ui_root") or artifact_data.get("U_t")
    composite_root = artifact_data.get("composite_root") or artifact_data.get("H_t")
    
    missing = []
    if not reasoning_root:
        missing.append("reasoning_root (or R_t)")
    if not ui_root:
        missing.append("ui_root (or U_t)")
    if not composite_root:
        missing.append("composite_root (or H_t)")
    
    if missing:
        raise AssertionError(
            f"Attestation artifact missing required fields: {', '.join(missing)}\n"
            f"  Artifact path: {artifact_path}\n"
            f"  Available keys: {list(artifact_data.keys())}"
        )
    
    # Verify root lengths (64 hex characters)
    root_errors = []
    if len(reasoning_root) != 64:
        root_errors.append(f"reasoning_root length={len(reasoning_root)}, expected 64")
    if len(ui_root) != 64:
        root_errors.append(f"ui_root length={len(ui_root)}, expected 64")
    if len(composite_root) != 64:
        root_errors.append(f"composite_root length={len(composite_root)}, expected 64")
    
    if root_errors:
        raise AssertionError(
            f"Attestation artifact root length violations:\n  " + "\n  ".join(root_errors) +
            f"\n  Artifact path: {artifact_path}"
        )
    
    # Verify roots match expected values
    if reasoning_root != r_t:
        raise AssertionError(
            f"reasoning_root mismatch: artifact={reasoning_root}, expected={r_t}"
        )
    if ui_root != u_t:
        raise AssertionError(
            f"ui_root mismatch: artifact={ui_root}, expected={u_t}"
        )
    if composite_root != h_t:
        raise AssertionError(
            f"composite_root mismatch: artifact={composite_root}, expected={h_t}"
        )
    
    harness_print_status("ARTIFACT", "Attestation artifact verified", "PASS")

    # Emit canonical PASS message
    log_first_organism_pass(h_t)


# ---------------------------------------------------------------------------
# Full integration test with database and API client
# ---------------------------------------------------------------------------


@pytest.mark.first_organism
@pytest.mark.integration
@pytest.mark.requires_db  # Requires database connection
@pytest.mark.skipif(
    os.getenv("SKIP_DB_TESTS", "0") == "1",
    reason="SKIP_DB_TESTS is set"
)
def test_first_organism_full_integration(test_client, test_db_connection, test_db_url, monkeypatch, tmp_path):
    """
    Full integration test with database and API client.

    This test validates that the attestation API endpoints return values
    that match the canonical dual-root computation.
    """
    from interface.api.schemas import ParentListResponse, ProofListResponse, StatementDetailResponse

    ui_event_store.clear()
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    monkeypatch.setenv("REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost")

    with test_db_connection.cursor() as cur:
        cur.execute("DELETE FROM proof_parents")
        cur.execute("DELETE FROM proofs")
        cur.execute("DELETE FROM statements")
        cur.execute("DELETE FROM blocks")
        cur.execute("DELETE FROM ledger_sequences")
        cur.execute("DELETE FROM runs")
        cur.execute("DELETE FROM theories")
        cur.execute("DELETE FROM policy_settings")
    test_db_connection.commit()

    slice_cfg = make_first_organism_slice()
    curriculum = CurriculumSystem(
        slug="first-organism",
        description="Integration slice",
        slices=[slice_cfg],
        active_index=0,
    )
    metrics = {
        "metrics": {
            "rfl": {
                "coverage": {"ci_lower": 0.95, "sample_size": 64},
            },
            "success_rates": {"abstention_rate": 12.5},
            "curriculum": {
                "active_slice": {
                    "attempt_mass": 4800,
                    "wallclock_minutes": 32,
                    "proof_velocity_cv": 0.08,
                }
            },
            "throughput": {"proofs_per_hour": 210.0, "coefficient_of_variation": 0.04},
            "frontier": {"queue_backlog": 0.22},
        },
        "provenance": {"merkle_hash": "integration-test-attestation" * 8},
    }
    verdict = should_ratchet(metrics, curriculum)
    assert verdict.advance, verdict.reason

    bounds = SliceBounds(
        max_atoms=slice_cfg.params.get("atoms", 4),
        max_formula_depth=slice_cfg.params.get("depth_max", 4),
        max_mp_depth=2,
        max_breadth=8,
        max_total=16,
        max_axiom_instances=16,
        max_formula_pool=64,
        lean_timeout_s=0.01,
    )
    pipeline = DerivationPipeline(bounds, StatementVerifier(bounds))
    outcome = pipeline.run_step(existing=[])
    assert outcome.statements, "Derivation pipeline produced no candidates"
    candidate = outcome.statements[0]

    ingestor = LedgerIngestor()
    with test_db_connection.cursor() as cur:
        ingest_result = ingestor.ingest(
            cur,
            theory_name="First Organism Integration",
            ascii_statement=candidate.pretty,
            proof_text="(lean-abstain)",
            prover="lean",
            status="abstain",
            module_name="integration.test",
            stdout=None,
            stderr="Lean abstained deterministically",
            derivation_rule=candidate.rule,
            derivation_depth=candidate.mp_depth,
            method="lean",
            duration_ms=250,
            truth_domain="pl",
            is_axiom=candidate.is_axiom,
            ui_events=materialize_ui_artifacts(),
            sealed_by="integration-test",
        )
    test_db_connection.commit()

    # Verify H_t is recomputable for the ingested block
    stored_r_t = ingest_result.block.reasoning_root
    stored_u_t = ingest_result.block.ui_root
    stored_h_t = ingest_result.block.composite_root

    recomputed_h_t = compute_composite_root(stored_r_t, stored_u_t)
    assert recomputed_h_t == stored_h_t, (
        f"H_t Invariant violated in LedgerIngestor: "
        f"recomputed={recomputed_h_t}, stored={stored_h_t}"
    )

    # Post UI event via API
    event_payload = {
        "event_id": "ui-organism-01",
        "action": "prove",
        "statement_hash": candidate.hash,
        "kind": "ui.select",
    }
    response = test_client.post("/attestation/ui-event", json=event_payload)
    response.raise_for_status()
    result = response.json()
    assert "event_id" in result and "leaf_hash" in result

    # Fetch attestation via API
    response = test_client.get("/attestation/latest")
    response.raise_for_status()
    attestation = response.json()
    assert attestation["composite_attestation_root"] == ingest_result.block.composite_root

    # Verify API-returned attestation is also recomputable
    api_r_t = attestation["reasoning_merkle_root"]
    api_u_t = attestation["ui_merkle_root"]
    api_h_t = attestation["composite_attestation_root"]

    api_recomputed_h_t = compute_composite_root(api_r_t, api_u_t)
    assert api_recomputed_h_t == api_h_t, (
        f"H_t Invariant violated in API response: "
        f"recomputed={api_recomputed_h_t}, stored={api_h_t}"
    )

    # Fetch statement bundle
    stmt_response = test_client.get(f"/ui/statement/{ingest_result.statement.hash}.json")
    stmt_response.raise_for_status()
    detail = StatementDetailResponse.parse_obj(stmt_response.json())
    assert detail.hash == ingest_result.statement.hash
    assert detail.proofs and detail.proofs[0].status == "abstain"

    # Build attestation artifact
    attestation_obj = build_first_organism_attestation(
        statement_hash=candidate.hash,
        reasoning_root=stored_r_t,
        ui_root=stored_u_t,
        composite_root=stored_h_t,
        block_id=ingest_result.block.id,
        proof_id=getattr(ingest_result.proof, "id", None),
        statement_id=getattr(ingest_result.statement, "id", None),
        environment_mode="full_integration",
        slice_name=slice_cfg.name,
    )
    artifact_path = _write_attestation_artifact(
        attestation_obj,
        extra_metadata={"test": "full_integration"},
    )
    harness_print_status("ARTIFACT", f"Written to {artifact_path}", "INFO")

    # --- Attestation Artifact Verification ---
    assert artifact_path.exists(), f"Attestation artifact not found: {artifact_path}"
    artifact_data = json.loads(artifact_path.read_text(encoding="utf-8"))
    
    # Verify required roots exist
    reasoning_root = artifact_data.get("reasoning_root") or artifact_data.get("R_t")
    ui_root = artifact_data.get("ui_root") or artifact_data.get("U_t")
    composite_root = artifact_data.get("composite_root") or artifact_data.get("H_t")
    
    missing = []
    if not reasoning_root:
        missing.append("reasoning_root (or R_t)")
    if not ui_root:
        missing.append("ui_root (or U_t)")
    if not composite_root:
        missing.append("composite_root (or H_t)")
    
    if missing:
        raise AssertionError(
            f"Attestation artifact missing required fields: {', '.join(missing)}\n"
            f"  Artifact path: {artifact_path}\n"
            f"  Available keys: {list(artifact_data.keys())}"
        )
    
    # Verify root lengths (64 hex characters)
    root_errors = []
    if len(reasoning_root) != 64:
        root_errors.append(f"reasoning_root length={len(reasoning_root)}, expected 64")
    if len(ui_root) != 64:
        root_errors.append(f"ui_root length={len(ui_root)}, expected 64")
    if len(composite_root) != 64:
        root_errors.append(f"composite_root length={len(composite_root)}, expected 64")
    
    if root_errors:
        raise AssertionError(
            f"Attestation artifact root length violations:\n  " + "\n  ".join(root_errors) +
            f"\n  Artifact path: {artifact_path}"
        )
    
    # Verify roots match expected values
    if reasoning_root != stored_r_t:
        raise AssertionError(
            f"reasoning_root mismatch: artifact={reasoning_root}, expected={stored_r_t}"
        )
    if ui_root != stored_u_t:
        raise AssertionError(
            f"ui_root mismatch: artifact={ui_root}, expected={stored_u_t}"
        )
    if composite_root != stored_h_t:
        raise AssertionError(
            f"composite_root mismatch: artifact={composite_root}, expected={stored_h_t}"
        )
    
    harness_print_status("ARTIFACT", "Attestation artifact verified", "PASS")

    # Emit canonical PASS message
    log_first_organism_pass(stored_h_t)


# ---------------------------------------------------------------------------
# PHASE 1: UI Event Capture
# ---------------------------------------------------------------------------
@pytest.mark.first_organism
@pytest.mark.hermetic  # This test does NOT require DB/Redis
def test_first_organism_ui_event_capture():
    """
    Phase 1: UI Event Capture and U_t Computation

    Validates:
    1. UI event is captured with deterministic ID and timestamp
    2. Event is stored in the UI event store
    3. Snapshot correctly retrieves the event
    4. Event payload is MDAP-compliant (no wall-clock time)

    Whitepaper mapping: UI Event → U_t leaf
    """
    _state.log_phase("UI", "Starting UI Event Capture phase", "INFO")

    # Clear any prior state
    ui_event_store.clear()

    # Generate deterministic UI event
    seed_content = "first-organism-ui-event-seed"
    event_id = mdap_deterministic_id("ui-event", seed_content)
    event_timestamp = mdap_deterministic_timestamp(seed_content)

    ui_event = {
        "event_id": event_id,
        "event_type": "select_statement",
        "actor": "first-organism-test",
        "statement_hash": deterministic_hash("p -> p"),
        "action": "toggle_abstain",
        "meta": {"origin": "first-organism-integration", "phase": 1},
        "timestamp": event_timestamp,
    }

    # Capture the event
    capture_ui_event(ui_event)
    _state.ui_event = ui_event
    _state.ui_event_id = event_id

    # Verify event is stored
    _state.ui_snapshot = snapshot_ui_events()
    assert len(_state.ui_snapshot) == 1, f"Expected 1 event, got {len(_state.ui_snapshot)}"

    # Verify determinism: same inputs produce same outputs
    event_id_2 = mdap_deterministic_id("ui-event", seed_content)
    assert event_id == event_id_2, "UI event ID not deterministic"

    event_ts_2 = mdap_deterministic_timestamp(seed_content)
    assert event_timestamp == event_ts_2, "UI event timestamp not deterministic"

    _state.log_phase("UI", f"Event {event_id[:12]}... captured", "PASS")


# ---------------------------------------------------------------------------
# PHASE 2: Curriculum Gate Evaluation
# ---------------------------------------------------------------------------
@pytest.mark.first_organism
@pytest.mark.hermetic  # This test does NOT require DB/Redis
def test_first_organism_curriculum_gate():
    """
    Phase 2: Curriculum Gate Evaluation

    Validates:
    1. First Organism slice is constructed with permissive gates
    2. Gate evaluator runs without error
    3. All gates pass (coverage, abstention, velocity, caps)
    4. Gate evaluation is deterministic

    Whitepaper mapping: Curriculum Gate
    """
    _state.log_phase("GATE", "Starting Curriculum Gate phase", "INFO")

    # Build the canonical First Organism slice
    _state.slice_cfg = make_first_organism_slice()

    # Build metrics that will pass all gates
    metrics = {
        "metrics": {
            "rfl": {
                "coverage": {"ci_lower": 0.95, "sample_size": 24},
            },
            "success_rates": {"abstention_rate": 12.0},
            "curriculum": {
                "active_slice": {
                    "attempt_mass": 3200,
                    "wallclock_minutes": 45.0,
                    "proof_velocity_cv": 0.05,
                }
            },
            "throughput": {
                "proofs_per_hour": 240.0,
                "coefficient_of_variation": 0.04,
                "window_minutes": 60,
            },
            "frontier": {"queue_backlog": 0.12},
        },
        "provenance": {"merkle_hash": "test-attestation-hash" * 8},
    }
    normalized = NormalizedMetrics.from_raw(metrics)

    # Evaluate gates
    gate_evaluator = GateEvaluator(normalized, _state.slice_cfg)
    _state.gate_statuses = gate_evaluator.evaluate()

    # Assert all gates pass
    failed_gates = [s for s in _state.gate_statuses if not s.passed]
    assert not failed_gates, f"Gates failed: {[(s.gate, s.message) for s in failed_gates]}"

    # Verify determinism: same inputs produce same gate statuses
    gate_statuses_2 = GateEvaluator(normalized, _state.slice_cfg).evaluate()
    assert len(_state.gate_statuses) == len(gate_statuses_2)
    for s1, s2 in zip(_state.gate_statuses, gate_statuses_2):
        assert s1.gate == s2.gate
        assert s1.passed == s2.passed

    _state.log_phase(
        "GATE",
        f"All {len(_state.gate_statuses)} gates passed for {_state.slice_cfg.name}",
        "PASS"
    )


# ---------------------------------------------------------------------------
# PHASE 3: Derivation and Abstention
# ---------------------------------------------------------------------------
@pytest.mark.first_organism
@pytest.mark.hermetic  # This test does NOT require DB/Redis
def test_first_organism_derivation_and_abstention():
    """
    Phase 3: Derivation Pipeline with Abstention

    Validates:
    1. First Organism derivation config produces expected seeds
    2. Derivation pipeline runs and produces candidates
    3. At least one candidate is abstained (non-tautology)
    4. Hash contract holds for all candidates
    5. Derivation is deterministic

    Whitepaper mapping: Derivation → Lean Verify (abstention)
    """
    _state.log_phase("DERIVE", "Starting Derivation phase", "INFO")

    # Get the First Organism derivation slice and seeds
    # make_first_organism_derivation_slice() is the controlled experimental apparatus
    fo_slice = make_first_organism_derivation_slice()
    fo_seeds = make_first_organism_seed_statements()

    # Run derivation with seeds
    _state.derivation_result = run_slice_for_test(
        fo_slice,
        existing=list(fo_seeds),
        limit=1,
    )

    # Verify we got candidates
    assert _state.derivation_result.n_candidates > 0, "No candidates considered"

    # Verify we got abstentions (the whole point of First Organism)
    assert _state.derivation_result.n_abstained >= 1, (
        f"Expected at least one abstention, got {_state.derivation_result.n_abstained}. "
        f"Expected: MP derives q from p and p->q, q fails truth-table (not a tautology), "
        f"Lean disabled -> abstention."
    )

    assert _state.derivation_result.abstained_candidates, "No abstained candidates"
    _state.candidate = _state.derivation_result.abstained_candidates[0]

    # Verify hash contract for the candidate
    assert_hash_contract(
        _state.candidate.normalized,
        _state.candidate.hash,
        context="derivation abstained candidate",
    )

    # Verify determinism: same config produces same results
    result_2 = run_slice_for_test(
        fo_slice,
        existing=list(fo_seeds),
        limit=1,
    )
    assert result_2.n_candidates == _state.derivation_result.n_candidates
    assert result_2.n_abstained == _state.derivation_result.n_abstained
    if result_2.abstained_candidates and _state.derivation_result.abstained_candidates:
        assert result_2.abstained_candidates[0].hash == _state.candidate.hash

    _state.log_phase(
        "DERIVE",
        f"Abstained: {_state.candidate.pretty or _state.candidate.normalized[:30]}... "
        f"({_state.candidate.verification_method})",
        "PASS"
    )


# ---------------------------------------------------------------------------
# PHASE 4: Dual Attestation Seal
# ---------------------------------------------------------------------------
@pytest.mark.first_organism
@pytest.mark.hermetic  # This test does NOT require DB/Redis
def test_first_organism_dual_attestation_seal():
    """
    Phase 4: Dual-Root Attestation (R_t, U_t, H_t)

    Validates:
    1. Block is sealed with reasoning and UI events
    2. R_t is computed from proof events
    3. U_t is computed from UI events
    4. H_t = SHA256(R_t || U_t) is verified
    5. H_t is recomputable from stored leaves
    6. Attestation metadata is complete

    Whitepaper mapping: Dual-Attest seal H_t
    """
    _state.log_phase("ATTEST", "Starting Dual Attestation phase", "INFO")

    # Ensure we have state from previous phases
    if _state.candidate is None:
        test_first_organism_derivation_and_abstention()
    if _state.ui_event is None:
        test_first_organism_ui_event_capture()

    # Build proof payload
    proof_payload = {
        "statement": _state.candidate.pretty or _state.candidate.normalized,
        "statement_hash": _state.candidate.hash,
        "status": "abstain",
        "prover": "lean-interface",
        "verification_method": _state.candidate.verification_method,
        "reason": "non-tautology",
    }

    # Seal block with dual roots
    _state.block = seal_block_with_dual_roots("pl", [proof_payload])

    # Extract attestation roots
    r_t = _state.block["reasoning_merkle_root"]
    u_t = _state.block["ui_merkle_root"]
    h_t = _state.block["composite_attestation_root"]

    # Verify roots are non-empty
    assert r_t, "R_t must be computed"
    assert u_t, "U_t must be computed"
    assert h_t, "H_t must be computed"
    assert len(r_t) == 64, f"R_t must be 64 hex chars, got {len(r_t)}"
    assert len(u_t) == 64, f"U_t must be 64 hex chars, got {len(u_t)}"
    assert len(h_t) == 64, f"H_t must be 64 hex chars, got {len(h_t)}"

    # Verify H_t formula: H_t = SHA256(R_t || U_t)
    expected_h_t = compute_composite_root(r_t, u_t)
    assert h_t == expected_h_t, f"H_t formula violated: stored={h_t}, expected={expected_h_t}"

    # Verify integrity helper
    assert verify_composite_integrity(r_t, u_t, h_t), "verify_composite_integrity failed"

    # Build attestation artifact
    _state.attestation = build_first_organism_attestation(
        statement_hash=_state.candidate.hash,
        reasoning_root=r_t,
        ui_root=u_t,
        composite_root=h_t,
        environment_mode="standalone",
        slice_name=_state.slice_cfg.name if _state.slice_cfg else "first-organism-test",
    )

    _state.log_phase("ATTEST", f"R_t={r_t[:12]}...", "INFO")
    _state.log_phase("ATTEST", f"U_t={u_t[:12]}...", "INFO")
    _state.log_phase("ATTEST", f"H_t={h_t[:12]}...", "PASS")


# ---------------------------------------------------------------------------
# PHASE 5: RFL Metabolism
# ---------------------------------------------------------------------------
@pytest.mark.first_organism
@pytest.mark.hermetic  # This test does NOT require DB/Redis (mocks DB loader)
def test_first_organism_rfl_metabolism(monkeypatch):
    """
    Phase 5: RFL Runner Metabolism Verification

    Validates:
    1. RFL runner accepts attested context
    2. Policy ledger records the abstention
    3. Abstention histogram is updated
    4. Symbolic descent is computed
    5. Metabolism verification passes
    6. H_t is correctly referenced in policy context

    Whitepaper mapping: RFL runner metabolism
    """
    _state.log_phase("RFL", "Starting RFL Metabolism phase", "INFO")

    # Ensure we have state from previous phases
    if _state.block is None:
        test_first_organism_dual_attestation_seal()
    if _state.candidate is None:
        test_first_organism_derivation_and_abstention()

    # Patch DB loader to avoid dependency
    monkeypatch.setattr("rfl.runner.load_baseline_from_db", lambda *args, **kwargs: [])

    # Build RFL config
    slice_name = _state.slice_cfg.name if _state.slice_cfg else "first-organism-test"
    from rfl.config import RFLConfig as RFLConfigCanonical
    config = RFLConfigCanonical(
        experiment_id="first-organism-test",
        num_runs=2,
        random_seed=MDAP_EPOCH_SEED,
        system_id=1,
        derive_steps=1,
        max_breadth=1,
        max_total=1,
        depth_max=1,
        bootstrap_replicates=1000,
        coverage_threshold=0.92,
        uplift_threshold=0.0,
        curriculum=[
            RFLCurriculumSlice(
                name=slice_name,
                start_run=1,
                end_run=2,
                derive_steps=1,
                max_breadth=1,
                max_total=1,
                depth_max=1,
            )
        ],
        dual_attestation=False,
    )

    _state.rfl_runner = RFLRunnerCanonical(config)

    # Build attested run context from block
    h_t = _state.block["composite_attestation_root"]
    r_t = _state.block["reasoning_merkle_root"]
    u_t = _state.block["ui_merkle_root"]

    attestation_context = AttestedRunContext(
        slice_id=slice_name,
        statement_hash=_state.candidate.hash,
        proof_status="failure",
        block_id=1,
        composite_root=h_t,
        reasoning_root=r_t,
        ui_root=u_t,
        abstention_metrics={"rate": 1.0, "mass": 1.0},
        policy_id="first-organism-policy",
        metadata={
            "attempt_mass": 1.0,
            "abstention_breakdown": {"lean_failure": 1},
            "first_organism_abstentions": 1,
        },
    )

    # Run metabolism
    _state.rfl_result = _state.rfl_runner.run_with_attestation(attestation_context)

    # Verify policy update
    assert _state.rfl_result.policy_update_applied is True, "Policy should be updated"
    assert _state.rfl_result.source_root == h_t, "Source root should match H_t"

    # Verify ledger entry
    assert len(_state.rfl_runner.policy_ledger) > 0, "Policy ledger should have entries"
    ledger_entry = _state.rfl_runner.policy_ledger[-1]
    assert ledger_entry.status == "attestation", f"Expected 'attestation', got {ledger_entry.status}"

    # Verify abstention histogram
    assert _state.rfl_runner.abstention_histogram["lean_failure"] >= 1, "lean_failure should be counted"

    # Verify first organism counter
    assert _state.rfl_runner.first_organism_runs_total >= 1, "First organism runs should be counted"

    # Verify attestation records
    attestations = _state.rfl_runner.dual_attestation_records.get("attestations", [])
    assert len(attestations) > 0, "Attestation records should be stored"
    assert attestations[-1]["composite_root"] == h_t, "H_t should be recorded"

    _state.log_phase(
        "RFL",
        f"Metabolism: H_t consumed, symbolic_descent={ledger_entry.symbolic_descent:.4f}",
        "PASS"
    )


# ---------------------------------------------------------------------------
# Determinism Verification Test
# ---------------------------------------------------------------------------
@pytest.mark.first_organism
@pytest.mark.determinism
@pytest.mark.hermetic  # This test does NOT require DB/Redis
def test_first_organism_determinism():
    """
    Verify that the entire First Organism pipeline is deterministic.

    Runs the derivation pipeline twice with identical inputs and asserts
    that all outputs (hashes, timestamps, IDs) are byte-for-byte identical.
    """
    _state.log_phase("DETERMINISM", "Starting determinism verification", "INFO")

    # Get the First Organism slice and seeds
    fo_slice = make_first_organism_derivation_slice()
    fo_seeds = make_first_organism_seed_statements()

    # Run 1
    result1 = run_slice_for_test(fo_slice, existing=list(fo_seeds), limit=1)

    # Run 2
    result2 = run_slice_for_test(fo_slice, existing=list(fo_seeds), limit=1)

    # Verify identical outputs
    assert result1.n_candidates == result2.n_candidates
    assert result1.n_verified == result2.n_verified
    assert result1.n_abstained == result2.n_abstained

    # Verify deterministic hashes
    hashes1 = sorted([s.hash for s in result1.abstained_candidates])
    hashes2 = sorted([s.hash for s in result2.abstained_candidates])
    assert hashes1 == hashes2, "Abstained candidate hashes must be identical"

    # Verify deterministic IDs
    id1 = mdap_deterministic_id("test", "determinism-check")
    id2 = mdap_deterministic_id("test", "determinism-check")
    assert id1 == id2, "MDAP IDs must be deterministic"

    # Verify deterministic timestamps
    ts1 = mdap_deterministic_timestamp("determinism-check")
    ts2 = mdap_deterministic_timestamp("determinism-check")
    assert ts1 == ts2, "MDAP timestamps must be deterministic"

    _state.log_phase("DETERMINISM", "All outputs are byte-for-byte identical", "PASS")


# ---------------------------------------------------------------------------
# Full Chain (combines all phases, emits PASS line for CI)
# ---------------------------------------------------------------------------
@pytest.mark.first_organism
@pytest.mark.hermetic  # This test does NOT require DB/Redis
def test_first_organism_full_chain(monkeypatch):
    """
    Full chain integration test combining all five phases.

    This test executes the complete First Organism loop in sequence:
    1. UI Event Capture
    2. Curriculum Gate
    3. Derivation and Abstention
    4. Dual Attestation Seal
    5. RFL Metabolism

    Emits the canonical PASS line that Cursor P relies on.
    """
    _state.log_phase("CHAIN", "Starting Full Chain test", "INFO")

    # Reset state for fresh run
    _state.reset()
    ui_event_store.clear()

    # --- Phase 1: UI Event ---
    seed_content = "first-organism-full-chain"
    event_id = mdap_deterministic_id("ui-event", seed_content)
    event_timestamp = mdap_deterministic_timestamp(seed_content)
    ui_event = {
        "event_id": event_id,
        "event_type": "select_statement",
        "actor": "full-chain-test",
        "statement_hash": deterministic_hash("p -> q"),
        "action": "toggle_abstain",
        "timestamp": event_timestamp,
    }
    capture_ui_event(ui_event)
    _state.ui_event = ui_event
    _state.log_phase("CHAIN", "Phase 1 (UI) complete", "INFO")

    # --- Phase 2: Curriculum Gate ---
    _state.slice_cfg = make_first_organism_slice()
    metrics = {
        "metrics": {
            "rfl": {
                "coverage": {"ci_lower": 0.95, "sample_size": 24},
            },
            "success_rates": {"abstention_rate": 12.0},
            "curriculum": {
                "active_slice": {
                    "attempt_mass": 3200,
                    "wallclock_minutes": 45.0,
                    "proof_velocity_cv": 0.05,
                }
            },
            "throughput": {
                "proofs_per_hour": 240.0,
                "coefficient_of_variation": 0.04,
                "window_minutes": 60,
            },
            "frontier": {"queue_backlog": 0.12},
        },
        "provenance": {"merkle_hash": "full-chain-attestation" * 8},
    }
    normalized = NormalizedMetrics.from_raw(metrics)
    _state.gate_statuses = GateEvaluator(normalized, _state.slice_cfg).evaluate()
    assert all(s.passed for s in _state.gate_statuses), "Gates must pass"
    _state.log_phase("CHAIN", "Phase 2 (Gate) complete", "INFO")

    # --- Phase 3: Derivation ---
    # Use the First Organism slice (axiom_instances=0, MP fires on seeds)
    fo_slice = make_first_organism_derivation_slice()
    fo_seeds = make_first_organism_seed_statements()
    _state.derivation_result = run_slice_for_test(
        fo_slice,
        existing=list(fo_seeds),
        limit=1,
    )
    assert _state.derivation_result.n_abstained >= 1, "Must have at least one abstention"
    _state.candidate = _state.derivation_result.abstained_candidates[0]
    _state.log_phase("CHAIN", "Phase 3 (Derive) complete", "INFO")

    # --- Phase 4: Dual Attestation ---
    proof_payload = {
        "statement": _state.candidate.pretty or _state.candidate.normalized,
        "statement_hash": _state.candidate.hash,
        "status": "abstain",
        "prover": "lean-interface",
        "verification_method": _state.candidate.verification_method,
        "reason": "non-tautology",
    }
    _state.block = seal_block_with_dual_roots("pl", [proof_payload])
    r_t = _state.block["reasoning_merkle_root"]
    u_t = _state.block["ui_merkle_root"]
    h_t = _state.block["composite_attestation_root"]
    assert verify_composite_integrity(r_t, u_t, h_t), "H_t integrity check failed"
    _state.log_phase("CHAIN", "Phase 4 (Attest) complete", "INFO")

    # --- Phase 5: RFL Metabolism ---
    monkeypatch.setattr("rfl.runner.load_baseline_from_db", lambda *args, **kwargs: [])
    slice_name = _state.slice_cfg.name
    from rfl.config import RFLConfig as RFLConfigCanonical
    rfl_config = RFLConfigCanonical(
        experiment_id="full-chain-test",
        num_runs=2,
        random_seed=MDAP_EPOCH_SEED,
        system_id=1,
        derive_steps=1,
        max_breadth=1,
        max_total=1,
        depth_max=1,
        bootstrap_replicates=1000,
        coverage_threshold=0.92,
        uplift_threshold=0.0,
        dual_attestation=False,
        curriculum=[
            RFLCurriculumSlice(
                name=slice_name,
                start_run=1,
                end_run=1,
                derive_steps=1,
                max_breadth=1,
                max_total=1,
                depth_max=1,
            )
        ],
    )
    _state.rfl_runner = RFLRunnerCanonical(rfl_config)
    attestation_context = AttestedRunContext(
        slice_id=slice_name,
        statement_hash=_state.candidate.hash,
        proof_status="failure",
        block_id=1,
        composite_root=h_t,
        reasoning_root=r_t,
        ui_root=u_t,
        abstention_metrics={"rate": 1.0, "mass": 1.0},
        policy_id="full-chain-policy",
        metadata={
            "attempt_mass": 1.0,
            "abstention_breakdown": {"lean_failure": 1},
            "first_organism_abstentions": 1,
        },
    )
    _state.rfl_result = _state.rfl_runner.run_with_attestation(attestation_context)
    assert _state.rfl_result.source_root == h_t, "RFL must reference H_t"
    _state.log_phase("CHAIN", "Phase 5 (RFL) complete", "INFO")

    # --- Write Attestation Artifact ---
    _state.attestation = build_first_organism_attestation(
        statement_hash=_state.candidate.hash,
        reasoning_root=r_t,
        ui_root=u_t,
        composite_root=h_t,
        environment_mode="standalone",
        slice_name=slice_name,
    )
    artifact_path = Path("artifacts/first_organism/attestation.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(_state.attestation.to_dict(), indent=2), encoding="utf-8")
    _state.log_phase("CHAIN", f"Artifact written to {artifact_path}", "INFO")

    # --- Emit Canonical PASS ---
    log_first_organism_pass(h_t)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
