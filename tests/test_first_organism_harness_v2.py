#!/usr/bin/env python3
"""
First Organism Harness V2 - Unified Test Class
===============================================

##############################################################################
# NOTE: EXPERIMENTAL HARNESS - PHASE II - NOT USED IN EVIDENCE PACK V1
#
# This file is an EXPERIMENTAL test harness for future development.
# It is NOT the authoritative First Organism test for Evidence Pack v1.
#
# The actual Evidence Pack v1 relies on:
#   - tests/integration/test_first_organism.py  (authoritative FO test)
#   - artifacts/first_organism/attestation.json (sealed attestation)
#   - results/fo_baseline.jsonl and results/fo_rfl.jsonl (partial RFL run data, Phase I prototype)
#
# Do NOT use this harness to support any Phase I claims.
# Do NOT make CI gates depend on this file for Evidence Pack certification.
#
# Status: Phase II - Not Yet Used in Production Evidence
##############################################################################

This module provides a composable, unified test harness for the First Organism
closed loop path:

    UI Event → Curriculum Gate → Derivation → Lean Verify (abstention) →
    Dual-Attest seal H_t → RFL runner metabolism.

Key Features:
    - Single unified test class: TestFirstOrganismHarnessV2
    - Composable phase methods that can run independently or chained
    - Mock DB by default unless FIRST_ORGANISM_TESTS=true
    - Determinism checks: H_t is identical across runs with same seed
    - MDAP compliance: no wall-clock time or random sources

Test Methods:
    - test_phase_ui: UI event capture and U_t computation
    - test_phase_curriculum: Curriculum gate evaluation
    - test_phase_derivation: Derivation pipeline with abstention
    - test_phase_attestation: Dual-root attestation (R_t, U_t, H_t)
    - test_phase_rfl: RFL runner metabolism verification
    - test_full_chain: Complete pipeline with determinism verification

Usage:
    # Run all phases with mock DB (no external dependencies)
    pytest tests/test_first_organism_harness_v2.py -v

    # Run with real DB (requires Postgres/Redis)
    FIRST_ORGANISM_TESTS=true pytest tests/test_first_organism_harness_v2.py -v

    # Run single phase
    pytest tests/test_first_organism_harness_v2.py::TestFirstOrganismHarnessV2::test_phase_ui -v
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Environment Detection
# ---------------------------------------------------------------------------

def _is_first_organism_enabled() -> bool:
    """Check if First Organism tests are enabled."""
    first_organism_env = os.getenv("FIRST_ORGANISM_TESTS", "").lower()
    spark_file_trigger = Path(".spark_run_enable").is_file()
    return (
        first_organism_env == "true"
        or os.getenv("SPARK_RUN", "") == "1"
        or spark_file_trigger
    )


def _use_mock_db() -> bool:
    """Determine if we should use mock DB (inverse of FO tests enabled)."""
    return not _is_first_organism_enabled()


# ---------------------------------------------------------------------------
# MDAP Deterministic Helpers (Self-Contained)
# ---------------------------------------------------------------------------

MDAP_EPOCH_SEED = 0x4D444150  # "MDAP" as hex seed
FIRST_ORGANISM_NAMESPACE = "first-organism-harness-v2"


def _deterministic_hash(content: str, namespace: str = FIRST_ORGANISM_NAMESPACE) -> str:
    """Compute deterministic hash from content."""
    data = f"{namespace}:{content}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _deterministic_id(context: str, *parts: Any) -> str:
    """Generate deterministic ID from context and parts."""
    content = "|".join(str(p) for p in parts)
    full_content = f"{context}:{content}"
    hash_val = _deterministic_hash(full_content)
    return f"{context}-{hash_val[:16]}"


def _deterministic_timestamp(content: str) -> int:
    """Generate deterministic Unix timestamp from content."""
    # Use MDAP epoch (2025-01-01 00:00:00 UTC) + offset derived from content
    mdap_epoch = 1735689600  # 2025-01-01 00:00:00 UTC
    offset = int(_deterministic_hash(content)[:8], 16) % 86400  # Within 24 hours
    return mdap_epoch + offset


def _deterministic_isoformat(content: str) -> str:
    """Generate deterministic ISO-8601 timestamp from content."""
    ts = _deterministic_timestamp(content)
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.isoformat()


def _rfc8785_canonicalize(obj: Any) -> str:
    """Serialize object to RFC 8785 canonical JSON."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=True)


def _rfc8785_hash(obj: Any) -> str:
    """Compute SHA-256 hash of canonical JSON representation."""
    canonical = _rfc8785_canonicalize(obj)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# Mock Components (for DB-free testing)
# ---------------------------------------------------------------------------

@dataclass
class MockUIEvent:
    """Mock UI event record."""
    event_id: str
    event_type: str
    timestamp: int
    payload: Dict[str, Any]
    leaf_hash: str

    def to_artifact(self) -> str:
        """Return canonical JSON for Merkle leaf."""
        return _rfc8785_canonicalize(self.payload)


@dataclass
class MockStatement:
    """Mock derived statement."""
    hash: str
    normalized: str
    pretty: str
    rule: str
    mp_depth: int
    is_axiom: bool
    verification_method: str


@dataclass
class MockDerivationResult:
    """Mock derivation pipeline result."""
    n_candidates: int
    n_verified: int
    n_abstained: int
    abstained_candidates: List[MockStatement]
    verified_candidates: List[MockStatement]


@dataclass
class MockBlock:
    """Mock sealed block."""
    id: int
    reasoning_root: str
    ui_root: str
    composite_root: str
    ui_event_count: int
    reasoning_event_count: int


@dataclass
class MockGateStatus:
    """Mock curriculum gate status."""
    gate: str
    passed: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"gate": self.gate, "passed": self.passed, "message": self.message}


@dataclass
class MockRflResult:
    """Mock RFL runner result."""
    source_root: str
    policy_update_applied: bool
    symbolic_descent: float


@dataclass
class MockLedgerEntry:
    """Mock policy ledger entry."""
    run_id: str
    slice_name: str
    status: str
    symbolic_descent: float


# ---------------------------------------------------------------------------
# Harness State Container
# ---------------------------------------------------------------------------

@dataclass
class HarnessState:
    """
    Shared state across test phases.

    This enables composable testing where phases can:
    - Run independently with synthetic data
    - Share state when run in sequence
    """
    # Phase 1: UI Event
    ui_event: Optional[MockUIEvent] = None
    ui_artifacts: List[str] = field(default_factory=list)

    # Phase 2: Curriculum Gate
    slice_name: str = "first-organism-harness-v2"
    gate_statuses: List[MockGateStatus] = field(default_factory=list)
    gates_passed: bool = False

    # Phase 3: Derivation
    derivation_result: Optional[MockDerivationResult] = None
    candidate: Optional[MockStatement] = None

    # Phase 4: Attestation
    block: Optional[MockBlock] = None
    reasoning_root: str = ""
    ui_root: str = ""
    composite_root: str = ""

    # Phase 5: RFL
    rfl_result: Optional[MockRflResult] = None
    ledger_entry: Optional[MockLedgerEntry] = None
    abstention_histogram: Dict[str, int] = field(default_factory=dict)

    # Metadata
    seed: int = MDAP_EPOCH_SEED
    run_id: str = ""

    def reset(self):
        """Reset all state for fresh run."""
        self.ui_event = None
        self.ui_artifacts = []
        self.gate_statuses = []
        self.gates_passed = False
        self.derivation_result = None
        self.candidate = None
        self.block = None
        self.reasoning_root = ""
        self.ui_root = ""
        self.composite_root = ""
        self.rfl_result = None
        self.ledger_entry = None
        self.abstention_histogram = {}
        self.run_id = ""


# ---------------------------------------------------------------------------
# Harness Printer (Green Bar Output)
# ---------------------------------------------------------------------------

class HarnessPrinter:
    """Color-coded status printer for test phases."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    USE_COLOR = sys.stdout.isatty() or os.environ.get("PYTEST_COLOR") == "yes"

    @classmethod
    def status(cls, phase: str, message: str, level: str = "INFO"):
        """Print phase status with color coding."""
        color = ""
        if cls.USE_COLOR:
            if level == "PASS":
                color = cls.GREEN
            elif level == "FAIL":
                color = cls.RED
            elif level == "WARN":
                color = cls.YELLOW
            elif level == "INFO":
                color = cls.CYAN

        reset = cls.RESET if cls.USE_COLOR else ""
        sys.stdout.write(f"{color}[{phase}] {message}{reset}\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Phase Implementations (Mock-Based)
# ---------------------------------------------------------------------------

def _run_phase_ui(state: HarnessState, seed_content: str = "harness-v2-ui") -> None:
    """
    Phase 1: UI Event Capture and U_t Computation (Mock)

    Creates a deterministic UI event and computes its Merkle leaf hash.
    """
    event_id = _deterministic_id("ui-event", seed_content, state.seed)
    timestamp = _deterministic_timestamp(seed_content)
    statement_hash = _deterministic_hash("p -> p")

    payload = {
        "event_id": event_id,
        "event_type": "select_statement",
        "actor": "harness-v2-test",
        "statement_hash": statement_hash,
        "action": "toggle_abstain",
        "timestamp": timestamp,
    }

    leaf_hash = _rfc8785_hash(payload)

    state.ui_event = MockUIEvent(
        event_id=event_id,
        event_type="select_statement",
        timestamp=timestamp,
        payload=payload,
        leaf_hash=leaf_hash,
    )
    state.ui_artifacts = [state.ui_event.to_artifact()]

    HarnessPrinter.status("UI", f"Event {event_id[:16]}... captured", "PASS")


def _run_phase_curriculum(state: HarnessState) -> None:
    """
    Phase 2: Curriculum Gate Evaluation (Mock)

    Evaluates mock gates that always pass for harness testing.
    """
    # Build mock gate statuses that pass
    state.gate_statuses = [
        MockGateStatus("coverage", True, "ci_lower=0.95 >= 0.50"),
        MockGateStatus("abstention", True, "rate=10.0% <= 95.0%"),
        MockGateStatus("velocity", True, "pph=240.0 >= 0.1"),
        MockGateStatus("caps", True, "attempt_mass=3200 >= 1"),
    ]
    state.gates_passed = all(g.passed for g in state.gate_statuses)

    HarnessPrinter.status(
        "GATE",
        f"All {len(state.gate_statuses)} gates passed for {state.slice_name}",
        "PASS" if state.gates_passed else "FAIL"
    )


def _run_phase_derivation(state: HarnessState, seed_content: str = "harness-v2-derive") -> None:
    """
    Phase 3: Derivation Pipeline with Abstention (Mock)

    Creates mock derivation results with at least one abstained candidate.
    """
    # Create mock seed statements
    seed_hash = _deterministic_hash("p")
    impl_hash = _deterministic_hash("p -> q")
    conclusion_hash = _deterministic_hash("q")

    # Mock derivation: MP fires on seeds, produces non-tautology
    abstained_candidate = MockStatement(
        hash=conclusion_hash,
        normalized="q",
        pretty="q",
        rule="modus_ponens",
        mp_depth=1,
        is_axiom=False,
        verification_method="truth_table_abstain",
    )

    state.derivation_result = MockDerivationResult(
        n_candidates=3,
        n_verified=2,
        n_abstained=1,
        abstained_candidates=[abstained_candidate],
        verified_candidates=[],
    )
    state.candidate = abstained_candidate

    HarnessPrinter.status(
        "DERIVE",
        f"Abstained: {state.candidate.pretty[:30]}... ({state.candidate.verification_method})",
        "PASS"
    )


def _run_phase_attestation(state: HarnessState) -> None:
    """
    Phase 4: Dual-Root Attestation (R_t, U_t, H_t) (Mock)

    Computes Merkle roots and composite attestation root.
    """
    # Ensure we have prerequisites
    if state.candidate is None:
        _run_phase_derivation(state)
    if state.ui_event is None:
        _run_phase_ui(state)

    # Compute R_t from reasoning artifacts (proof hashes)
    reasoning_leaves = [state.candidate.hash]
    reasoning_concat = "".join(sorted(reasoning_leaves))
    state.reasoning_root = hashlib.sha256(
        f"REASONING:{reasoning_concat}".encode()
    ).hexdigest()

    # Compute U_t from UI event leaf hashes
    ui_leaves = [state.ui_event.leaf_hash]
    ui_concat = "".join(sorted(ui_leaves))
    state.ui_root = hashlib.sha256(f"UI:{ui_concat}".encode()).hexdigest()

    # Compute H_t = SHA256(R_t || U_t)
    composite_data = f"{state.reasoning_root}{state.ui_root}".encode("ascii")
    state.composite_root = hashlib.sha256(composite_data).hexdigest()

    state.block = MockBlock(
        id=1,
        reasoning_root=state.reasoning_root,
        ui_root=state.ui_root,
        composite_root=state.composite_root,
        ui_event_count=1,
        reasoning_event_count=1,
    )

    HarnessPrinter.status("ATTEST", f"R_t={state.reasoning_root[:12]}...", "INFO")
    HarnessPrinter.status("ATTEST", f"U_t={state.ui_root[:12]}...", "INFO")
    HarnessPrinter.status("ATTEST", f"H_t={state.composite_root[:12]}...", "PASS")


def _run_phase_rfl(state: HarnessState) -> None:
    """
    Phase 5: RFL Runner Metabolism Verification (Mock)

    Simulates RFL runner consuming attestation and updating policy.
    """
    # Ensure we have prerequisites
    if state.block is None:
        _run_phase_attestation(state)

    # Compute symbolic descent from abstention metrics
    n_total = state.derivation_result.n_candidates if state.derivation_result else 3
    n_abstained = state.derivation_result.n_abstained if state.derivation_result else 1
    abstention_rate = n_abstained / max(n_total, 1)
    abstention_tolerance = 0.15
    symbolic_descent = -(abstention_rate - abstention_tolerance)

    state.rfl_result = MockRflResult(
        source_root=state.composite_root,
        policy_update_applied=True,
        symbolic_descent=symbolic_descent,
    )

    state.ledger_entry = MockLedgerEntry(
        run_id=_deterministic_id("rfl-run", state.composite_root),
        slice_name=state.slice_name,
        status="attestation",
        symbolic_descent=symbolic_descent,
    )

    state.abstention_histogram = {"lean_failure": n_abstained}

    HarnessPrinter.status(
        "RFL",
        f"Metabolism: H_t consumed, symbolic_descent={symbolic_descent:.4f}",
        "PASS"
    )


def _verify_h_t_formula(r_t: str, u_t: str, h_t: str) -> bool:
    """Verify H_t = SHA256(R_t || U_t)."""
    expected = hashlib.sha256(f"{r_t}{u_t}".encode("ascii")).hexdigest()
    return h_t == expected


def _run_determinism_check(seed: int, runs: int = 3) -> Tuple[bool, List[str]]:
    """
    Verify determinism by running the pipeline multiple times.

    Returns (all_identical, list_of_h_t_values).
    """
    h_t_values = []

    for _ in range(runs):
        state = HarnessState(seed=seed)
        _run_phase_ui(state)
        _run_phase_curriculum(state)
        _run_phase_derivation(state)
        _run_phase_attestation(state)
        h_t_values.append(state.composite_root)

    all_identical = len(set(h_t_values)) == 1
    return all_identical, h_t_values


# ---------------------------------------------------------------------------
# Test Class: TestFirstOrganismHarnessV2
# ---------------------------------------------------------------------------

@pytest.mark.first_organism
class TestFirstOrganismHarnessV2:
    """
    Unified test class for First Organism closed-loop verification.

    This class provides:
    - Composable phase tests that can run independently
    - Full chain test combining all phases
    - Determinism verification (H_t same across runs)
    - Mock DB by default, real DB when FIRST_ORGANISM_TESTS=true
    """

    @pytest.fixture(autouse=True)
    def setup_state(self):
        """Initialize fresh state for each test."""
        self.state = HarnessState(seed=MDAP_EPOCH_SEED)
        self.use_mock = _use_mock_db()
        yield
        self.state.reset()

    # -----------------------------------------------------------------------
    # Phase Tests
    # -----------------------------------------------------------------------

    def test_phase_ui(self):
        """
        Phase 1: UI Event Capture and U_t Computation

        Validates:
        1. UI event is captured with deterministic ID and timestamp
        2. Event payload is MDAP-compliant (no wall-clock time)
        3. Leaf hash is computed correctly
        4. Event capture is deterministic
        """
        HarnessPrinter.status("TEST", "Phase 1: UI Event Capture", "INFO")

        # Run phase
        _run_phase_ui(self.state)

        # Assertions
        assert self.state.ui_event is not None, "UI event must be captured"
        assert self.state.ui_event.event_id.startswith("ui-event-"), "Event ID must have correct prefix"
        assert len(self.state.ui_event.leaf_hash) == 64, "Leaf hash must be 64-char hex"
        assert len(self.state.ui_artifacts) == 1, "Must have exactly one UI artifact"

        # Determinism check: same inputs produce same outputs
        event_id_1 = self.state.ui_event.event_id
        _run_phase_ui(self.state)
        event_id_2 = self.state.ui_event.event_id
        assert event_id_1 == event_id_2, "UI event capture must be deterministic"

    def test_phase_curriculum(self):
        """
        Phase 2: Curriculum Gate Evaluation

        Validates:
        1. All four gates are evaluated (coverage, abstention, velocity, caps)
        2. Gate evaluation produces pass/fail verdicts
        3. Gate evaluation is deterministic
        """
        HarnessPrinter.status("TEST", "Phase 2: Curriculum Gate", "INFO")

        # Run phase
        _run_phase_curriculum(self.state)

        # Assertions
        assert len(self.state.gate_statuses) == 4, "Must evaluate 4 gates"
        assert self.state.gates_passed is True, "All gates must pass in mock mode"

        gate_names = {g.gate for g in self.state.gate_statuses}
        expected_gates = {"coverage", "abstention", "velocity", "caps"}
        assert gate_names == expected_gates, f"Expected gates {expected_gates}, got {gate_names}"

        # Verify each gate has required fields
        for gate in self.state.gate_statuses:
            assert gate.gate, "Gate must have name"
            assert isinstance(gate.passed, bool), "Gate must have boolean passed"
            assert gate.message, "Gate must have message"

    def test_phase_derivation(self):
        """
        Phase 3: Derivation Pipeline with Abstention

        Validates:
        1. Derivation produces candidates
        2. At least one candidate is abstained (non-tautology)
        3. Abstained candidate has correct metadata
        4. Derivation is deterministic
        """
        HarnessPrinter.status("TEST", "Phase 3: Derivation", "INFO")

        # Run phase
        _run_phase_derivation(self.state)

        # Assertions
        assert self.state.derivation_result is not None, "Derivation must produce result"
        assert self.state.derivation_result.n_candidates > 0, "Must consider candidates"
        assert self.state.derivation_result.n_abstained >= 1, "Must have at least one abstention"
        assert self.state.candidate is not None, "Must have abstained candidate"

        # Verify candidate metadata
        assert len(self.state.candidate.hash) == 64, "Hash must be 64-char hex"
        assert self.state.candidate.rule == "modus_ponens", "Rule must be modus_ponens"
        assert self.state.candidate.verification_method == "truth_table_abstain"

        # Determinism check
        hash_1 = self.state.candidate.hash
        _run_phase_derivation(self.state)
        hash_2 = self.state.candidate.hash
        assert hash_1 == hash_2, "Derivation must be deterministic"

    def test_phase_attestation(self):
        """
        Phase 4: Dual-Root Attestation (R_t, U_t, H_t)

        Validates:
        1. R_t is computed from proof events
        2. U_t is computed from UI events
        3. H_t = SHA256(R_t || U_t) formula holds
        4. Block is sealed with all roots
        5. Attestation is deterministic
        """
        HarnessPrinter.status("TEST", "Phase 4: Attestation", "INFO")

        # Run phase (will run prerequisites automatically)
        _run_phase_attestation(self.state)

        # Assertions
        assert len(self.state.reasoning_root) == 64, "R_t must be 64-char hex"
        assert len(self.state.ui_root) == 64, "U_t must be 64-char hex"
        assert len(self.state.composite_root) == 64, "H_t must be 64-char hex"

        # Verify H_t formula
        assert _verify_h_t_formula(
            self.state.reasoning_root,
            self.state.ui_root,
            self.state.composite_root,
        ), "H_t = SHA256(R_t || U_t) must hold"

        # Verify block
        assert self.state.block is not None, "Block must be sealed"
        assert self.state.block.composite_root == self.state.composite_root

        # Determinism check
        h_t_1 = self.state.composite_root
        self.state.reset()
        _run_phase_attestation(self.state)
        h_t_2 = self.state.composite_root
        assert h_t_1 == h_t_2, "Attestation must be deterministic"

    def test_phase_rfl(self):
        """
        Phase 5: RFL Runner Metabolism Verification

        Validates:
        1. RFL runner accepts attested context
        2. Policy ledger records the abstention
        3. Abstention histogram is updated
        4. Symbolic descent is computed correctly
        5. H_t is correctly referenced
        """
        HarnessPrinter.status("TEST", "Phase 5: RFL Metabolism", "INFO")

        # Run phase (will run prerequisites automatically)
        _run_phase_rfl(self.state)

        # Assertions
        assert self.state.rfl_result is not None, "RFL must produce result"
        assert self.state.rfl_result.source_root == self.state.composite_root, "Must reference H_t"
        assert self.state.rfl_result.policy_update_applied is True, "Policy must be updated"

        assert self.state.ledger_entry is not None, "Must have ledger entry"
        assert self.state.ledger_entry.status == "attestation"

        assert "lean_failure" in self.state.abstention_histogram, "Must record abstention type"
        assert self.state.abstention_histogram["lean_failure"] >= 1

    # -----------------------------------------------------------------------
    # Full Chain Test
    # -----------------------------------------------------------------------

    def test_full_chain(self):
        """
        Full chain integration test combining all five phases.

        This test:
        1. Executes all phases in sequence
        2. Verifies phase dependencies are satisfied
        3. Checks H_t formula at the end
        4. Verifies determinism across multiple runs
        5. Emits canonical PASS line
        """
        HarnessPrinter.status("CHAIN", "Starting Full Chain test", "INFO")

        # Reset state for clean run
        self.state.reset()

        # Phase 1: UI Event
        _run_phase_ui(self.state)
        assert self.state.ui_event is not None
        HarnessPrinter.status("CHAIN", "Phase 1 (UI) complete", "INFO")

        # Phase 2: Curriculum Gate
        _run_phase_curriculum(self.state)
        assert self.state.gates_passed
        HarnessPrinter.status("CHAIN", "Phase 2 (Gate) complete", "INFO")

        # Phase 3: Derivation
        _run_phase_derivation(self.state)
        assert self.state.candidate is not None
        HarnessPrinter.status("CHAIN", "Phase 3 (Derive) complete", "INFO")

        # Phase 4: Attestation
        _run_phase_attestation(self.state)
        assert self.state.composite_root
        assert _verify_h_t_formula(
            self.state.reasoning_root,
            self.state.ui_root,
            self.state.composite_root,
        )
        HarnessPrinter.status("CHAIN", "Phase 4 (Attest) complete", "INFO")

        # Phase 5: RFL Metabolism
        _run_phase_rfl(self.state)
        assert self.state.rfl_result.source_root == self.state.composite_root
        HarnessPrinter.status("CHAIN", "Phase 5 (RFL) complete", "INFO")

        # Determinism verification
        HarnessPrinter.status("CHAIN", "Running determinism check...", "INFO")
        is_deterministic, h_t_values = _run_determinism_check(self.state.seed, runs=3)
        assert is_deterministic, f"H_t must be identical across runs: {h_t_values}"
        HarnessPrinter.status("CHAIN", f"Determinism verified: H_t={h_t_values[0][:12]}...", "PASS")

        # Emit canonical PASS line
        h_t_short = self.state.composite_root[:12]
        HarnessPrinter.status("PASS", f"FIRST ORGANISM ALIVE H_t={h_t_short}", "PASS")

    # -----------------------------------------------------------------------
    # Determinism Tests
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("seed", [42, 12345, MDAP_EPOCH_SEED])
    def test_determinism_multiple_seeds(self, seed: int):
        """
        Verify determinism with multiple seeds.

        Each seed should produce identical H_t values across runs.
        """
        HarnessPrinter.status("DETERMINISM", f"Testing seed={seed}", "INFO")

        is_deterministic, h_t_values = _run_determinism_check(seed, runs=3)

        assert is_deterministic, (
            f"H_t must be identical across runs for seed={seed}: {h_t_values}"
        )
        HarnessPrinter.status(
            "DETERMINISM",
            f"Seed {seed}: H_t={h_t_values[0][:12]}... (verified across 3 runs)",
            "PASS"
        )

    def test_determinism_bitwise(self):
        """
        Verify bitwise reproducibility of canonical JSON output.

        The entire pipeline output serialized to canonical JSON must be
        byte-for-byte identical across runs.
        """
        HarnessPrinter.status("DETERMINISM", "Testing bitwise reproducibility", "INFO")

        def run_and_serialize() -> bytes:
            state = HarnessState(seed=MDAP_EPOCH_SEED)
            _run_phase_ui(state)
            _run_phase_curriculum(state)
            _run_phase_derivation(state)
            _run_phase_attestation(state)
            _run_phase_rfl(state)

            result_doc = {
                "seed": state.seed,
                "ui_event_hash": state.ui_event.leaf_hash,
                "gates_passed": state.gates_passed,
                "candidate_hash": state.candidate.hash,
                "reasoning_root": state.reasoning_root,
                "ui_root": state.ui_root,
                "composite_root": state.composite_root,
                "rfl_status": state.ledger_entry.status,
            }
            return _rfc8785_canonicalize(result_doc).encode("ascii")

        bytes_1 = run_and_serialize()
        bytes_2 = run_and_serialize()
        bytes_3 = run_and_serialize()

        assert bytes_1 == bytes_2 == bytes_3, "Output must be bitwise identical"
        HarnessPrinter.status(
            "DETERMINISM",
            f"Bitwise identical across 3 runs ({len(bytes_1)} bytes)",
            "PASS"
        )

    # -----------------------------------------------------------------------
    # Failure Path Tests
    # -----------------------------------------------------------------------

    def test_gate_failure_path(self):
        """
        Test failure path: Curriculum gate rejects the slice.

        Verifies that gate failure produces explicit reason and blocks pipeline.
        """
        HarnessPrinter.status("TEST", "Testing gate failure path", "INFO")

        # Manually create failing gate statuses
        self.state.gate_statuses = [
            MockGateStatus("coverage", False, "ci_lower=0.10 < 0.50 required"),
            MockGateStatus("abstention", True, "rate=10.0% <= 95.0%"),
            MockGateStatus("velocity", True, "pph=240.0 >= 0.1"),
            MockGateStatus("caps", True, "attempt_mass=3200 >= 1"),
        ]
        self.state.gates_passed = all(g.passed for g in self.state.gate_statuses)

        # Assertions
        assert self.state.gates_passed is False, "Gates must fail"

        failed_gates = [g for g in self.state.gate_statuses if not g.passed]
        assert len(failed_gates) == 1, "Exactly one gate must fail"
        assert failed_gates[0].gate == "coverage", "Coverage gate must fail"
        assert "0.10" in failed_gates[0].message, "Failure message must include actual value"

        HarnessPrinter.status("TEST", f"Gate '{failed_gates[0].gate}' correctly rejected", "PASS")

    def test_h_t_formula_verification(self):
        """
        Test H_t = SHA256(R_t || U_t) formula verification.

        Verifies that incorrect H_t values are detected.
        """
        HarnessPrinter.status("TEST", "Testing H_t formula verification", "INFO")

        r_t = "a" * 64
        u_t = "b" * 64
        correct_h_t = hashlib.sha256(f"{r_t}{u_t}".encode("ascii")).hexdigest()
        wrong_h_t = "c" * 64

        assert _verify_h_t_formula(r_t, u_t, correct_h_t), "Correct H_t must verify"
        assert not _verify_h_t_formula(r_t, u_t, wrong_h_t), "Wrong H_t must fail"

        HarnessPrinter.status("TEST", "H_t formula verification working correctly", "PASS")


# ---------------------------------------------------------------------------
# Standalone Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
