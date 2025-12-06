import pytest
from unittest.mock import MagicMock, ANY
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

from backend.axiom_engine.bounds import SliceBounds
from backend.axiom_engine.pipeline import DerivationPipeline
from backend.axiom_engine.verification import StatementVerifier
from backend.frontier.curriculum import (
    CurriculumSystem,
    GateEvaluator,
    NormalizedMetrics,
    make_first_organism_slice,
)
from backend.ledger.ingest import LedgerIngestor
from backend.ledger.ui_events import capture_ui_event, materialize_ui_artifacts, ui_event_store
from normalization.canon import normalize
from backend.rfl.config import CurriculumSlice as RFLCurriculumSlice, RFLConfig
from backend.rfl.runner import AttestationInput, RFLRunner
from backend.repro.determinism import deterministic_timestamp_from_content

# --- Mocks for DB ---

class MockCursor:
    def __init__(self):
        self.executed = []
        self.return_values = []
        self._call_count = 0

    def execute(self, query, params=None):
        self.executed.append((query, params))
        self._call_count += 1
        
    def fetchone(self):
        # Return mock data based on the last executed query or a generic sequence
        last_query = self.executed[-1][0] if self.executed else ""
        
        if "INSERT INTO theories" in last_query:
            return (1, "pl")
        if "UPDATE theories" in last_query:
            return ("pl",)
        if "SELECT id FROM runs" in last_query:
            return None # Simulate new run
        if "INSERT INTO runs" in last_query:
            return (100,)
        if "INSERT INTO ledger_sequences" in last_query:
            return (1, 0, None, None, None, 100) # system_id, height, prev_block_id, prev_hash, prev_root, run_id
        if "SELECT system_id, height" in last_query:
            # Ledger sequence lookup
            return (1, 0, None, None, None, 100)
        if "INSERT INTO statements" in last_query:
            return (500, False) # id, is_axiom
        if "INSERT INTO proofs" in last_query:
            return (1000,) # id
        if "INSERT INTO blocks" in last_query:
            return (999,) # id
        
        return (1,) # Default fallback

@dataclass
class OrganismRunResult:
    ui_artifacts: List[str]
    derived_statement: str
    block_composite_root: str
    block_sealed_at: str
    rfl_source_root: str

def run_first_organism_chain(seed: int) -> OrganismRunResult:
    # 1. UI Event
    ui_event_store.clear()
    event = {
        "event_id": "evt-det-test",
        "action": "prove",
        "statement": "(p /\ q) -> p",
        "seed": seed # Include seed to vary input if needed, but here we want same input -> same output
    }
    capture_ui_event(event)
    ui_artifacts = materialize_ui_artifacts()

    # 2. Curriculum Gate
    slice_cfg = make_first_organism_slice()
    curriculum = CurriculumSystem(
        slug="pl",
        description="First organism curriculum",
        slices=[slice_cfg],
        active_index=0,
    )
    metrics = NormalizedMetrics(
        coverage_ci_lower=0.95,
        coverage_sample_size=48,
        abstention_rate_pct=12.5,
        attempt_mass=3200,
        slice_runtime_minutes=40.0,
        proof_velocity_pph=180.0,
        velocity_cv=0.04,
        backlog_fraction=0.12,
        attestation_hash="0000deadbeef",
    )
    gate_evaluator = GateEvaluator(metrics, curriculum.active_slice)
    gate_evaluator.evaluate()

    # 3. Derivation Pipeline
    bounds = SliceBounds(max_atoms=4, max_formula_depth=4, max_mp_depth=3, max_total=8)
    verifier = StatementVerifier(bounds)
    pipeline = DerivationPipeline(bounds, verifier)
    derivation_outcome = pipeline.run_step(existing=[])
    
    # Select a specific statement deterministically
    derived_statement = sorted(derivation_outcome.statements, key=lambda s: s.normalized)[0]

    # 4. Ledger Ingest (Real Logic with Mock DB)
    ingestor = LedgerIngestor()
    mock_cur = MockCursor()
    
    ingest_outcome = ingestor.ingest(
        cur=mock_cur,
        theory_name="Propositional Logic",
        ascii_statement=derived_statement.normalized,
        proof_text="lean abstained",
        prover="lean",
        status="abstain",
        module_name="first_organism",
        stdout="lean: proof incomplete",
        stderr="",
        derivation_rule=derived_statement.rule,
        derivation_depth=derived_statement.mp_depth,
        method="lean",
        duration_ms=12,
        truth_domain="pl",
        is_axiom=derived_statement.is_axiom,
        ui_events=ui_artifacts,
        sealed_by="integration-test",
    )

    # Extract sealed_at from the INSERT INTO blocks query params to verify it's deterministic
    # The block insert is likely the last or second to last insert
    block_insert_params = None
    for query, params in mock_cur.executed:
        if "INSERT INTO blocks" in query:
            block_insert_params = params
            break
            
    # block_insert_params: run_id, system_id, ..., sealed_at, ...
    # We need to find which param is sealed_at. 
    # In ingest.py: 
    # VALUES (..., %s, %s, %s, %s)
    # The code passes: ..., sealed_at, sealed_by, payload_hash, block_hash)
    # So sealed_at is roughly 4th from end.
    # Actually we can check the type or just rely on block_hash being deterministic which covers it.
    
    block_sealed_at = "unknown"
    if block_insert_params:
        # It's hard to index accurately without counting carefully, but we can look for datetime
        for p in block_insert_params:
            if isinstance(p, datetime): # It might be the sealed_at
                # There's also NOW() (but we replaced that with python datetime)
                # Wait, _ensure_run used to use NOW(), now it uses a param.
                # _seal_block uses sealed_at param.
                block_sealed_at = p.isoformat()
    
    # 5. RFL Metabolism
    # Mocking RFL experiment runner to avoid complex setups
    rfl_config = RFLConfig(
        experiment_id="first_organism",
        num_runs=1,
        random_seed=1,
        system_id=1,
        derive_steps=1,
        max_breadth=1,
        max_total=1,
        depth_max=1,
        database_url="postgresql://example.test",
        redis_url="redis://example.test",
        artifacts_dir=".",
        curriculum=[
            RFLCurriculumSlice(
                name=slice_cfg.name,
                start_run=1,
                end_run=1,
                derive_steps=1,
                max_breadth=1,
                max_total=1,
                depth_max=1,
            )
        ],
        dual_attestation=True,
    )
    
    runner = RFLRunner(rfl_config)
    runner.experiment = MagicMock() # Mock the experiment execution
    runner.experiment.run.return_value = MagicMock(status="success", statement_hashes=[])
    runner.coverage_tracker = MagicMock()
    
    attestation_input = AttestationInput(
        composite_root=ingest_outcome.block.composite_root,
        reasoning_root=ingest_outcome.block.reasoning_root,
        ui_root=ingest_outcome.block.composite_root, # Hack for test, usually separate
        abstention_rate=1.0,
        abstention_mass=1.0,
        slice_name=slice_cfg.name,
        metadata={
            "block": ingest_outcome.block.number,
        },
    )
    
    rfl_result = runner.run_with_attestation(attestation_input)
    
    return OrganismRunResult(
        ui_artifacts=ui_artifacts,
        derived_statement=derived_statement.normalized,
        block_composite_root=ingest_outcome.block.composite_root,
        block_sealed_at=block_sealed_at,
        rfl_source_root=rfl_result.source_root
    )

def test_first_organism_determinism():
    """
    Verify that the entire First Organism path is bit-for-bit deterministic.
    """
    print("\nRunning Run 1...")
    result1 = run_first_organism_chain(seed=42)
    
    print("\nRunning Run 2...")
    result2 = run_first_organism_chain(seed=42)
    
    print("\nComparing results...")
    assert result1.ui_artifacts == result2.ui_artifacts
    assert result1.derived_statement == result2.derived_statement
    assert result1.block_composite_root == result2.block_composite_root
    assert result1.block_sealed_at == result2.block_sealed_at
    assert result1.rfl_source_root == result2.rfl_source_root
    
    print(f"Sealed At: {result1.block_sealed_at}")
    print(f"Composite Root: {result1.block_composite_root}")
    
    # Verify the sealed_at is NOT just the current time (it should be fixed for a given content)
    # We can try a third run with DIFFERENT content to see if it changes?
    # Or just ensure it's consistent.

