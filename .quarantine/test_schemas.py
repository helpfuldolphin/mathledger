#!/usr/bin/env python3
"""
Test script to verify the Pydantic schemas work correctly.
This tests the schema validation and serialization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from api.schemas import (
    Metrics, BlockSummary, BlockWithProofs, Lemma, StatementWithProofs,
    Theory, TheoryWithStats, ExportManifest, ThroughputMetrics, FrontierMetrics,
    FailuresByClass, StatementMetrics, ProofMetrics, BlockMetrics, LemmaMetrics,
    QueueMetrics, StatementBase, ProofBase
)
from datetime import datetime

def test_basic_schemas():
    """Test basic schema creation and validation."""
    print("Testing basic schemas...")

    # Test StatementBase
    stmt = StatementBase(
        id=1,
        hash="abc123",
        text="p -> p",
        system_id=1,
        derivation_rule="modus_ponens",
        derivation_depth=2,
        created_at=datetime.now()
    )
    print(f"✓ StatementBase: {stmt.text}")

    # Test ProofBase
    proof = ProofBase(
        id=1,
        statement_id=1,
        prover="lean4",
        method="tactics",
        status="success",
        derivation_rule="modus_ponens",
        duration_ms=150,
        created_at=datetime.now()
    )
    print(f"✓ ProofBase: {proof.prover}")

    # Test BlockSummary
    block = BlockSummary(
        id=1,
        run_id=1,
        system_id=1,
        root_hash="def456",
        counts={"total_statements": 10, "proven": 8},
        created_at=datetime.now()
    )
    print(f"✓ BlockSummary: {block.root_hash}")

    # Test Theory
    theory = Theory(
        id=1,
        name="Propositional Logic",
        slug="pl",
        parent_id=None,
        created_at=datetime.now()
    )
    print(f"✓ Theory: {theory.name}")

    # Test Lemma
    lemma = Lemma(
        id=1,
        statement=stmt,
        usage_count=5,
        last_used=datetime.now()
    )
    print(f"✓ Lemma: usage_count={lemma.usage_count}")

    print("✓ All basic schemas work correctly!")

def test_complex_schemas():
    """Test complex nested schemas."""
    print("\nTesting complex schemas...")

    # Test Metrics
    metrics = Metrics(
        statements=StatementMetrics(
            total=100,
            axioms=5,
            derived=95,
            max_depth=10
        ),
        proofs=ProofMetrics(
            by_status={"success": 80, "failed": 20},
            by_prover={"lean4": 70, "coq": 10},
            recent_hour=15,
            success_rate=85.5
        ),
        derivation_rules={"modus_ponens": 50, "axiom": 5},
        blocks=BlockMetrics(
            total=5,
            latest_id=5,
            latest_counts={"statements": 20}
        ),
        lemmas=LemmaMetrics(
            total=25,
            top=[{"text": "p -> p", "usage_count": 10}]
        ),
        throughput=ThroughputMetrics(
            proofs_per_min=2.5,
            statements_per_min=1.8
        ),
        frontier=FrontierMetrics(
            depth_max=10,
            queue_backlog=5
        ),
        failures_by_class=FailuresByClass(
            LEAN_TIMEOUT=2,
            LEAN_COMPILE=1,
            LEAN_RUNTIME=0,
            DERIVATION_ERROR=1,
            OTHER=0
        ),
        queue=QueueMetrics(length=5)
    )

    # Test JSON serialization
    json_data = metrics.model_dump()
    print(f"✓ Metrics serialized: {len(json_data)} top-level fields")

    # Test StatementWithProofs
    stmt_with_proofs = StatementWithProofs(
        id=1,
        hash="abc123",
        text="p -> p",
        system_id=1,
        derivation_rule="modus_ponens",
        derivation_depth=2,
        created_at=datetime.now(),
        proofs=[
            ProofBase(
                id=1,
                statement_id=1,
                prover="lean4",
                method="tactics",
                status="success",
                derivation_rule="modus_ponens",
                duration_ms=150,
                created_at=datetime.now()
            )
        ],
        parents=["p", "p -> q"]
    )
    print(f"✓ StatementWithProofs: {len(stmt_with_proofs.proofs)} proofs, {len(stmt_with_proofs.parents)} parents")

    print("✓ All complex schemas work correctly!")

def test_validation():
    """Test schema validation."""
    print("\nTesting validation...")

    try:
        # This should work
        stmt = StatementBase(
            id=1,
            hash="abc123",
            text="p -> p",
            system_id=1,
            created_at=datetime.now()
        )
        print("✓ Valid StatementBase created")

        # This should fail validation
        try:
            invalid_stmt = StatementBase(
                id="not_an_int",  # Should be int
                hash="abc123",
                text="p -> p",
                system_id=1,
                created_at=datetime.now()
            )
            print("❌ Invalid StatementBase should have failed validation")
        except Exception as e:
            print(f"✓ Validation correctly caught error: {type(e).__name__}")

    except Exception as e:
        print(f"❌ Unexpected error in validation test: {e}")
        raise

if __name__ == "__main__":
    print("Testing MathLedger Pydantic schemas...")
    print("=" * 50)

    try:
        test_basic_schemas()
        test_complex_schemas()
        test_validation()
        print("\n" + "=" * 50)
        print("🎉 All schema tests passed! Pydantic models are working correctly.")
    except Exception as e:
        print(f"\n❌ Schema test failed: {e}")
        raise
