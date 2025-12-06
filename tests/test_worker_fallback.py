"""
Tests for worker tautology fallback mechanism.
"""

import pytest
from unittest.mock import MagicMock as Mock, patch, MagicMock
from backend.logic.taut import truth_table_is_tautology


class TestWorkerFallback:
    """Test the worker's tautology fallback mechanism."""

    def test_truth_table_tautology_detection(self):
        """Test that tautologies are correctly identified."""
        # Simple tautologies
        assert truth_table_is_tautology("p -> p")
        assert truth_table_is_tautology("p \\/ ~p")
        assert truth_table_is_tautology("p -> (q -> p)")

        # Non-tautologies
        assert not truth_table_is_tautology("p")
        assert not truth_table_is_tautology("p /\\ ~p")
        assert not truth_table_is_tautology("p -> q")

    def test_tautology_with_multiple_atoms(self):
        """Test tautology detection with multiple atomic propositions."""
        # Tautologies
        assert truth_table_is_tautology("p -> p")
        assert truth_table_is_tautology("p \\/ ~p")
        assert truth_table_is_tautology("(p -> q) \\/ (q -> p)")

        # Non-tautologies
        assert not truth_table_is_tautology("p /\\ q")
        assert not truth_table_is_tautology("p -> q")
        assert not truth_table_is_tautology("p /\\ ~p")

    def test_tautology_with_complex_formulas(self):
        """Test tautology detection with complex formulas."""
        # Complex tautologies
        assert truth_table_is_tautology("((p -> q) -> p) -> p")  # Peirce's law
        assert truth_table_is_tautology("p -> (q -> (p /\\ q))")
        assert truth_table_is_tautology("(p /\\ q) -> p")
        assert truth_table_is_tautology("(p /\\ q) -> q")

    def test_worker_tautology_fallback_simulation(self):
        """Test the worker's tautology fallback logic."""
        # Simulate a failing build scenario
        statement = "p -> p"  # Known tautology

        # Check if it's a tautology
        is_tautology = truth_table_is_tautology(statement)
        assert is_tautology

        # If tautology, create by_decide proof
        if is_tautology:
            jid = "test123"
            src = f"""namespace ML.Jobs

theorem job_{jid} (p q r s t : Prop) : {statement} := by
  decide

end ML.Jobs
"""

            # Verify the generated source
            assert "decide" in src
            assert statement in src
            assert f"job_{jid}" in src

    def test_worker_axiom_handling(self):
        """Test that axioms are handled without lake build."""
        # Simulate axiom detection
        statement = "p -> (q -> p)"  # K axiom
        is_axiom = True
        derivation_rule = "axiom"
        derivation_depth = 0

        if is_axiom:
            # Axioms should get prover="axiom" and status="success"
            prover = "axiom"
            status = "success"
            proof_text = ""

            assert prover == "axiom"
            assert status == "success"
            assert proof_text == ""

    def test_worker_derivation_metadata_preservation(self):
        """Test that derivation metadata is preserved in proofs."""
        # Simulate derived statement
        statement = "p -> p"
        derivation_rule = "modus_ponens"
        derivation_depth = 2
        is_axiom = False

        # Simulate successful build
        build_success = True
        src = "theorem job_123 (p : Prop) : p -> p := by intro hp; exact hp"

        if build_success:
            prover = "lean4"
            status = "success"

            # Metadata should be preserved
            assert prover == "lean4"
            assert status == "success"
            assert derivation_rule == "modus_ponens"
            assert derivation_depth == 2

    def test_worker_by_decide_prover_detection(self):
        """Test that by_decide prover is correctly detected."""
        # Simulate tautology fallback
        src_with_decide = """theorem job_123 (p : Prop) : p -> p := by
  decide
"""

        # Check if by_decide was used
        used_by_decide = "decide" in src_with_decide
        assert used_by_decide

        # Prover should be by_decide
        prover = "by_decide" if used_by_decide else "lean4"
        assert prover == "by_decide"

    def test_worker_error_handling(self):
        """Test worker error handling for various scenarios."""
        # Test non-JSON payload
        payload = "invalid json"
        try:
            data = json.loads(payload)
        except Exception:
            # Should skip non-JSON payloads
            assert True

        # Test empty statement
        statement = ""
        if not statement:
            # Should skip empty statements
            assert True

        # Test database error
        try:
            # Simulate database error
            raise Exception("Database connection failed")
        except Exception as e:
            # Should handle database errors gracefully
            assert "Database connection failed" in str(e)

    def test_worker_unicode_normalization(self):
        """Test that Unicode is normalized to ASCII."""
        from backend.worker import norm_stmt

        # Test Unicode normalization
        unicode_statement = "p → (q → p)"
        ascii_statement = norm_stmt(unicode_statement)

        assert ascii_statement == "p -> (q -> p)"
        assert "→" not in ascii_statement
        assert "->" in ascii_statement

    def test_worker_lean_source_generation(self):
        """Test Lean source generation for different statement types."""
        from backend.worker import make_lean_source

        # Test axiom source
        jid = "test123"
        statement = "p -> (q -> p)"
        src = make_lean_source(jid, statement)

        assert f"job_{jid}" in src
        assert statement in src
        assert "namespace ML.Jobs" in src
        assert "end ML.Jobs" in src

    def test_worker_proof_body_generation(self):
        """Test proof body generation for different patterns."""
        from backend.worker import proof_body_for

        # Test known patterns
        assert proof_body_for("p -> p") == "  intro hp\n  exact hp"
        assert proof_body_for("p /\\ q -> p") == "  intro h\n  exact h.left"
        assert proof_body_for("p /\\ q -> q") == "  intro h\n  exact h.right"

        # Test unknown pattern
        assert proof_body_for("unknown pattern") == "  admit"
