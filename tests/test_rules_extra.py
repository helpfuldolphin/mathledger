"""
Extra tests for backend/axiom_engine/rules.py inference functions.
"""

import pytest
from backend.axiom_engine.rules import Statement, ModusPonens


class TestInferenceEngineExtra:
    """Additional tests for inference engine functionality."""

    def test_modus_ponens_simple_case(self):
        """Test simple MP case: p, p->q should derive q."""
        # Create premises
        premise1 = Statement("p", is_axiom=True)
        premise2 = Statement("p -> q", is_axiom=True)

        # Apply modus ponens
        result = ModusPonens.apply([premise1, premise2])

        # Should derive exactly one statement: q
        assert len(result) == 1
        derived = result[0]
        assert derived.text == "q"
        assert derived.is_axiom == False
        assert derived.derivation_rule == "MP"
        assert derived.parent_statements == ["p", "p -> q"]

    def test_modus_ponens_reverse_order(self):
        """Test MP case with reversed premise order: p->q, p should derive q."""
        # Create premises in reverse order
        premise1 = Statement("p -> q", is_axiom=True)
        premise2 = Statement("p", is_axiom=True)

        # Apply modus ponens
        result = ModusPonens.apply([premise1, premise2])

        # Should derive exactly one statement: q
        assert len(result) == 1
        derived = result[0]
        assert derived.text == "q"
        assert derived.is_axiom == False
        assert derived.derivation_rule == "MP"
        assert derived.parent_statements == ["p", "p -> q"]

    def test_modus_ponens_can_apply_positive(self):
        """Test that ModusPonens.can_apply returns True for valid premises."""
        premise1 = Statement("p", is_axiom=True)
        premise2 = Statement("p -> q", is_axiom=True)

        assert ModusPonens.can_apply([premise1, premise2]) == True
        assert ModusPonens.can_apply([premise2, premise1]) == True

    def test_modus_ponens_can_apply_negative(self):
        """Test that ModusPonens.can_apply returns False for invalid premises."""
        premise1 = Statement("p", is_axiom=True)
        premise2 = Statement("q", is_axiom=True)
        premise3 = Statement("p -> q", is_axiom=True)

        # Two non-implications
        assert ModusPonens.can_apply([premise1, premise2]) == False

        # Wrong number of premises
        assert ModusPonens.can_apply([premise1]) == False
        assert ModusPonens.can_apply([premise1, premise2, premise3]) == False

        # Mismatched antecedent
        premise4 = Statement("r -> q", is_axiom=True)
        assert ModusPonens.can_apply([premise1, premise4]) == False

    def test_modus_ponens_no_duplicates(self):
        """Test that MP doesn't create duplicate derivations."""
        premise1 = Statement("p", is_axiom=True)
        premise2 = Statement("p -> q", is_axiom=True)

        # Apply modus ponens multiple times
        result1 = ModusPonens.apply([premise1, premise2])
        result2 = ModusPonens.apply([premise2, premise1])

        # Both should return the same single result
        assert len(result1) == 1
        assert len(result2) == 1
        assert result1[0].content == result2[0].content
        assert result1[0].text == result2[0].text
