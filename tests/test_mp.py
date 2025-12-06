"""
Tests for Modus Ponens inference rule.
"""

import pytest
from backend.axiom_engine.rules import ModusPonens, Statement


class TestModusPonens:
    """Test the Modus Ponens inference rule."""

    def test_can_apply_with_valid_premises(self):
        """Test that Modus Ponens can be applied with valid premises."""
        p = Statement("p", is_axiom=True)
        p_implies_q = Statement("p -> q", is_axiom=False)

        assert ModusPonens.can_apply([p, p_implies_q])
        assert ModusPonens.can_apply([p_implies_q, p])  # Order shouldn't matter

    def test_can_apply_with_invalid_premises(self):
        """Test that Modus Ponens cannot be applied with invalid premises."""
        p = Statement("p", is_axiom=True)
        q = Statement("q", is_axiom=True)
        r_implies_s = Statement("r -> s", is_axiom=False)

        # Two atomic statements
        assert not ModusPonens.can_apply([p, q])

        # Implication but wrong antecedent
        assert not ModusPonens.can_apply([p, r_implies_s])

        # Only one premise
        assert not ModusPonens.can_apply([p])

        # Three premises
        assert not ModusPonens.can_apply([p, q, r_implies_s])

    def test_derive_q_from_p_and_p_implies_q(self):
        """Test deriving q from p and p -> q."""
        p = Statement("p", is_axiom=True)
        p_implies_q = Statement("p -> q", is_axiom=False)

        derived = ModusPonens.apply([p, p_implies_q])

        assert len(derived) == 1
        assert derived[0].content == "q"
        assert not derived[0].is_axiom
        assert derived[0].derivation_rule == "MP"
        assert set(derived[0].parent_statements) == {"p", "p -> q"}

    def test_derive_r_from_q_and_q_implies_r(self):
        """Test deriving r from q and q -> r."""
        q = Statement("q", is_axiom=False)
        q_implies_r = Statement("q -> r", is_axiom=False)

        derived = ModusPonens.apply([q, q_implies_r])

        assert len(derived) == 1
        assert derived[0].content == "r"
        assert not derived[0].is_axiom
        assert derived[0].derivation_rule == "MP"
        assert set(derived[0].parent_statements) == {"q", "q -> r"}

    def test_no_duplicate_derivations(self):
        """Test that repeated runs don't produce duplicates."""
        p = Statement("p", is_axiom=True)
        p_implies_q = Statement("p -> q", is_axiom=False)

        # First application
        derived1 = ModusPonens.apply([p, p_implies_q])

        # Second application with same premises
        derived2 = ModusPonens.apply([p, p_implies_q])

        # Should get same result
        assert len(derived1) == len(derived2) == 1
        assert derived1[0].content == derived2[0].content == "q"

    def test_derive_chain_p_to_q_to_r(self):
        """Test deriving q then r in a chain."""
        p = Statement("p", is_axiom=True)
        p_implies_q = Statement("p -> q", is_axiom=False)
        q_implies_r = Statement("q -> r", is_axiom=False)

        # First: derive q from p and p -> q
        derived_q = ModusPonens.apply([p, p_implies_q])
        assert len(derived_q) == 1
        assert derived_q[0].content == "q"

        # Second: derive r from q and q -> r
        q = derived_q[0]
        derived_r = ModusPonens.apply([q, q_implies_r])
        assert len(derived_r) == 1
        assert derived_r[0].content == "r"

    def test_parse_implication(self):
        """Test parsing implications."""
        antecedent, consequent = ModusPonens._parse_implication("p -> q")
        assert antecedent == "p"
        assert consequent == "q"

        antecedent, consequent = ModusPonens._parse_implication("(p) -> (q)")
        assert antecedent == "p"
        assert consequent == "q"

    def test_is_implication(self):
        """Test identifying implications."""
        assert ModusPonens._is_implication("p -> q")
        assert ModusPonens._is_implication("(p) -> (q)")
        assert not ModusPonens._is_implication("p")
        assert not ModusPonens._is_implication("p /\\ q")
        assert not ModusPonens._is_implication("(p -> q)")  # Parenthesized implication
