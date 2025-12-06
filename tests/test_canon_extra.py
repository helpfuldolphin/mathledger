"""
Extra tests for backend/logic/canon.py canonicalization functions.
"""

import pytest
from normalization.canon import normalize, normalize_pretty, are_equivalent


class TestNormalizeExtra:
    """Additional tests for the normalize function."""

    def test_conjunction_commutativity_set_equal(self):
        """Test that p/\\q and q/\\p normalize to the same form (set-equal after normalize)."""
        result1 = normalize("p /\\ q")
        result2 = normalize("q /\\ p")
        assert result1 == result2, f"Expected {result1} == {result2}"
        # Verify they are equivalent using the are_equivalent function
        assert are_equivalent("p /\\ q", "q /\\ p")

    def test_idempotence_conjunction(self):
        """Test that p/p collapses to p (idempotence)."""
        assert normalize("p /\\ p") == "p"
        assert normalize("p /\\ p /\\ p") == "p"
        assert normalize("(p /\\ p)") == "p"

    def test_disjunction_commutativity_set_equal(self):
        """Test that p\\/q and q\\/p normalize to the same form (set-equal after normalize)."""
        result1 = normalize("p \\/ q")
        result2 = normalize("q \\/ p")
        assert result1 == result2, f"Expected {result1} == {result2}"
        # Verify they are equivalent using the are_equivalent function
        assert are_equivalent("p \\/ q", "q \\/ p")

    def test_idempotence_disjunction(self):
        """Test that p\\/p collapses to p (idempotence)."""
        assert normalize("p \\/ p") == "p"
        assert normalize("p \\/ p \\/ p") == "p"
        assert normalize("(p \\/ p)") == "p"

    def test_complex_commutativity_preservation(self):
        """Test that (p/\\q)/(q/\\p) preserves structure under OR."""
        result = normalize("(p /\\ q) \\/ (q /\\ p)")
        # Should preserve the structure: (p/\\q)\\/(q/\\p)
        assert result == "(p/\\q)\\/(q/\\p)"


class TestNormalizePrettyExtra:
    """Additional tests for the normalize_pretty function."""

    def test_implication_chain_form(self):
        """Test exact form 'p -> q -> r' becomes 'p -> (q -> r)'."""
        result = normalize_pretty("p -> q -> r")
        assert result == "p -> (q -> r)"

    def test_implication_grouped_form(self):
        """Test exact form '(p -> q) -> r' stays '(p -> q) -> r'."""
        result = normalize_pretty("(p -> q) -> r")
        assert result == "(p -> q) -> r"

    def test_implication_chain_with_spaces(self):
        """Test implication chain with extra spaces."""
        result = normalize_pretty("p  ->  q  ->  r")
        assert result == "p -> (q -> r)"

    def test_implication_grouped_with_spaces(self):
        """Test grouped implication with extra spaces."""
        result = normalize_pretty("( p -> q ) -> r")
        assert result == "(p -> q) -> r"
