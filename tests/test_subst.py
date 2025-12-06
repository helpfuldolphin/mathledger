"""
Tests for substitution rule.
"""

import pytest
from backend.axiom_engine.substitution import SubstitutionRule


class TestSubstitutionRule:
    """Test the SubstitutionRule class."""

    def test_generate_formulas_depth_0(self):
        """Test generating formulas at depth 0."""
        rule = SubstitutionRule(max_depth=0, atoms=['p', 'q'])
        formulas = rule.generate_formulas(0)

        assert set(formulas) == {'p', 'q'}

    def test_generate_formulas_depth_1(self):
        """Test generating formulas at depth 1."""
        rule = SubstitutionRule(max_depth=1, atoms=['p', 'q'])
        formulas = rule.generate_formulas(1)

        # Should include atoms and simple implications
        assert 'p' in formulas
        assert 'q' in formulas
        assert 'p->q' in formulas or '(p->q)' in formulas
        assert 'q->p' in formulas or '(q->p)' in formulas

    def test_substitute_axiom_k(self):
        """Test substituting the K axiom."""
        rule = SubstitutionRule(max_depth=2, atoms=['p', 'q', 'r'])
        axiom = "p -> (q -> p)"

        # Substitute p with r, q with p
        substitution = {"p": "r", "q": "p"}
        result = rule.substitute_axiom(axiom, substitution)

        # Should get "r -> (p -> r)"
        assert "r" in result
        assert "p" in result
        assert "->" in result

    def test_substitute_axiom_s(self):
        """Test substituting the S axiom."""
        rule = SubstitutionRule(max_depth=2, atoms=['p', 'q', 'r'])
        axiom = "(p -> (q -> r)) -> ((p -> q) -> (p -> r))"

        # Substitute with simple atoms
        substitution = {"p": "p", "q": "q", "r": "r"}
        result = rule.substitute_axiom(axiom, substitution)

        # Should contain the structure
        assert "p" in result
        assert "q" in result
        assert "r" in result
        assert "->" in result

    def test_generate_instances_k_axiom(self):
        """Test generating instances of K axiom."""
        rule = SubstitutionRule(max_depth=2, atoms=['p', 'q'])
        axiom = "p -> (q -> p)"

        instances = rule.generate_instances(axiom, max_instances=10)

        assert len(instances) > 0
        assert len(instances) <= 10

        # Check that all instances are valid
        for instance, substitution in instances:
            assert isinstance(instance, str)
            assert isinstance(substitution, dict)
            assert "->" in instance

    def test_generate_instances_s_axiom(self):
        """Test generating instances of S axiom."""
        rule = SubstitutionRule(max_depth=2, atoms=['p', 'q', 'r'])
        axiom = "(p -> (q -> r)) -> ((p -> q) -> (p -> r))"

        instances = rule.generate_instances(axiom, max_instances=5)

        assert len(instances) > 0
        assert len(instances) <= 5

        # Check that all instances are valid
        for instance, substitution in instances:
            assert isinstance(instance, str)
            assert isinstance(substitution, dict)
            assert "->" in instance

    def test_apply_to_axioms(self):
        """Test applying substitution to multiple axioms."""
        rule = SubstitutionRule(max_depth=1, atoms=['p', 'q'])
        axioms = ["p -> (q -> p)", "(p -> (q -> r)) -> ((p -> q) -> (p -> r))"]

        instances = rule.apply_to_axioms(axioms, max_instances_per_axiom=3)

        assert len(instances) > 0
        assert len(instances) <= 6  # 2 axioms * 3 instances each

        # Check structure
        for original, instance, substitution in instances:
            assert original in axioms
            assert isinstance(instance, str)
            assert isinstance(substitution, dict)

    def test_size_budget_enforcement(self):
        """Test that size budget prevents explosion."""
        rule = SubstitutionRule(max_depth=3, atoms=['p', 'q', 'r'])
        rule.size_budget = 10  # Small budget

        formulas = rule.generate_formulas(3)

        # Should be limited by budget
        assert len(formulas) <= 10

    def test_extract_metavariables(self):
        """Test extracting metavariables from formulas."""
        rule = SubstitutionRule()

        metavars = rule._extract_metavariables("p -> q")
        assert set(metavars) == {'p', 'q'}

        metavars = rule._extract_metavariables("(p -> (q -> r)) -> ((p -> q) -> (p -> r))")
        assert set(metavars) == {'p', 'q', 'r'}

        metavars = rule._extract_metavariables("p")
        assert metavars == ['p']

    def test_formula_depth_estimation(self):
        """Test formula depth estimation."""
        rule = SubstitutionRule()

        assert rule._formula_depth("p") == 0
        assert rule._formula_depth("p -> q") == 1
        assert rule._formula_depth("p -> (q -> r)") == 2
        assert rule._formula_depth("(p -> q) /\\ (r -> s)") == 2
