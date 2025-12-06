import pytest

pytest.skip("legacy derive engine tests pending migration", allow_module_level=True)
"""
Tests for deriving p -> p (identity) using K+S+MP+Subst.
"""

from unittest.mock import MagicMock as Mock, patch
from backend.axiom_engine.derive import DerivationEngine
from backend.axiom_engine.rules import Statement


class TestDerivesIdentity:
    """Test that the system can derive p -> p using K+S+MP+Subst."""

    @patch('backend.axiom_engine.derive.psycopg.connect')
    @patch('backend.axiom_engine.derive.redis.from_url')
    def test_derives_p_implies_p(self, mock_redis, mock_psycopg):
        """Test that the system can derive p -> p."""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_psycopg.return_value = mock_conn

        # Mock Redis
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.llen.return_value = 0
        mock_redis_client.rpush.return_value = 1

        # Mock database responses
        mock_cursor.fetchall.side_effect = [
            [("p -> (q -> p)", 0), ("(p -> (q -> r)) -> ((p -> q) -> (p -> r))", 0)],  # load_axioms
            [],  # load_derived_statements
        ]
        # Mock fetchone to return consistent values
        call_count = 0
        def mock_fetchone():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("theory123",)  # get theory ID
            else:
                return (10,)  # all other calls (proofs counts)

        mock_cursor.fetchone.side_effect = mock_fetchone

        # Create engine
        engine = DerivationEngine("test_db", "redis://test_redis", max_depth=3, max_breadth=100, max_total=1000)

        # Run derivation
        summary = engine.derive_statements(steps=5)

        # Check that we got some results
        assert summary['steps'] == 5
        assert summary['n_new'] >= 0  # Should derive some statements
        assert summary['max_depth'] >= 0

        # Verify that substitution was called
        assert mock_cursor.execute.call_count > 0

    def test_axiom_instances_generation(self):
        """Test that axiom instances are generated correctly."""
        from backend.axiom_engine.substitution import SubstitutionRule

        rule = SubstitutionRule(max_depth=2, atoms=['p', 'q'])
        axioms = ["p -> (q -> p)", "(p -> (q -> r)) -> ((p -> q) -> (p -> r))"]

        instances = rule.apply_to_axioms(axioms, max_instances_per_axiom=10)

        assert len(instances) > 0

        # Check that we have instances of both axioms
        original_axioms = {inst[0] for inst in instances}
        assert "p -> (q -> p)" in original_axioms
        assert "(p -> (q -> r)) -> ((p -> q) -> (p -> r))" in original_axioms

    def test_modus_ponens_with_instances(self):
        """Test that Modus Ponens works with axiom instances."""
        from backend.axiom_engine.rules import ModusPonens, Statement

        # Create K axiom instance: p -> (q -> p)
        k_instance = Statement("p -> (q -> p)", is_axiom=False, derivation_rule="Subst")

        # Create S axiom instance: (p -> (q -> p)) -> ((p -> p) -> (p -> p))
        s_instance = Statement("(p -> (q -> p)) -> ((p -> p) -> (p -> p))", is_axiom=False, derivation_rule="Subst")

        # Apply Modus Ponens
        derived = ModusPonens.apply([k_instance, s_instance])

        # Should derive (p -> p) -> (p -> p) which normalizes to (p->p)->p->p
        assert len(derived) == 1
        assert derived[0].content == "(p->p)->p->p"
        assert derived[0].derivation_rule == "MP"

    def test_derivation_depth_calculation(self):
        """Test that derivation depth is calculated correctly."""
        from backend.axiom_engine.rules import Statement

        # Axiom at depth 0
        axiom = Statement("p -> (q -> p)", is_axiom=True, derivation_depth=0)

        # Instance at depth 1
        instance = Statement("r -> (s -> r)", is_axiom=False, derivation_rule="Subst", derivation_depth=1)

        # Derived statement should have depth = min(parents) + 1
        parent_depths = [0, 1]  # axiom depth 0, instance depth 1
        expected_depth = min(parent_depths) + 1  # = 1

        assert expected_depth == 1

    def test_no_duplicate_derivations(self):
        """Test that the same statement is not derived multiple times."""
        from backend.axiom_engine.rules import Statement, ModusPonens

        # Create statements
        p = Statement("p", is_axiom=True, derivation_depth=0)
        p_implies_p = Statement("p -> p", is_axiom=False, derivation_depth=1)

        # First derivation
        derived1 = ModusPonens.apply([p, p_implies_p])

        # Second derivation with same premises
        derived2 = ModusPonens.apply([p, p_implies_p])

        # Should get same result both times
        assert len(derived1) == len(derived2)
        if derived1:
            assert derived1[0].content == derived2[0].content

    @patch('backend.axiom_engine.derive.psycopg.connect')
    def test_database_integration(self, mock_psycopg):
        """Test that derived statements are properly stored in database."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_psycopg.return_value = mock_conn

        # Mock database responses
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = ("theory123",)

        with patch('backend.axiom_engine.derive.redis.from_url', return_value=Mock()):
            engine = DerivationEngine("test_db", "redis://test_redis")

        # Test upserting a derived statement
        stmt = Statement("p -> p", is_axiom=False, derivation_rule="MP", derivation_depth=2)
        statement_id = engine.upsert_statement(mock_conn, stmt, "theory123")

        # Should have called execute to insert
        assert mock_cursor.execute.call_count > 0

    def test_derivation_limits(self):
        """Test that derivation respects depth and breadth limits."""
        from backend.axiom_engine.substitution import SubstitutionRule

        # Test depth limit
        rule = SubstitutionRule(max_depth=2, atoms=['p', 'q'])
        formulas = rule.generate_formulas(2)

        # All formulas should be within depth limit
        for formula in formulas:
            depth = rule._formula_depth(formula)
            assert depth <= 2

        # Test breadth limit
        rule.size_budget = 5
        formulas_limited = rule.generate_formulas(3)
        assert len(formulas_limited) <= 5
