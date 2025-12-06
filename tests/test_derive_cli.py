"""
Tests for the derivation CLI engine.
"""

import pytest
from unittest.mock import MagicMock as Mock, patch, MagicMock
import redis
from backend.axiom_engine.derive import DerivationEngine


class TestDerivationEngine:
    """Test the DerivationEngine class."""

    @pytest.fixture
    def mock_db_conn(self):
        """Mock database connection."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        return mock_conn, mock_cursor

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock = Mock()
        mock.llen.return_value = 0
        mock.rpush.return_value = 1
        return mock

    def test_load_axioms(self, mock_db_conn, mock_redis):
        """Test loading axioms from database."""
        mock_conn, mock_cursor = mock_db_conn
        mock_cursor.fetchall.return_value = [
            ("p -> (q -> p)", 0),
            ("(p -> (q -> r)) -> ((p -> q) -> (p -> r))", 0)
        ]

        with patch('psycopg.connect', return_value=mock_conn), \
             patch('redis.from_url', return_value=mock_redis):
            engine = DerivationEngine("test_db", "redis://test_redis")
            axioms = engine.load_axioms()

        assert len(axioms) == 2
        assert axioms[0].content == "p -> (q -> p)"
        assert axioms[0].is_axiom is True
        assert axioms[0].derivation_depth == 0

    def test_load_derived_statements(self, mock_db_conn, mock_redis):
        """Test loading derived statements from database."""
        mock_conn, mock_cursor = mock_db_conn
        mock_cursor.fetchall.return_value = [
            ("q", "modus_ponens", 1),
            ("r", "modus_ponens", 2)
        ]

        with patch('psycopg.connect', return_value=mock_conn), \
             patch('redis.from_url', return_value=mock_redis):
            engine = DerivationEngine("test_db", "redis://test_redis")
            statements = engine.load_derived_statements()

        assert len(statements) == 2
        assert statements[0].content == "q"
        assert statements[0].is_axiom is False
        assert statements[0].derivation_rule == "modus_ponens"
        assert statements[0].derivation_depth == 1

    def test_upsert_statement_new(self, mock_db_conn, mock_redis):
        """Test upserting a new statement."""
        mock_conn, mock_cursor = mock_db_conn
        # Mock system_id resolution
        mock_cursor.fetchone.side_effect = [("theory123",), None, ("stmt123",)]  # system_id, no existing, new ID

        from backend.axiom_engine.rules import Statement
        stmt = Statement("p -> q", is_axiom=False, derivation_rule="modus_ponens")

        with patch('psycopg.connect', return_value=mock_conn), \
             patch('redis.from_url', return_value=mock_redis):
            engine = DerivationEngine("test_db", "redis://test_redis")
            result = engine.upsert_statement(mock_conn, stmt, "theory123")

        assert result == "stmt123"
        mock_cursor.execute.assert_called()

    def test_upsert_statement_existing(self, mock_db_conn, mock_redis):
        """Test upserting an existing statement."""
        mock_conn, mock_cursor = mock_db_conn
        # Mock system_id resolution and existing statement
        mock_cursor.fetchone.side_effect = [("theory123",), ("existing123",)]  # system_id, existing statement

        from backend.axiom_engine.rules import Statement
        stmt = Statement("p -> q", is_axiom=False, derivation_rule="modus_ponens")

        with patch('psycopg.connect', return_value=mock_conn), \
             patch('redis.from_url', return_value=mock_redis):
            engine = DerivationEngine("test_db", "redis://test_redis")
            result = engine.upsert_statement(mock_conn, stmt, "theory123")

        assert result == "existing123"
        # Should check for existing statement but not insert new one
        assert mock_cursor.execute.call_count == 2  # system_id resolution + check existing

    def test_enqueue_job(self, mock_db_conn, mock_redis):
        """Test enqueuing a job."""
        mock_conn, mock_cursor = mock_db_conn
        # Mock system_id resolution
        mock_cursor.fetchone.return_value = ("theory123",)

        from backend.axiom_engine.rules import Statement
        stmt = Statement("p -> q", is_axiom=False)

        with patch('psycopg.connect', return_value=mock_conn), \
             patch('redis.from_url', return_value=mock_redis):
            engine = DerivationEngine("test_db", "redis://test_redis")
            engine.redis_client = mock_redis
            engine.enqueue_job(stmt)

        mock_redis.rpush.assert_called_once()
        call_args = mock_redis.rpush.call_args
        assert call_args[0][0] == "ml:jobs"
        job_data = call_args[0][1]
        assert "p -> q" in job_data
        assert "Propositional" in job_data

    @patch('backend.axiom_engine.derive.InferenceEngine')
    def test_derive_statements_basic(self, mock_inference_engine, mock_db_conn, mock_redis):
        """Test basic derivation process."""
        mock_conn, mock_cursor = mock_db_conn

        # Mock database responses
        mock_cursor.fetchall.side_effect = [
            [("p -> (q -> p)", 0)],  # load_axioms
            [],  # load_derived_statements
        ]
        # Mock fetchone to return different values based on the query
        mock_cursor.fetchone.return_value = (0,)

        # Mock inference engine
        mock_engine = Mock()
        mock_inference_engine.return_value = mock_engine
        mock_engine.derive_new_statements.return_value = []

        with patch('psycopg.connect', return_value=mock_conn), \
             patch('redis.from_url', return_value=mock_redis):
            engine = DerivationEngine("test_db", "redis://test_redis")
            engine.redis_client = mock_redis
            summary = engine.derive_statements(steps=1)

        assert summary['steps'] == 1
        assert summary['n_new'] == 0
        assert summary['max_depth'] == 0
        assert summary['n_jobs'] == 0
        assert summary['pct_success'] == 0.0

    @patch('backend.axiom_engine.derive.InferenceEngine')
    def test_derive_statements_with_new_statements(self, mock_inference_engine, mock_db_conn, mock_redis):
        """Test derivation with new statements."""
        mock_conn, mock_cursor = mock_db_conn

        # Mock database responses
        mock_cursor.fetchall.side_effect = [
            [("p -> (q -> p)", 0)],  # load_axioms
            [],  # load_derived_statements
        ]
        # Mock fetchone to return different values based on the query
        call_count = 0
        def mock_fetchone():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("theory123",)  # get theory ID
            elif call_count == 2:
                return None  # statement doesn't exist
            elif call_count == 3:
                return ("stmt123",)  # new statement ID
            else:
                return (10,)  # all other calls (proofs counts)

        mock_cursor.fetchone.side_effect = mock_fetchone

        # Mock inference engine to return new statements
        from backend.axiom_engine.rules import Statement
        new_stmt = Statement("q", is_axiom=False, derivation_rule="modus_ponens",
                           parent_statements=["p", "p -> q"])

        mock_engine = Mock()
        mock_inference_engine.return_value = mock_engine
        mock_engine.derive_new_statements.return_value = [new_stmt]

        with patch('psycopg.connect', return_value=mock_conn), \
             patch('redis.from_url', return_value=mock_redis):
            engine = DerivationEngine("test_db", "redis://test_redis")
            engine.redis_client = mock_redis

            # Mock the existing statements to include the parent statements
            # This is needed for the depth calculation
            def mock_load_axioms():
                return [Statement("p -> (q -> p)", is_axiom=True, derivation_depth=0)]

            def mock_load_derived_statements():
                return [Statement("p", is_axiom=False, derivation_depth=1),
                       Statement("p -> q", is_axiom=False, derivation_depth=1)]

            with patch.object(engine, 'load_axioms', side_effect=mock_load_axioms), \
                 patch.object(engine, 'load_derived_statements', side_effect=mock_load_derived_statements):
                summary = engine.derive_statements(steps=1)

        assert summary['steps'] == 1
        assert summary['n_new'] == 1
        assert summary['max_depth'] == 2  # parent depth 1 + 1 = 2
        assert summary['pct_success'] == 100.0  # 10/10 * 100


class TestDeriveCLI:
    """Test the CLI interface."""

    @patch('backend.axiom_engine.derive.DerivationEngine')
    def test_main_function(self, mock_engine_class):
        """Test the main CLI function."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.derive_statements.return_value = {
            'steps': 3,
            'n_new': 5,
            'max_depth': 2,
            'n_jobs': 10,
            'pct_success': 85.0
        }

        with patch('backend.axiom_engine.derive.append_to_progress') as mock_append:
            from backend.axiom_engine.derive import main
            result = main()

        assert result == 0
        mock_engine.derive_statements.assert_called_once()
        mock_append.assert_called_once()

    @patch('backend.axiom_engine.derive.DerivationEngine')
    def test_main_function_error(self, mock_engine_class):
        """Test the main CLI function with error."""
        mock_engine_class.side_effect = Exception("Database connection failed")

        from backend.axiom_engine.derive import main
        result = main()

        assert result == 1
