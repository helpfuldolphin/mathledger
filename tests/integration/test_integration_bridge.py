"""
Tests for integration bridge functionality.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from backend.integration.bridge import IntegrationBridge
from backend.integration.metrics import LatencyTracker


@pytest.fixture
def mock_db_url():
    """Mock database URL."""
    return "postgresql://test:test@localhost:5432/test"


@pytest.fixture
def mock_redis_url():
    """Mock Redis URL."""
    return "redis://localhost:6379/0"


@pytest.fixture
def bridge(mock_db_url, mock_redis_url):
    """Create integration bridge with mocked connections."""
    return IntegrationBridge(
        db_url=mock_db_url,
        redis_url=mock_redis_url,
        metrics_enabled=True
    )


def test_bridge_initialization(bridge, mock_db_url, mock_redis_url):
    """Test bridge initializes correctly."""
    assert bridge.db_url == mock_db_url
    assert bridge.redis_url == mock_redis_url
    assert bridge.metrics_enabled is True
    assert isinstance(bridge.tracker, LatencyTracker)


def test_track_operation_context_manager(bridge):
    """Test operation tracking context manager."""
    with bridge.track_operation("test_op", {"key": "value"}):
        pass

    assert len(bridge.tracker.measurements) == 1
    measurement = bridge.tracker.measurements[0]
    assert measurement.operation == "test_op"
    assert measurement.success is True
    assert measurement.metadata["key"] == "value"


def test_track_operation_with_error(bridge):
    """Test operation tracking with error."""
    with pytest.raises(ValueError):
        with bridge.track_operation("test_op_error"):
            raise ValueError("Test error")

    assert len(bridge.tracker.measurements) == 1
    measurement = bridge.tracker.measurements[0]
    assert measurement.operation == "test_op_error"
    assert measurement.success is False
    assert "Test error" in measurement.error


@patch('backend.integration.bridge.psycopg')
def test_get_db_connection(mock_psycopg, bridge):
    """Test database connection retrieval."""
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn

    conn = bridge.get_db_connection()

    assert conn == mock_conn
    mock_psycopg.connect.assert_called_once()
    assert len(bridge.tracker.measurements) == 1
    assert bridge.tracker.measurements[0].operation == "db_connect"


@patch('backend.integration.bridge.redis')
def test_get_redis_client(mock_redis, bridge):
    """Test Redis client retrieval."""
    mock_client = MagicMock()
    mock_redis.from_url.return_value = mock_client

    client = bridge.get_redis_client()

    assert client == mock_client
    mock_redis.from_url.assert_called_once()
    assert len(bridge.tracker.measurements) == 1
    assert bridge.tracker.measurements[0].operation == "redis_connect"


@patch('backend.integration.bridge.psycopg')
def test_query_statements(mock_psycopg, bridge):
    """Test statement querying."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_cursor.fetchone.return_value = (1,)
    mock_cursor.fetchall.return_value = [
        (1, "p -> p", "p->p", "abc123", None)
    ]

    mock_psycopg.connect.return_value = mock_conn
    bridge._db_conn = mock_conn

    results = bridge.query_statements(system="pl", limit=10)

    assert len(results) == 1
    assert results[0]["text"] == "p -> p"
    assert results[0]["hash"] == "abc123"


@patch('backend.integration.bridge.psycopg')
def test_get_metrics_summary(mock_psycopg, bridge):
    """Test metrics summary retrieval."""
    mock_conn = MagicMock()
    mock_conn.closed = False
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_cursor.fetchone.side_effect = [(100,), (50,), (40,), (5,)]
    mock_cursor.fetchall.return_value = [
        ("success", "boolean")
    ]

    mock_psycopg.connect.return_value = mock_conn
    bridge._db_conn = mock_conn

    metrics = bridge.get_metrics_summary()

    assert metrics["statements"] == 100
    assert metrics["proofs"] == 50
    assert metrics["blocks"] == 5


@patch('backend.integration.bridge.redis')
def test_enqueue_verification_job(mock_redis, bridge):
    """Test job enqueueing."""
    mock_client = MagicMock()
    mock_redis.from_url.return_value = mock_client

    result = bridge.enqueue_verification_job("p -> p", "Propositional")

    assert result is True
    mock_client.rpush.assert_called_once()


def test_get_latency_stats(bridge):
    """Test latency statistics retrieval."""
    with bridge.track_operation("op1"):
        pass
    with bridge.track_operation("op2"):
        pass

    stats = bridge.get_latency_stats()

    assert "op1" in stats
    assert "op2" in stats
    assert stats["op1"]["count"] == 1
    assert stats["op2"]["count"] == 1


def test_metrics_disabled(mock_db_url, mock_redis_url):
    """Test bridge with metrics disabled."""
    bridge = IntegrationBridge(
        db_url=mock_db_url,
        redis_url=mock_redis_url,
        metrics_enabled=False
    )

    assert bridge.tracker is None

    with bridge.track_operation("test_op"):
        pass

    stats = bridge.get_latency_stats()
    assert stats == {}


def test_close_connections(bridge):
    """Test connection cleanup."""
    mock_conn = MagicMock()
    mock_conn.closed = False
    mock_client = MagicMock()

    bridge._db_conn = mock_conn
    bridge._redis_client = mock_client

    bridge.close()

    mock_conn.close.assert_called_once()
    mock_client.close.assert_called_once()
