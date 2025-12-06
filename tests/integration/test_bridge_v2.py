"""
Tests for Integration Bridge V2 with connection pooling and retry logic.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from backend.integration.bridge_v2 import (
    IntegrationBridgeV2,
    RetryConfig,
    BridgeToken,
    with_retry
)


def test_retry_config_defaults():
    """Test RetryConfig default values."""
    config = RetryConfig()
    assert config.max_attempts == 3
    assert config.initial_delay == 0.1
    assert config.max_delay == 2.0
    assert config.exponential_base == 2.0


def test_retry_config_custom():
    """Test RetryConfig with custom values."""
    config = RetryConfig(
        max_attempts=5,
        initial_delay=0.05,
        max_delay=1.0,
        exponential_base=1.5
    )
    assert config.max_attempts == 5
    assert config.initial_delay == 0.05
    assert config.max_delay == 1.0
    assert config.exponential_base == 1.5


def test_with_retry_decorator_success():
    """Test retry decorator with successful operation."""
    call_count = [0]
    
    @with_retry(RetryConfig(max_attempts=3))
    def test_func():
        call_count[0] += 1
        return "success"
    
    result = test_func()
    assert result == "success"
    assert call_count[0] == 1


def test_with_retry_decorator_eventual_success():
    """Test retry decorator with eventual success."""
    call_count = [0]
    
    @with_retry(RetryConfig(max_attempts=3, initial_delay=0.01))
    def test_func():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError("Not yet")
        return "success"
    
    result = test_func()
    assert result == "success"
    assert call_count[0] == 3


def test_with_retry_decorator_failure():
    """Test retry decorator with persistent failure."""
    call_count = [0]
    
    @with_retry(RetryConfig(max_attempts=3, initial_delay=0.01))
    def test_func():
        call_count[0] += 1
        raise ValueError("Always fails")
    
    with pytest.raises(ValueError, match="Always fails"):
        test_func()
    
    assert call_count[0] == 3


def test_bridge_token_creation():
    """Test BridgeToken creation."""
    token = BridgeToken("test_op", 1234567890.0, {"key": "value"})
    
    assert token.operation == "test_op"
    assert token.timestamp == 1234567890.0
    assert token.metadata == {"key": "value"}
    assert len(token.token_id) == 64  # SHA256 hex


def test_bridge_token_deterministic():
    """Test BridgeToken generates deterministic IDs."""
    token1 = BridgeToken("test_op", 1234567890.0, {"key": "value"})
    token2 = BridgeToken("test_op", 1234567890.0, {"key": "value"})
    
    assert token1.token_id == token2.token_id


def test_bridge_token_different_for_different_data():
    """Test BridgeToken generates different IDs for different data."""
    token1 = BridgeToken("test_op", 1234567890.0, {"key": "value1"})
    token2 = BridgeToken("test_op", 1234567890.0, {"key": "value2"})
    
    assert token1.token_id != token2.token_id


def test_bridge_token_verify_success():
    """Test BridgeToken verification success."""
    token = BridgeToken("test_op", 1234567890.0, {"key": "value"})
    
    assert token.verify("test_op") is True
    assert token.verify() is True


def test_bridge_token_verify_failure():
    """Test BridgeToken verification failure."""
    token = BridgeToken("test_op", 1234567890.0, {"key": "value"})
    
    assert token.verify("wrong_op") is False


def test_bridge_token_to_dict():
    """Test BridgeToken dictionary conversion."""
    token = BridgeToken("test_op", 1234567890.0, {"key": "value"})
    data = token.to_dict()
    
    assert data["operation"] == "test_op"
    assert data["timestamp"] == 1234567890.0
    assert data["metadata"] == {"key": "value"}
    assert "token_id" in data


@pytest.fixture
def mock_db_url():
    """Mock database URL."""
    return "postgresql://test:test@localhost:5432/test"


@pytest.fixture
def mock_redis_url():
    """Mock Redis URL."""
    return "redis://localhost:6379/0"


@patch('backend.integration.bridge_v2.ConnectionPool')
@patch('backend.integration.bridge_v2.RedisConnectionPool')
def test_bridge_v2_initialization(mock_redis_pool, mock_db_pool, mock_db_url, mock_redis_url):
    """Test BridgeV2 initialization."""
    bridge = IntegrationBridgeV2(
        db_url=mock_db_url,
        redis_url=mock_redis_url,
        metrics_enabled=True,
        pool_size=10
    )
    
    assert bridge.db_url == mock_db_url
    assert bridge.redis_url == mock_redis_url
    assert bridge.metrics_enabled is True
    assert bridge._pool_size == 10
    assert isinstance(bridge.retry_config, RetryConfig)


@patch('backend.integration.bridge_v2.ConnectionPool')
@patch('backend.integration.bridge_v2.RedisConnectionPool')
def test_bridge_v2_track_operation(mock_redis_pool, mock_db_pool, mock_db_url, mock_redis_url):
    """Test operation tracking with token generation."""
    bridge = IntegrationBridgeV2(
        db_url=mock_db_url,
        redis_url=mock_redis_url,
        metrics_enabled=True
    )
    
    with bridge.track_operation("test_op", {"key": "value"}) as token:
        assert isinstance(token, BridgeToken)
        assert token.operation == "test_op"
        assert token.metadata == {"key": "value"}
    
    assert token.token_id in bridge._tokens
    assert len(bridge.tracker.measurements) == 1


@patch('backend.integration.bridge_v2.ConnectionPool')
@patch('backend.integration.bridge_v2.RedisConnectionPool')
def test_bridge_v2_verify_token(mock_redis_pool, mock_db_pool, mock_db_url, mock_redis_url):
    """Test token verification."""
    bridge = IntegrationBridgeV2(
        db_url=mock_db_url,
        redis_url=mock_redis_url
    )
    
    with bridge.track_operation("test_op") as token:
        token_id = token.token_id
    
    assert bridge.verify_token(token_id, "test_op") is True
    assert bridge.verify_token(token_id, "wrong_op") is False
    assert bridge.verify_token("invalid_token") is False


@patch('backend.integration.bridge_v2.ConnectionPool')
@patch('backend.integration.bridge_v2.RedisConnectionPool')
def test_bridge_v2_get_bridge_integrity_hash(mock_redis_pool, mock_db_pool, mock_db_url, mock_redis_url):
    """Test bridge integrity hash generation."""
    bridge = IntegrationBridgeV2(
        db_url=mock_db_url,
        redis_url=mock_redis_url
    )
    
    with bridge.track_operation("op1"):
        pass
    with bridge.track_operation("op2"):
        pass
    
    integrity_hash = bridge.get_bridge_integrity_hash()
    
    assert len(integrity_hash) == 64  # SHA256 hex
    assert isinstance(integrity_hash, str)


@patch('backend.integration.bridge_v2.ConnectionPool')
@patch('backend.integration.bridge_v2.RedisConnectionPool')
def test_bridge_v2_get_latency_stats_with_integrity(mock_redis_pool, mock_db_pool, mock_db_url, mock_redis_url):
    """Test latency stats include integrity information."""
    bridge = IntegrationBridgeV2(
        db_url=mock_db_url,
        redis_url=mock_redis_url,
        metrics_enabled=True
    )
    
    with bridge.track_operation("test_op"):
        pass
    
    stats = bridge.get_latency_stats()
    
    assert "_bridge_integrity" in stats
    assert "_token_count" in stats
    assert len(stats["_bridge_integrity"]) == 64
    assert stats["_token_count"] == 1


@patch('backend.integration.bridge_v2.ConnectionPool')
@patch('backend.integration.bridge_v2.RedisConnectionPool')
@patch('backend.integration.bridge_v2.redis')
def test_bridge_v2_enqueue_with_token(mock_redis, mock_redis_pool, mock_db_pool, mock_db_url, mock_redis_url):
    """Test job enqueueing includes token."""
    mock_client = MagicMock()
    mock_redis.Redis.return_value = mock_client
    
    bridge = IntegrationBridgeV2(
        db_url=mock_db_url,
        redis_url=mock_redis_url
    )
    bridge._redis_pool = MagicMock()
    
    result = bridge.enqueue_verification_job("p -> p", "Propositional")
    
    assert result["success"] is True
    assert "token_id" in result
    assert len(result["token_id"]) == 64


@patch('backend.integration.bridge_v2.ConnectionPool')
@patch('backend.integration.bridge_v2.RedisConnectionPool')
def test_bridge_v2_close(mock_redis_pool, mock_db_pool, mock_db_url, mock_redis_url):
    """Test bridge cleanup."""
    mock_db_pool_instance = MagicMock()
    mock_redis_pool_instance = MagicMock()
    
    mock_db_pool.return_value = mock_db_pool_instance
    mock_redis_pool.from_url.return_value = mock_redis_pool_instance
    
    bridge = IntegrationBridgeV2(
        db_url=mock_db_url,
        redis_url=mock_redis_url
    )
    
    bridge._db_pool = mock_db_pool_instance
    bridge._redis_pool = mock_redis_pool_instance
    
    bridge.close()
    
    mock_db_pool_instance.close.assert_called_once()
    mock_redis_pool_instance.disconnect.assert_called_once()


def test_bridge_token_metadata_ordering():
    """Test BridgeToken handles metadata ordering correctly."""
    token1 = BridgeToken("test", 123.0, {"b": 2, "a": 1})
    token2 = BridgeToken("test", 123.0, {"a": 1, "b": 2})
    
    assert token1.token_id == token2.token_id
