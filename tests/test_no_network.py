"""
Tests for NO_NETWORK discipline and network isolation framework.

Validates mocks, stubs, and replay capabilities for offline CI testing.
"""

import os
import json
import pytest
import tempfile
from pathlib import Path

os.environ['NO_NETWORK'] = 'true'

from backend.testing.no_network import (
    is_no_network_mode,
    enforce_no_network,
    MockCursor,
    MockConnection,
    mock_psycopg_connect,
    MockRedis,
    mock_redis_from_url,
    HTTPRecorder,
    mock_requests_session,
    network_sandbox,
    get_mock_db_connection,
    get_mock_redis,
)


class TestEnvironmentDetection:
    """Test NO_NETWORK environment detection."""
    
    def test_is_no_network_mode_true(self):
        """Test NO_NETWORK mode detection when enabled."""
        os.environ['NO_NETWORK'] = 'true'
        assert is_no_network_mode() is True
        
    def test_is_no_network_mode_false(self):
        """Test NO_NETWORK mode detection when disabled."""
        old_value = os.environ.get('NO_NETWORK')
        os.environ['NO_NETWORK'] = 'false'
        assert is_no_network_mode() is False
        if old_value:
            os.environ['NO_NETWORK'] = old_value
            
    def test_enforce_no_network_raises(self):
        """Test enforce_no_network raises when disabled."""
        old_value = os.environ.get('NO_NETWORK')
        os.environ['NO_NETWORK'] = 'false'
        with pytest.raises(RuntimeError, match="NO_NETWORK mode not enabled"):
            enforce_no_network()
        if old_value:
            os.environ['NO_NETWORK'] = old_value


class TestMockCursor:
    """Test pattern-based mock cursor."""
    
    def test_cursor_pattern_matching(self):
        """Test cursor matches SQL patterns."""
        rules = [
            (lambda s, p: 'COUNT(*)' in s, lambda: ('one', (42,))),
            (lambda s, p: 'SELECT id' in s, lambda: ('all', [(1,), (2,), (3,)])),
        ]
        
        cursor = MockCursor(rules)
        
        cursor.execute("SELECT COUNT(*) FROM proofs")
        assert cursor.fetchone() == (42,)
        
        cursor.execute("SELECT id FROM statements")
        assert cursor.fetchall() == [(1,), (2,), (3,)]
        
    def test_cursor_default_empty_result(self):
        """Test cursor returns empty result for unmatched patterns."""
        cursor = MockCursor([])
        cursor.execute("SELECT * FROM unknown_table")
        assert cursor.fetchall() == []
        
    def test_cursor_context_manager(self):
        """Test cursor works as context manager."""
        cursor = MockCursor([])
        with cursor as c:
            assert c is cursor


class TestMockConnection:
    """Test mock PostgreSQL connection."""
    
    def test_connection_provides_cursor(self):
        """Test connection provides cursor."""
        rules = [(lambda s, p: True, lambda: ('one', (1,)))]
        conn = MockConnection(rules)
        
        cursor = conn.cursor()
        assert isinstance(cursor, MockCursor)
        
    def test_connection_commit_rollback(self):
        """Test connection commit/rollback are no-ops."""
        conn = MockConnection([])
        conn.commit()  # Should not raise
        conn.rollback()  # Should not raise
        
    def test_connection_close(self):
        """Test connection can be closed."""
        conn = MockConnection([])
        assert conn.closed is False
        conn.close()
        assert conn.closed is True
        
    def test_connection_context_manager(self):
        """Test connection works as context manager."""
        conn = MockConnection([])
        with conn as c:
            assert c is conn
        assert conn.closed is True


class TestMockPsycopg:
    """Test psycopg mock factory."""
    
    def test_mock_psycopg_connect_default_rules(self):
        """Test mock connect with default rules."""
        connect = mock_psycopg_connect()
        conn = connect()
        
        with conn.cursor() as cur:
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='proofs'")
            result = cur.fetchall()
            assert len(result) > 0
            
    def test_mock_psycopg_connect_custom_rules(self):
        """Test mock connect with custom rules."""
        rules = [
            (lambda s, p: 'custom_query' in s, lambda: ('one', ('custom_result',))),
        ]
        connect = mock_psycopg_connect(rules)
        conn = connect()
        
        with conn.cursor() as cur:
            cur.execute("SELECT custom_query()")
            assert cur.fetchone() == ('custom_result',)


class TestMockRedis:
    """Test in-memory Redis mock."""
    
    def test_redis_lpush_rpop(self):
        """Test Redis list push/pop operations."""
        redis = MockRedis()
        
        redis.lpush('queue', 'job1', 'job2')
        assert redis.llen('queue') == 2
        
        assert redis.rpop('queue') == 'job2'
        assert redis.rpop('queue') == 'job1'
        assert redis.rpop('queue') is None
        
    def test_redis_rpush_lpop(self):
        """Test Redis reverse push/pop."""
        redis = MockRedis()
        
        redis.rpush('queue', 'job1', 'job2')
        assert redis.lpop('queue') == 'job1'
        assert redis.lpop('queue') == 'job2'
        
    def test_redis_lrange(self):
        """Test Redis list range."""
        redis = MockRedis()
        redis.rpush('queue', 'a', 'b', 'c', 'd')
        
        assert redis.lrange('queue', 0, 1) == ['a', 'b']
        assert redis.lrange('queue', 1, 2) == ['b', 'c']
        
    def test_redis_delete(self):
        """Test Redis key deletion."""
        redis = MockRedis()
        redis.lpush('key1', 'val1')
        redis.lpush('key2', 'val2')
        
        assert redis.delete('key1') == 1
        assert redis.llen('key1') == 0
        
    def test_redis_flushdb(self):
        """Test Redis flush database."""
        redis = MockRedis()
        redis.lpush('key1', 'val1')
        redis.lpush('key2', 'val2')
        
        redis.flushdb()
        assert redis.llen('key1') == 0
        assert redis.llen('key2') == 0
        
    def test_redis_ping(self):
        """Test Redis health check."""
        redis = MockRedis()
        assert redis.ping() is True


class TestHTTPRecorder:
    """Test HTTP request/response recorder."""
    
    def test_recorder_record_replay(self):
        """Test recording and replaying HTTP requests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HTTPRecorder(tmpdir)
            
            recorder.record(
                'GET', 'http://example.com/api/test',
                None, 200, '{"status": "ok"}',
                {'Content-Type': 'application/json'}
            )
            
            response = recorder.replay('GET', 'http://example.com/api/test')
            assert response is not None
            assert response['status'] == 200
            assert response['body'] == '{"status": "ok"}'
            
    def test_recorder_replay_not_found(self):
        """Test replaying non-existent request returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HTTPRecorder(tmpdir)
            response = recorder.replay('GET', 'http://example.com/not-recorded')
            assert response is None
            
    def test_recorder_request_hash_uniqueness(self):
        """Test different requests get different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HTTPRecorder(tmpdir)
            
            recorder.record('GET', 'http://example.com/api/1', None, 200, 'response1')
            recorder.record('GET', 'http://example.com/api/2', None, 200, 'response2')
            
            resp1 = recorder.replay('GET', 'http://example.com/api/1')
            resp2 = recorder.replay('GET', 'http://example.com/api/2')
            
            assert resp1['body'] == 'response1'
            assert resp2['body'] == 'response2'


class TestMockHTTPSession:
    """Test mock HTTP session."""
    
    def test_session_replay_get(self):
        """Test session GET request replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HTTPRecorder(tmpdir)
            recorder.record('GET', 'http://api.example.com/data', None, 200, '{"data": "value"}')
            
            session = mock_requests_session(recorder)
            response = session.get('http://api.example.com/data')
            
            assert response.status_code == 200
            assert response.text == '{"data": "value"}'
            assert response.json() == {"data": "value"}
            
    def test_session_replay_post(self):
        """Test session POST request replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HTTPRecorder(tmpdir)
            body = '{"input": "test"}'
            recorder.record('POST', 'http://api.example.com/submit', body, 201, '{"id": 123}')
            
            session = mock_requests_session(recorder)
            response = session.post('http://api.example.com/submit', data=body)
            
            assert response.status_code == 201
            
    def test_session_missing_recording_raises(self):
        """Test session raises when recording not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HTTPRecorder(tmpdir)
            session = mock_requests_session(recorder)
            
            with pytest.raises(RuntimeError, match="No recording found"):
                session.get('http://api.example.com/not-recorded')


class TestNetworkSandbox:
    """Test network sandbox context manager."""
    
    def test_sandbox_requires_no_network_mode(self):
        """Test sandbox requires NO_NETWORK mode in strict mode."""
        old_value = os.environ.get('NO_NETWORK')
        os.environ['NO_NETWORK'] = 'false'
        
        with pytest.raises(RuntimeError, match="NO_NETWORK mode must be enabled"):
            with network_sandbox(strict=True):
                pass
                
        if old_value:
            os.environ['NO_NETWORK'] = old_value
            
    def test_sandbox_non_strict_mode(self):
        """Test sandbox works in non-strict mode."""
        old_value = os.environ.get('NO_NETWORK')
        os.environ['NO_NETWORK'] = 'true'
        
        with network_sandbox(strict=False) as sandbox:
            assert sandbox is not None
            
        if old_value:
            os.environ['NO_NETWORK'] = old_value


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_mock_db_connection(self):
        """Test get_mock_db_connection convenience function."""
        conn = get_mock_db_connection()
        assert isinstance(conn, MockConnection)
        
    def test_get_mock_redis(self):
        """Test get_mock_redis convenience function."""
        redis = get_mock_redis()
        assert isinstance(redis, MockRedis)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_orchestrator_metrics_mock(self):
        """Test mocking orchestrator metrics endpoint."""
        rules = [
            (lambda s, p: 'information_schema.columns' in s and 'proofs' in s,
             lambda: ('all', [('success', 'boolean'), ('created_at', 'timestamp')])),
            (lambda s, p: 'COUNT(*) FROM proofs WHERE success' in s,
             lambda: ('one', (42,))),
            (lambda s, p: 'COUNT(*) FROM proofs' in s and 'WHERE' not in s,
             lambda: ('one', (50,))),
            (lambda s, p: 'COUNT(*) FROM blocks' in s,
             lambda: ('one', (10,))),
            (lambda s, p: 'COUNT(*) FROM statements' in s,
             lambda: ('one', (100,))),
            (lambda s, p: 'MAX(block_number)' in s,
             lambda: ('one', (9,))),
            (lambda s, p: 'MAX(derivation_depth)' in s,
             lambda: ('one', (5,))),
        ]
        
        conn = MockConnection(rules)
        
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM proofs WHERE success = TRUE")
            proofs_success = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM proofs")
            proofs_total = cur.fetchone()[0]
            
            cur.execute("SELECT COALESCE(MAX(block_number), 0) FROM blocks")
            block_height = cur.fetchone()[0]
            
        assert proofs_success == 42
        assert proofs_total == 50
        assert block_height == 9
        
    def test_worker_redis_queue_mock(self):
        """Test mocking worker Redis queue operations."""
        redis = MockRedis()
        
        redis.lpush('ml:jobs', '{"statement": "p -> p", "id": 1}')
        redis.lpush('ml:jobs', '{"statement": "q -> q", "id": 2}')
        
        assert redis.llen('ml:jobs') == 2
        
        job1 = redis.rpop('ml:jobs')
        assert job1 is not None
        assert 'p -> p' in job1
        
        job2 = redis.rpop('ml:jobs')
        assert job2 is not None
        assert 'q -> q' in job2
        
        assert redis.llen('ml:jobs') == 0
