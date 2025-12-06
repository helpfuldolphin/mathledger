"""
from backend.repro.determinism import deterministic_timestamp

_GLOBAL_SEED = 0

NO_NETWORK discipline: Mocks, stubs, and replay frameworks for offline CI testing.

This module provides comprehensive network isolation capabilities for MathLedger's
CI environment, enabling all tests to run without external dependencies.

Architecture:
- Database mocks: In-memory SQLite or pattern-based cursor mocks
- Redis mocks: In-memory queue simulation
- HTTP replay: Recorded request/response pairs for deterministic testing
- Network sandbox: Environment-aware isolation enforcement

Usage:
    import os
    os.environ['NO_NETWORK'] = 'true'
    
    from backend.testing.no_network import mock_psycopg, mock_redis
    
"""

import os
import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Callable
from contextlib import contextmanager
from datetime import datetime
import hashlib


# ============================================================================
# ============================================================================

def is_no_network_mode() -> bool:
    """Check if NO_NETWORK mode is enabled."""
    return os.getenv('NO_NETWORK', '').lower() in ('true', '1', 'yes')


def enforce_no_network():
    """Raise exception if NO_NETWORK mode is not enabled."""
    if not is_no_network_mode():
        raise RuntimeError(
            "NO_NETWORK mode not enabled. Set NO_NETWORK=true environment variable."
        )


# ============================================================================
# ============================================================================

class MockCursor:
    """
    Pattern-based mock cursor for PostgreSQL operations.
    
    Matches SQL patterns and returns pre-configured responses.
    Useful for testing schema-adaptive code without real database.
    """
    
    def __init__(self, rules: List[Tuple[Callable, Callable]]):
        """
        Initialize with pattern matching rules.
        
        Args:
            rules: List of (predicate, responder) tuples
                   predicate(sql, params) -> bool
                   responder() -> ('one'|'all', data)
        """
        self.rules = list(rules)
        self.pending = None
        self.description = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc, tb):
        return False
        
    def execute(self, sql: str, params: Optional[Tuple] = None):
        """Execute SQL by matching against patterns."""
        normalized = " ".join(sql.split())
        for predicate, responder in self.rules:
            if predicate(normalized, params):
                self.pending = responder()
                return
        self.pending = ('all', [])
        
    def fetchone(self) -> Optional[Tuple]:
        """Fetch single row."""
        if not self.pending:
            return None
        kind, value = self.pending
        self.pending = None
        if kind == 'one':
            return value
        elif kind == 'all' and value:
            return value[0]
        return None
        
    def fetchall(self) -> List[Tuple]:
        """Fetch all rows."""
        if not self.pending:
            return []
        kind, value = self.pending
        self.pending = None
        if kind == 'all':
            return value
        elif kind == 'one' and value:
            return [value]
        return []


class MockConnection:
    """Mock PostgreSQL connection."""
    
    def __init__(self, rules: List[Tuple[Callable, Callable]]):
        """Initialize with cursor rules."""
        self.cursor_obj = MockCursor(rules)
        self.closed = False
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
        
    def cursor(self):
        """Return mock cursor."""
        return self.cursor_obj
        
    def commit(self):
        """No-op commit."""
        pass
        
    def rollback(self):
        """No-op rollback."""
        pass
        
    def close(self):
        """Mark connection as closed."""
        self.closed = True


def mock_psycopg_connect(rules: Optional[List[Tuple[Callable, Callable]]] = None):
    """
    Create mock psycopg.connect function.
    
    Args:
        rules: Optional pattern matching rules for cursor
        
    Returns:
        Function that returns MockConnection
    """
    if rules is None:
        rules = [
            (lambda s, p: 'information_schema.columns' in s and 'proofs' in s,
             lambda: ('all', [('success', 'boolean'), ('created_at', 'timestamp')])),
            (lambda s, p: 'information_schema.columns' in s and 'statements' in s,
             lambda: ('all', [('text',), ('normalized_text',), ('hash',)])),
            (lambda s, p: 'COUNT(*)' in s,
             lambda: ('one', (0,))),
            (lambda s, p: 'MAX(' in s,
             lambda: ('one', (0,))),
        ]
    
    def connect(*args, **kwargs):
        return MockConnection(rules)
    
    return connect


# ============================================================================
# ============================================================================

class MockRedis:
    """
    In-memory Redis mock for queue operations.
    
    Simulates Redis list operations (LPUSH, RPOP, LLEN) without network.
    """
    
    def __init__(self):
        """Initialize empty queues."""
        self.queues: Dict[str, List[str]] = {}
        
    def lpush(self, key: str, *values: str) -> int:
        """Push values to left of list."""
        if key not in self.queues:
            self.queues[key] = []
        for value in reversed(values):
            self.queues[key].insert(0, value)
        return len(self.queues[key])
        
    def rpush(self, key: str, *values: str) -> int:
        """Push values to right of list."""
        if key not in self.queues:
            self.queues[key] = []
        self.queues[key].extend(values)
        return len(self.queues[key])
        
    def rpop(self, key: str) -> Optional[str]:
        """Pop value from right of list."""
        if key not in self.queues or not self.queues[key]:
            return None
        return self.queues[key].pop()
        
    def lpop(self, key: str) -> Optional[str]:
        """Pop value from left of list."""
        if key not in self.queues or not self.queues[key]:
            return None
        return self.queues[key].pop(0)
        
    def llen(self, key: str) -> int:
        """Get length of list."""
        return len(self.queues.get(key, []))
        
    def lrange(self, key: str, start: int, stop: int) -> List[str]:
        """Get range of values from list."""
        if key not in self.queues:
            return []
        return self.queues[key][start:stop+1]
        
    def delete(self, *keys: str) -> int:
        """Delete keys."""
        count = 0
        for key in keys:
            if key in self.queues:
                del self.queues[key]
                count += 1
        return count
        
    def flushdb(self):
        """Clear all keys."""
        self.queues.clear()
        
    def ping(self) -> bool:
        """Health check."""
        return True


def mock_redis_from_url(url: str, **kwargs):
    """
    Create mock Redis client from URL.
    
    Args:
        url: Redis URL (ignored in mock)
        **kwargs: Additional arguments (ignored)
        
    Returns:
        MockRedis instance
    """
    return MockRedis()


# ============================================================================
# ============================================================================

class HTTPRecorder:
    """
    Record and replay HTTP requests for deterministic testing.
    
    Records request/response pairs to JSON files for later replay.
    """
    
    def __init__(self, recording_dir: str = 'artifacts/no_network/recordings'):
        """
        Initialize recorder.
        
        Args:
            recording_dir: Directory to store recordings
        """
        self.recording_dir = recording_dir
        os.makedirs(recording_dir, exist_ok=True)
        
    def _request_hash(self, method: str, url: str, body: Optional[str] = None) -> str:
        """Generate hash for request."""
        key = f"{method}:{url}:{body or ''}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
        
    def _recording_path(self, request_hash: str) -> str:
        """Get path to recording file."""
        return os.path.join(self.recording_dir, f"{request_hash}.json")
        
    def record(self, method: str, url: str, body: Optional[str], 
               status: int, response: str, headers: Optional[Dict] = None):
        """
        Record HTTP request/response pair.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            body: Request body
            status: Response status code
            response: Response body
            headers: Response headers
        """
        request_hash = self._request_hash(method, url, body)
        recording = {
            'request': {
                'method': method,
                'url': url,
                'body': body,
            },
            'response': {
                'status': status,
                'body': response,
                'headers': headers or {},
            },
            'recorded_at': deterministic_timestamp(_GLOBAL_SEED).isoformat(),
        }
        
        path = self._recording_path(request_hash)
        with open(path, 'w') as f:
            json.dump(recording, f, indent=2)
            
    def replay(self, method: str, url: str, body: Optional[str] = None) -> Optional[Dict]:
        """
        Replay recorded HTTP request.
        
        Args:
            method: HTTP method
            url: Request URL
            body: Request body
            
        Returns:
            Recorded response or None if not found
        """
        request_hash = self._request_hash(method, url, body)
        path = self._recording_path(request_hash)
        
        if not os.path.exists(path):
            return None
            
        with open(path, 'r') as f:
            recording = json.load(f)
            
        return recording['response']


class MockHTTPResponse:
    """Mock HTTP response object."""
    
    def __init__(self, status: int, body: str, headers: Optional[Dict] = None):
        self.status_code = status
        self.text = body
        self.content = body.encode()
        self.headers = headers or {}
        
    def json(self):
        """Parse response as JSON."""
        return json.loads(self.text)
        
    def raise_for_status(self):
        """Raise exception for error status codes."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def mock_requests_session(recorder: HTTPRecorder):
    """
    Create mock requests.Session that uses HTTP recorder.
    
    Args:
        recorder: HTTPRecorder instance
        
    Returns:
        Mock session object
    """
    class MockSession:
        def __init__(self):
            self.recorder = recorder
            
        def request(self, method: str, url: str, **kwargs):
            """Make HTTP request (replay from recording)."""
            body = kwargs.get('data') or kwargs.get('json')
            if body and not isinstance(body, str):
                body = json.dumps(body)
                
            response = self.recorder.replay(method, url, body)
            if response is None:
                raise RuntimeError(
                    f"No recording found for {method} {url}. "
                    "Run with NO_NETWORK=false to record."
                )
                
            return MockHTTPResponse(
                response['status'],
                response['body'],
                response.get('headers')
            )
            
        def get(self, url: str, **kwargs):
            return self.request('GET', url, **kwargs)
            
        def post(self, url: str, **kwargs):
            return self.request('POST', url, **kwargs)
            
        def put(self, url: str, **kwargs):
            return self.request('PUT', url, **kwargs)
            
        def delete(self, url: str, **kwargs):
            return self.request('DELETE', url, **kwargs)
            
    return MockSession()


# ============================================================================
# ============================================================================

class NetworkSandbox:
    """
    Network isolation enforcement for CI environments.
    
    Ensures no network calls escape during testing.
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize sandbox.
        
        Args:
            strict: If True, raise exceptions on network attempts
        """
        self.strict = strict
        self.violations: List[str] = []
        
    def __enter__(self):
        """Enter sandbox context."""
        if not is_no_network_mode():
            if self.strict:
                raise RuntimeError("NO_NETWORK mode must be enabled for sandbox")
        return self
        
    def __exit__(self, exc_type, exc, tb):
        """Exit sandbox context."""
        if self.violations and self.strict:
            raise RuntimeError(
                f"Network violations detected: {', '.join(self.violations)}"
            )
        return False
        
    def record_violation(self, operation: str):
        """Record network violation."""
        self.violations.append(operation)
        if self.strict:
            raise RuntimeError(f"Network violation: {operation}")


@contextmanager
def network_sandbox(strict: bool = True):
    """
    Context manager for network isolation.
    
    Args:
        strict: If True, raise exceptions on network attempts
        
    Yields:
        NetworkSandbox instance
    """
    sandbox = NetworkSandbox(strict=strict)
    with sandbox:
        yield sandbox


# ============================================================================
# ============================================================================

def get_mock_db_connection(rules: Optional[List[Tuple[Callable, Callable]]] = None):
    """
    Get mock database connection for testing.
    
    Args:
        rules: Optional pattern matching rules
        
    Returns:
        MockConnection instance
    """
    enforce_no_network()
    connect = mock_psycopg_connect(rules)
    return connect()


def get_mock_redis():
    """
    Get mock Redis client for testing.
    
    Returns:
        MockRedis instance
    """
    enforce_no_network()
    return MockRedis()


def get_http_recorder(recording_dir: str = 'artifacts/no_network/recordings'):
    """
    Get HTTP recorder for replay testing.
    
    Args:
        recording_dir: Directory for recordings
        
    Returns:
        HTTPRecorder instance
    """
    return HTTPRecorder(recording_dir)
