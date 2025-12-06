"""
from backend.repro.determinism import deterministic_unix_timestamp

_GLOBAL_SEED = 0

Integration Bridge V2 - Proof Fabric Architect

Optimized for sub-150ms latency with:
- Connection pooling (PostgreSQL and Redis)
- Retry logic with exponential backoff
- Token propagation and SHA256 verification
- Deterministic proof flow for Reasoning Merkle
"""

import os
import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from functools import wraps

try:
    import psycopg
    from psycopg_pool import ConnectionPool
except ImportError:
    psycopg = None
    ConnectionPool = None

try:
    import redis
    from redis.connection import ConnectionPool as RedisConnectionPool
except ImportError:
    redis = None
    RedisConnectionPool = None

from backend.integration.metrics import LatencyTracker


class RetryConfig:
    """Configuration for retry logic."""
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 2.0,
        exponential_base: float = 2.0
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base


def with_retry(config: RetryConfig = None):
    """Decorator for retry logic with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = config.initial_delay
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < config.max_attempts - 1:
                        time.sleep(min(delay, config.max_delay))
                        delay *= config.exponential_base
                    else:
                        raise last_exception
            
            raise last_exception
        
        return wrapper
    return decorator


class BridgeToken:
    """Authentication token for bridge operations."""
    
    def __init__(self, operation: str, timestamp: float, metadata: Dict[str, Any] = None):
        self.operation = operation
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.token_id = self._generate_token_id()
    
    def _generate_token_id(self) -> str:
        """Generate deterministic token ID."""
        data = f"{self.operation}:{self.timestamp}:{json.dumps(self.metadata, sort_keys=True)}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def verify(self, expected_operation: str = None) -> bool:
        """Verify token integrity."""
        if expected_operation and self.operation != expected_operation:
            return False
        
        expected_id = self._generate_token_id()
        return self.token_id == expected_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class IntegrationBridgeV2:
    """
    V2 Integration Bridge with connection pooling and retry logic.
    
    Optimized for sub-150ms latency with authenticated proof flow.
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        metrics_enabled: bool = True,
        pool_size: int = 10,
        retry_config: Optional[RetryConfig] = None
    ):
        from backend.security.runtime_env import get_database_url, get_redis_url

        self.db_url = db_url or get_database_url()
        self.redis_url = redis_url or get_redis_url()
        self.metrics_enabled = metrics_enabled
        self.tracker = LatencyTracker() if metrics_enabled else None
        self.retry_config = retry_config or RetryConfig()
        
        self._db_pool = None
        self._redis_pool = None
        self._pool_size = pool_size
        
        self._init_db_pool()
        self._init_redis_pool()
        
        self._tokens: Dict[str, BridgeToken] = {}
    
    def _init_db_pool(self):
        """Initialize PostgreSQL connection pool."""
        if not psycopg or not ConnectionPool:
            return
        
        try:
            self._db_pool = ConnectionPool(
                self.db_url,
                min_size=2,
                max_size=self._pool_size,
                timeout=5.0,
                max_idle=300.0,
                max_lifetime=3600.0
            )
        except Exception as e:
            print(f"[BRIDGE_V2] DB pool init failed: {e}")
    
    def _init_redis_pool(self):
        """Initialize Redis connection pool."""
        if not redis or not RedisConnectionPool:
            return
        
        try:
            self._redis_pool = RedisConnectionPool.from_url(
                self.redis_url,
                max_connections=self._pool_size,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
                decode_responses=True
            )
        except Exception as e:
            print(f"[BRIDGE_V2] Redis pool init failed: {e}")
    
    @contextmanager
    def track_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Track operation latency with token generation."""
        token = BridgeToken(operation, deterministic_unix_timestamp(_GLOBAL_SEED), metadata)
        self._tokens[token.token_id] = token
        
        if self.metrics_enabled and self.tracker:
            with self.tracker.track(operation, metadata):
                yield token
        else:
            yield token
    
    @with_retry()
    def get_db_connection(self):
        """Get database connection from pool with retry."""
        if not self._db_pool:
            raise RuntimeError("DB pool not initialized")
        
        with self.track_operation("db_connect"):
            return self._db_pool.getconn()
    
    def return_db_connection(self, conn):
        """Return connection to pool."""
        if self._db_pool and conn:
            self._db_pool.putconn(conn)
    
    @with_retry()
    def get_redis_client(self):
        """Get Redis client from pool with retry."""
        if not self._redis_pool:
            raise RuntimeError("Redis pool not initialized")
        
        with self.track_operation("redis_connect"):
            return redis.Redis(connection_pool=self._redis_pool)
    
    @with_retry(RetryConfig(max_attempts=3))
    def query_statements(
        self,
        system: str = "pl",
        limit: int = 100,
        hash_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query statements with retry and connection pooling.
        
        Optimized for sub-150ms latency.
        """
        with self.track_operation("query_statements", {"system": system, "limit": limit}) as token:
            conn = None
            try:
                conn = self.get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM systems WHERE name = %s LIMIT 1", (system,))
                    system_row = cur.fetchone()
                    if not system_row:
                        return []
                    
                    system_id = system_row[0]
                    
                    if hash_filter:
                        cur.execute("""
                            SELECT id, text, normalized_text, hash, created_at
                            FROM statements
                            WHERE system_id = %s AND hash = %s
                            LIMIT %s
                        """, (system_id, hash_filter, limit))
                    else:
                        cur.execute("""
                            SELECT id, text, normalized_text, hash, created_at
                            FROM statements
                            WHERE system_id = %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (system_id, limit))
                    
                    results = []
                    for row in cur.fetchall():
                        results.append({
                            "id": row[0],
                            "text": row[1],
                            "normalized_text": row[2],
                            "hash": row[3],
                            "created_at": row[4].isoformat() if row[4] else None,
                            "token_id": token.token_id
                        })
                    
                    return results
            finally:
                if conn:
                    self.return_db_connection(conn)
    
    @with_retry(RetryConfig(max_attempts=3))
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get system metrics with retry and connection pooling.
        
        Optimized for sub-150ms latency.
        """
        with self.track_operation("get_metrics") as token:
            conn = None
            try:
                conn = self.get_db_connection()
                with conn.cursor() as cur:
                    metrics = {
                        "statements": 0,
                        "proofs": 0,
                        "blocks": 0,
                        "success_rate": 0.0,
                        "token_id": token.token_id
                    }
                    
                    cur.execute("SELECT COUNT(*) FROM statements")
                    metrics["statements"] = int(cur.fetchone()[0])
                    
                    cur.execute("SELECT COUNT(*) FROM proofs")
                    total_proofs = int(cur.fetchone()[0])
                    metrics["proofs"] = total_proofs
                    
                    if total_proofs > 0:
                        cur.execute("""
                            SELECT column_name, data_type
                            FROM information_schema.columns
                            WHERE table_name='proofs'
                        """)
                        cols = {r[0].lower(): r[1].lower() for r in cur.fetchall()}
                        
                        if "success" in cols and "boolean" in cols["success"]:
                            cur.execute("SELECT COUNT(*) FROM proofs WHERE success = TRUE")
                            success_count = int(cur.fetchone()[0])
                            metrics["success_rate"] = (success_count / total_proofs) * 100.0
                    
                    cur.execute("SELECT COUNT(*) FROM blocks")
                    metrics["blocks"] = int(cur.fetchone()[0])
                    
                    return metrics
            finally:
                if conn:
                    self.return_db_connection(conn)
    
    @with_retry(RetryConfig(max_attempts=5, initial_delay=0.05))
    def enqueue_verification_job(
        self,
        statement: str,
        theory: str = "Propositional"
    ) -> Dict[str, Any]:
        """
        Enqueue verification job with retry and token.
        
        Returns job info with token for tracking.
        """
        with self.track_operation("enqueue_job", {"theory": theory}) as token:
            try:
                client = self.get_redis_client()
                job_data = json.dumps({
                    "statement": statement,
                    "theory": theory,
                    "token_id": token.token_id,
                    "timestamp": token.timestamp
                })
                client.rpush("ml:jobs", job_data)
                
                return {
                    "success": True,
                    "token_id": token.token_id,
                    "statement": statement,
                    "theory": theory
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "token_id": token.token_id
                }
    
    def verify_token(self, token_id: str, operation: str = None) -> bool:
        """Verify token integrity."""
        if token_id not in self._tokens:
            return False
        
        token = self._tokens[token_id]
        return token.verify(operation)
    
    def get_bridge_integrity_hash(self) -> str:
        """
        Generate bridge integrity hash for CI verification.
        
        Returns SHA256 hash of all active tokens.
        """
        token_ids = sorted(self._tokens.keys())
        combined = "".join(token_ids)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics with integrity verification."""
        if not self.tracker:
            return {}
        
        stats = {}
        operations = set(m.operation for m in self.tracker.measurements)
        
        for op in operations:
            stats[op] = self.tracker.get_stats(op)
        
        stats["_bridge_integrity"] = self.get_bridge_integrity_hash()
        stats["_token_count"] = len(self._tokens)
        
        return stats
    
    def close(self):
        """Close all connection pools."""
        if self._db_pool:
            try:
                self._db_pool.close()
            except Exception:
                pass
        
        if self._redis_pool:
            try:
                self._redis_pool.disconnect()
            except Exception:
                pass
