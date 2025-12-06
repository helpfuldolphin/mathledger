"""
Integration Bridge for MathLedger cross-language systems.

Provides unified interfaces for Python backend components with
latency tracking and error handling.
"""

import os
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

try:
    import psycopg
except ImportError:
    psycopg = None

try:
    import redis
except ImportError:
    redis = None

from backend.integration.metrics import LatencyTracker


class IntegrationBridge:
    """
    Unified bridge for cross-language integration.
    
    Wraps backend components (axiom_engine, orchestrator, worker)
    with latency tracking and standardized interfaces.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        metrics_enabled: bool = True
    ):
        from backend.security.runtime_env import get_database_url, get_redis_url

        self.db_url = db_url or get_database_url()
        self.redis_url = redis_url or get_redis_url()
        self.metrics_enabled = metrics_enabled
        self.tracker = LatencyTracker() if metrics_enabled else None

        self._db_conn = None
        self._redis_client = None

    @contextmanager
    def track_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Track operation latency if metrics enabled."""
        if self.metrics_enabled and self.tracker:
            with self.tracker.track(operation, metadata):
                yield
        else:
            yield

    def get_db_connection(self):
        """Get database connection with connection pooling."""
        if not psycopg:
            raise RuntimeError("psycopg not available")

        with self.track_operation("db_connect"):
            if not self._db_conn or self._db_conn.closed:
                self._db_conn = psycopg.connect(self.db_url, connect_timeout=5)
            return self._db_conn

    def get_redis_client(self):
        """Get Redis client with connection pooling."""
        if not redis:
            raise RuntimeError("redis not available")

        with self.track_operation("redis_connect"):
            if not self._redis_client:
                self._redis_client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0
                )
            return self._redis_client

    def execute_derivation(
        self,
        system: str = "pl",
        steps: int = 10,
        max_breadth: int = 100,
        max_total: int = 1000
    ) -> Dict[str, Any]:
        """
        Execute derivation with latency tracking.
        
        Args:
            system: Logical system (pl, fol, etc.)
            steps: Number of derivation steps
            max_breadth: Maximum breadth per step
            max_total: Maximum total statements
            
        Returns:
            Derivation results with statistics
        """
        with self.track_operation("derivation", {"system": system, "steps": steps}):
            from backend.axiom_engine.derive import DerivationEngine

            engine = DerivationEngine(
                db_url=self.db_url,
                redis_url=self.redis_url,
                max_depth=3,
                max_breadth=max_breadth,
                max_total=max_total
            )

            return engine.derive_statements(steps=steps)

    def query_statements(
        self,
        system: str = "pl",
        limit: int = 100,
        hash_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query statements with latency tracking.
        
        Args:
            system: Logical system filter
            limit: Maximum number of results
            hash_filter: Optional hash to filter by
            
        Returns:
            List of statement dictionaries
        """
        with self.track_operation("query_statements", {"system": system, "limit": limit}):
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
                        "created_at": row[4].isoformat() if row[4] else None
                    })

                return results

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get system metrics with latency tracking.
        
        Returns:
            Dictionary with system metrics
        """
        with self.track_operation("get_metrics"):
            conn = self.get_db_connection()
            with conn.cursor() as cur:
                metrics = {
                    "statements": 0,
                    "proofs": 0,
                    "blocks": 0,
                    "success_rate": 0.0
                }

                try:
                    cur.execute("SELECT COUNT(*) FROM statements")
                    metrics["statements"] = int(cur.fetchone()[0])
                except Exception:
                    pass

                try:
                    cur.execute("SELECT COUNT(*) FROM proofs")
                    total_proofs = int(cur.fetchone()[0])
                    metrics["proofs"] = total_proofs

                    cur.execute("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name='proofs'
                    """)
                    cols = {r[0].lower(): r[1].lower() for r in cur.fetchall()}

                    if "success" in cols and "boolean" in cols["success"]:
                        cur.execute("SELECT COUNT(*) FROM proofs WHERE success = TRUE")
                        success_count = int(cur.fetchone()[0])
                        if total_proofs > 0:
                            metrics["success_rate"] = (success_count / total_proofs) * 100.0
                except Exception:
                    pass

                try:
                    cur.execute("SELECT COUNT(*) FROM blocks")
                    metrics["blocks"] = int(cur.fetchone()[0])
                except Exception:
                    pass

                return metrics

    def enqueue_verification_job(
        self,
        statement: str,
        theory: str = "Propositional"
    ) -> bool:
        """
        Enqueue verification job to Redis with latency tracking.
        
        Args:
            statement: Statement to verify
            theory: Theory name
            
        Returns:
            True if successfully enqueued
        """
        with self.track_operation("enqueue_job", {"theory": theory}):
            try:
                import json
                client = self.get_redis_client()
                job_data = json.dumps({
                    "statement": statement,
                    "theory": theory
                })
                client.rpush("ml:jobs", job_data)
                return True
            except Exception:
                return False

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics for all tracked operations."""
        if not self.tracker:
            return {}

        stats = {}
        operations = set(m.operation for m in self.tracker.measurements)

        for op in operations:
            stats[op] = self.tracker.get_stats(op)

        return stats

    def close(self):
        """Close all connections."""
        if self._db_conn and not self._db_conn.closed:
            self._db_conn.close()
        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception:
                pass
