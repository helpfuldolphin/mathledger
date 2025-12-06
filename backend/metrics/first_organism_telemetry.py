"""
First Organism Telemetry Emitter.

Provides helpers for emitting First Organism metrics from test runs
and production derivation pipelines. All metrics are written to Redis
for collection by the Metrics Oracle (Cursor K).

Redis Keys:
  - ml:metrics:first_organism:runs_total (counter)
  - ml:metrics:first_organism:last_ht (string: short H_t hash, 16 chars)
  - ml:metrics:first_organism:last_ht_full (string: full H_t hash, 64 chars)
  - ml:metrics:first_organism:latency_seconds (float)
  - ml:metrics:first_organism:duration_seconds (float)
  - ml:metrics:first_organism:last_abstentions (int)
  - ml:metrics:first_organism:last_run_timestamp (ISO 8601)
  - ml:metrics:first_organism:duration_history (list of floats)
  - ml:metrics:first_organism:abstention_history (list of ints)
  - ml:metrics:first_organism:success_history (list: "success" | "failure")
  - ml:metrics:first_organism:ht_history (list: short H_t hashes for verification)
  - ml:metrics:first_organism:last_status (string: "success" | "failure")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.repro.determinism import deterministic_isoformat, deterministic_seed_from_content

try:
    import redis
except ImportError:
    redis = None


@dataclass
class FirstOrganismRunResult:
    """Structured result from a First Organism test run."""

    duration_seconds: float
    ht_hash: str  # Composite root H_t (short or full)
    abstention_count: int
    success: bool
    timestamp: Optional[str] = None  # ISO 8601
    metadata: Dict[str, Any] = field(default_factory=dict)


class FirstOrganismTelemetry:
    """
    Emitter for First Organism metrics to Redis.

    Usage:
        telemetry = FirstOrganismTelemetry()
        telemetry.emit(result)
    """

    REDIS_KEY_PREFIX = "ml:metrics:first_organism"
    HISTORY_MAX_LEN = 20

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL", "redis://localhost:6379/0"
        )
        self._client: Optional[Any] = None
        self._connect()

    def _connect(self) -> None:
        if redis is None:
            return
        try:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
            self._client.ping()
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def _key(self, name: str) -> str:
        return f"{self.REDIS_KEY_PREFIX}:{name}"

    def emit(self, result: FirstOrganismRunResult) -> bool:
        """
        Emit a First Organism run result to Redis.

        Args:
            result: The structured run result

        Returns:
            True if metrics were emitted, False if Redis unavailable
        """
        if not self._client:
            return False

        try:
            pipe = self._client.pipeline()

            # Increment run counter
            pipe.incr(self._key("runs_total"))

            # Set scalar metrics - store both short and full H_t for verification
            short_ht = result.ht_hash[:16] if result.ht_hash else ""
            full_ht = result.ht_hash if result.ht_hash else ""
            pipe.set(self._key("last_ht"), short_ht)
            pipe.set(self._key("last_ht_full"), full_ht)
            pipe.set(self._key("duration_seconds"), f"{result.duration_seconds:.6f}")
            pipe.set(self._key("last_abstentions"), str(result.abstention_count))

            # Timestamp - use deterministic timestamp if not provided
            if result.timestamp:
                ts = result.timestamp
            else:
                # DETERMINISM: Derive timestamp from H_t content
                ts = deterministic_isoformat(result.ht_hash or "default", result.duration_seconds)
            pipe.set(self._key("last_run_timestamp"), ts)

            # Status
            status = "success" if result.success else "failure"
            pipe.set(self._key("last_status"), status)

            # History lists (LPUSH + LTRIM for rolling window)
            pipe.lpush(self._key("duration_history"), f"{result.duration_seconds:.6f}")
            pipe.ltrim(self._key("duration_history"), 0, self.HISTORY_MAX_LEN - 1)

            pipe.lpush(self._key("abstention_history"), str(result.abstention_count))
            pipe.ltrim(self._key("abstention_history"), 0, self.HISTORY_MAX_LEN - 1)

            pipe.lpush(self._key("success_history"), status)
            pipe.ltrim(self._key("success_history"), 0, self.HISTORY_MAX_LEN - 1)

            # H_t history for cryptographic verification
            if short_ht:
                pipe.lpush(self._key("ht_history"), short_ht)
                pipe.ltrim(self._key("ht_history"), 0, self.HISTORY_MAX_LEN - 1)

            pipe.execute()
            return True

        except Exception:
            return False

    def emit_from_context(
        self,
        duration_seconds: float,
        composite_root: str,
        abstention_count: int,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Convenience method to emit metrics from test context.

        Args:
            duration_seconds: Duration of the run in seconds
            composite_root: H_t composite attestation root
            abstention_count: Number of abstentions in run
            success: Whether the test passed
            metadata: Optional additional metadata
            timestamp: Optional explicit timestamp (ISO 8601). If not provided,
                       a deterministic timestamp is derived from composite_root.

        Returns:
            True if metrics were emitted
        """
        # DETERMINISM: Use provided timestamp or derive from content
        if timestamp is None:
            timestamp = deterministic_isoformat(composite_root, abstention_count, success)
        
        result = FirstOrganismRunResult(
            duration_seconds=duration_seconds,
            ht_hash=composite_root,
            abstention_count=abstention_count,
            success=success,
            timestamp=timestamp,
            metadata=metadata or {},
        )
        return self.emit(result)

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Retrieve current First Organism metrics from Redis.

        Returns:
            Dictionary of current metrics or empty dict if unavailable
        """
        if not self._client:
            return {}

        try:
            pipe = self._client.pipeline()
            pipe.get(self._key("runs_total"))
            pipe.get(self._key("last_ht"))
            pipe.get(self._key("duration_seconds"))
            pipe.get(self._key("last_abstentions"))
            pipe.get(self._key("last_run_timestamp"))
            pipe.get(self._key("last_status"))
            pipe.lrange(self._key("duration_history"), 0, self.HISTORY_MAX_LEN - 1)
            pipe.lrange(self._key("abstention_history"), 0, self.HISTORY_MAX_LEN - 1)
            pipe.lrange(self._key("success_history"), 0, self.HISTORY_MAX_LEN - 1)

            results = pipe.execute()

            def safe_int(val: Optional[str]) -> int:
                try:
                    return int(val) if val else 0
                except (ValueError, TypeError):
                    return 0

            def safe_float(val: Optional[str]) -> float:
                try:
                    return float(val) if val else 0.0
                except (ValueError, TypeError):
                    return 0.0

            duration_history = [safe_float(v) for v in (results[6] or [])]
            abstention_history = [safe_int(v) for v in (results[7] or [])]
            success_history: List[str] = results[8] or []

            return {
                "runs_total": safe_int(results[0]),
                "last_ht_hash": results[1] or "",
                "last_duration_seconds": safe_float(results[2]),
                "last_abstentions": safe_int(results[3]),
                "last_run_timestamp": results[4] or "",
                "last_status": results[5] or "",
                "duration_history": duration_history,
                "abstention_history": abstention_history,
                "success_history": success_history,
            }

        except Exception:
            return {}

    def clear(self) -> bool:
        """Clear all First Organism metrics from Redis (for testing)."""
        if not self._client:
            return False

        try:
            keys = [
                self._key("runs_total"),
                self._key("last_ht"),
                self._key("duration_seconds"),
                self._key("last_abstentions"),
                self._key("last_run_timestamp"),
                self._key("last_status"),
                self._key("latency_seconds"),
                self._key("duration_history"),
                self._key("abstention_history"),
                self._key("success_history"),
            ]
            self._client.delete(*keys)
            return True
        except Exception:
            return False


# Singleton instance for convenience
_default_telemetry: Optional[FirstOrganismTelemetry] = None


def get_telemetry() -> FirstOrganismTelemetry:
    """Get the default FirstOrganismTelemetry instance."""
    global _default_telemetry
    if _default_telemetry is None:
        _default_telemetry = FirstOrganismTelemetry()
    return _default_telemetry


def emit_first_organism_metrics(
    duration_seconds: float,
    ht_hash: str,
    abstention_count: int,
    success: bool,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> bool:
    """
    Convenience function to emit First Organism metrics.

    Args:
        duration_seconds: Run duration in seconds
        ht_hash: Composite root H_t
        abstention_count: Number of abstentions
        success: Whether the test passed
        metadata: Optional additional metadata
        timestamp: Optional explicit timestamp (ISO 8601). If not provided,
                   a deterministic timestamp is derived from ht_hash.

    Returns:
        True if metrics were emitted
    """
    # DETERMINISM: Use provided timestamp or derive from content
    if timestamp is None:
        timestamp = deterministic_isoformat(ht_hash, abstention_count, success)
    
    result = FirstOrganismRunResult(
        duration_seconds=duration_seconds,
        ht_hash=ht_hash,
        abstention_count=abstention_count,
        success=success,
        timestamp=timestamp,
        metadata=metadata or {},
    )
    return get_telemetry().emit(result)
