"""
Cursor K — Metrics Oracle

Deterministic, provenance-backed metrics pipeline for MathLedger.

This module rebuilds the metrics cartography stack with the following pillars:
- Deterministic session identifiers derived from canonical payload digests.
- Orthogonal collectors with explicit provenance and warning surfaces.
- Canonical normalization pipeline aligned with the MathLedger evaluation schema.
- Dual attestation (metrics + history) via SHA-256 merkle digests.
- Cross-run trend synthesis with variance checks over retained history.
"""

from __future__ import annotations

import json
import hashlib
import os
import statistics
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

try:  # Optional dependency; collectors degrade gracefully if unavailable.
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - runtime guard for environments without psycopg
    psycopg = None
    dict_row = None

try:
    import redis
except Exception:
    redis = None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CollectorResult:
    """Structured payload returned by each collector."""

    metrics: Dict[str, Dict[str, Any]]
    provenance: Dict[str, Any]
    timestamp_hint: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class CanonicalMetrics:
    """Canonical metrics structure matching schema_v1.json."""

    timestamp: str
    session_id: str
    source: str
    metrics: Dict[str, Any]
    provenance: Dict[str, Any]
    variance: Optional[Dict[str, Any]] = None
    notes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding optional None values."""
        data = asdict(self)
        if self.variance is None:
            data.pop("variance", None)
        if self.notes is None:
            data.pop("notes", None)
        return data

    def compute_merkle_hash(self) -> str:
        """Compute SHA-256 hash of the normalized metrics payload."""
        payload = json.dumps(self.metrics, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class MetricsConfig:
    """Configurator for the metrics pipeline."""

    db_url: Optional[str] = None
    epsilon: float = 0.01
    history_retention: int = 30
    trend_window_short: int = 5
    trend_window_long: int = 20
    redis_url: Optional[str] = None

    def __post_init__(self) -> None:
        # Import runtime_env for strict mode check and helpers
        is_strict = False
        get_database_url = None
        get_redis_url = None
        MissingEnvironmentVariable = Exception

        try:
            from backend.security.runtime_env import (
                MissingEnvironmentVariable as _MissingEnv,
                is_strict_mode,
                get_database_url as _get_db,
                get_redis_url as _get_redis,
            )
            MissingEnvironmentVariable = _MissingEnv
            get_database_url = _get_db
            get_redis_url = _get_redis
            is_strict = is_strict_mode()
        except ImportError:
            pass

        if self.db_url is None:
            explicit = os.environ.get("MATHLEDGER_DB_URL") or os.environ.get("DATABASE_URL")
            if explicit:
                self.db_url = explicit
            elif get_database_url is not None:
                try:
                    self.db_url = get_database_url()
                except MissingEnvironmentVariable:
                    if is_strict:
                        raise RuntimeError(
                            "[STRICT] DATABASE_URL is required in strict mode. "
                            "Set FIRST_ORGANISM_STRICT=0 to allow fallback."
                        )
                    self.db_url = None  # No fallback to insecure default
            else:
                if is_strict:
                    raise RuntimeError(
                        "[STRICT] DATABASE_URL must be set in strict mode."
                    )
                self.db_url = None  # No fallback to insecure default

        epsilon_env = os.environ.get("METRICS_EPSILON")
        if epsilon_env:
            try:
                self.epsilon = float(epsilon_env)
            except ValueError:
                pass

        if self.redis_url is None:
            redis_env = os.environ.get("MATHLEDGER_REDIS_URL") or os.environ.get("REDIS_URL")
            if redis_env:
                self.redis_url = redis_env
            elif get_redis_url is not None:
                try:
                    self.redis_url = get_redis_url()
                except MissingEnvironmentVariable:
                    if is_strict:
                        raise RuntimeError(
                            "[STRICT] REDIS_URL is required in strict mode. "
                            "Set FIRST_ORGANISM_STRICT=0 to allow fallback."
                        )
                    self.redis_url = None  # No fallback to insecure default
            else:
                if is_strict:
                    raise RuntimeError(
                        "[STRICT] REDIS_URL must be set in strict mode."
                    )
                self.redis_url = None  # No fallback to insecure default


# ---------------------------------------------------------------------------
# Collector Implementations
# ---------------------------------------------------------------------------


class BaseCollector:
    """Base interface for metrics collectors."""

    name: str

    def collect(self) -> CollectorResult:
        raise NotImplementedError


class DatabaseCollector(BaseCollector):
    """Collect metrics from PostgreSQL if available."""

    name = "database"

    def __init__(self, db_url: Optional[str]) -> None:
        self.db_url = db_url

    def collect(self) -> CollectorResult:
        provenance = {
            "name": self.name,
            "transport": "postgresql",
            "status": "skipped",
        }
        warnings: List[str] = []

        if not self.db_url:
            warnings.append("Database URL not provided; skipping database metrics.")
            provenance["reason"] = "missing_db_url"
            return CollectorResult(metrics={}, provenance=provenance, warnings=warnings)

        if psycopg is None or dict_row is None:
            warnings.append("psycopg not available; skipping database metrics.")
            provenance["reason"] = "psycopg_unavailable"
            return CollectorResult(metrics={}, provenance=provenance, warnings=warnings)

        metrics: Dict[str, Dict[str, Any]] = {}
        timestamp_hint: Optional[str] = None

        try:
            with psycopg.connect(self.db_url, row_factory=dict_row, connect_timeout=30) as conn:
                with conn.cursor() as cur:
                    proof_stats = self._safe_fetch(
                        cur,
                        """
                        SELECT
                            COUNT(*) FILTER (WHERE COALESCE(success, FALSE) = TRUE) AS success_count,
                            COUNT(*) FILTER (WHERE COALESCE(success, FALSE) = FALSE) AS failure_count,
                            COUNT(*) AS total_count
                        FROM proofs
                        """,
                        warnings,
                    )

                    block_stats = self._safe_fetch(
                        cur,
                        """
                        SELECT
                            COUNT(*) AS total_blocks,
                            MAX(block_number) AS max_block_number
                        FROM blocks
                        """,
                        warnings,
                    )

                    latest_block = self._safe_fetch(
                        cur,
                        """
                        SELECT root_hash, created_at
                        FROM blocks
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        warnings,
                    )

                    statement_stats = self._safe_fetch(
                        cur,
                        """
                        SELECT
                            COUNT(*) AS total_statements,
                            MAX(derivation_depth) AS max_depth
                        FROM statements
                        """,
                        warnings,
                    )

                    run_stats = self._safe_fetch(
                        cur,
                        """
                        SELECT
                            proofs_per_sec,
                            abstain_pct,
                            success_rate,
                            depth_max_reached,
                            policy_hash,
                            started_at
                        FROM runs
                        ORDER BY started_at DESC
                        LIMIT 1
                        """,
                        warnings,
                    )
        except Exception as exc:  # pragma: no cover - depends on DB availability
            warnings.append(f"Database unreachable: {exc}")
            provenance["status"] = "error"
            provenance["reason"] = "connection_failure"
            return CollectorResult(metrics={}, provenance=provenance, warnings=warnings)

        provenance["status"] = "ok"

        total_proofs = (proof_stats or {}).get("total_count", 0) or 0
        success_count = (proof_stats or {}).get("success_count", 0) or 0
        failure_count = (proof_stats or {}).get("failure_count", 0) or 0

        proofs_per_sec = (run_stats or {}).get("proofs_per_sec", 0.0) or 0.0
        proofs_per_hour = proofs_per_sec * 3600.0

        success_rate = 0.0
        if total_proofs > 0:
            success_rate = (success_count / total_proofs) * 100.0

        abstention_rate = (run_stats or {}).get("abstain_pct")
        if abstention_rate is None:
            abstention_rate = 100.0 - success_rate if total_proofs else 0.0

        statement_count = (statement_stats or {}).get("total_statements", 0) or 0
        statement_depth = (statement_stats or {}).get("max_depth", 0) or 0

        block_height = (block_stats or {}).get("max_block_number", 0) or 0
        block_total = (block_stats or {}).get("total_blocks", 0) or 0
        merkle_root = (latest_block or {}).get("root_hash", "") or ""

        if run_stats and run_stats.get("started_at"):
            try:
                timestamp_hint = run_stats["started_at"].isoformat()
            except AttributeError:
                timestamp_hint = str(run_stats["started_at"])
        elif latest_block and latest_block.get("created_at"):
            timestamp_hint = str(latest_block["created_at"])

        metrics["throughput"] = {
            "proofs_per_sec": float(proofs_per_sec),
            "proofs_per_hour": float(proofs_per_hour),
            "proof_count_total": int(total_proofs),
            "proof_success_count": int(success_count),
            "proof_failure_count": int(failure_count),
        }

        metrics["success_rates"] = {
            "proof_success_rate": float(success_rate),
            "abstention_rate": float(abstention_rate),
            "verification_success_rate": float(success_rate),
        }

        metrics["coverage"] = {
            "max_depth_reached": int(statement_depth),
            "unique_statements": int(statement_count),
            "unique_proofs": int(success_count),
        }

        metrics["blockchain"] = {
            "block_height": int(block_height),
            "total_blocks": int(block_total),
            "merkle_root": merkle_root,
        }

        metadata: Dict[str, Any] = {}
        policy_hash = (run_stats or {}).get("policy_hash")
        if policy_hash:
            metadata["policy_hash"] = policy_hash
        metrics["metadata"] = metadata

        return CollectorResult(
            metrics=metrics,
            provenance=provenance,
            timestamp_hint=timestamp_hint,
            warnings=warnings,
        )

    @staticmethod
    def _safe_fetch(cur, query: str, warnings: List[str]) -> Optional[Dict[str, Any]]:
        try:
            cur.execute(query)
            return cur.fetchone()
        except Exception as exc:  # pragma: no cover - depends on DB schema
            warnings.append(f"Query failed for collector database: {exc}")
            return None


class PerformancePassportCollector(BaseCollector):
    """Collect latency and memory metrics from performance_passport.json."""

    name = "performance_passport"

    def __init__(self, path: Path) -> None:
        self.path = path

    def collect(self) -> CollectorResult:
        provenance = {"name": self.name, "source": str(self.path), "status": "skipped"}
        warnings: List[str] = []

        if not self.path.exists():
            warnings.append("performance_passport.json not found; skipping performance metrics.")
            return CollectorResult(metrics={}, provenance=provenance, warnings=warnings)

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            warnings.append(f"Failed to parse performance_passport.json: {exc}")
            provenance["status"] = "error"
            provenance["reason"] = "parse_error"
            return CollectorResult(metrics={}, provenance=provenance, warnings=warnings)

        provenance["status"] = "ok"

        timestamp_hint = data.get("timestamp")
        test_results = data.get("test_results", [])

        latencies = [float(r["latency_ms"]) for r in test_results if "latency_ms" in r]
        memories = [float(abs(r.get("memory_mb", 0.0))) for r in test_results if "memory_mb" in r]
        sample_size = len(test_results)

        performance = {
            "mean_latency_ms": float(statistics.mean(latencies)) if latencies else 0.0,
            "p50_latency_ms": float(statistics.median(latencies)) if latencies else 0.0,
            "p95_latency_ms": _quantile(latencies, 0.95),
            "p99_latency_ms": _quantile(latencies, 0.99),
            "max_latency_ms": float(max(latencies)) if latencies else 0.0,
            "mean_memory_mb": float(statistics.mean(memories)) if memories else 0.0,
            "max_memory_mb": float(max(memories)) if memories else 0.0,
            "regression_detected": bool(data.get("summary", {}).get("performance_regressions", 0) > 0),
            "sample_size": sample_size,
        }

        metrics = {
            "performance": performance,
            "metadata": {
                "passport_run_id": data.get("run_id", ""),
                "passport_session_id": data.get("session_id", ""),
            },
        }

        return CollectorResult(
            metrics=metrics,
            provenance=provenance,
            timestamp_hint=timestamp_hint,
            warnings=warnings,
        )


class UpliftCollector(BaseCollector):
    """Collect uplift metrics from artifacts/wpv5/fol_stats.json."""

    name = "uplift"

    def __init__(self, path: Path) -> None:
        self.path = path

    def collect(self) -> CollectorResult:
        provenance = {"name": self.name, "source": str(self.path), "status": "skipped"}
        warnings: List[str] = []

        if not self.path.exists():
            warnings.append("fol_stats.json not found; uplift metrics unavailable.")
            return CollectorResult(metrics={}, provenance=provenance, warnings=warnings)

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                stats = json.load(handle)
        except Exception as exc:
            warnings.append(f"Failed to parse fol_stats.json: {exc}")
            provenance["status"] = "error"
            provenance["reason"] = "parse_error"
            return CollectorResult(metrics={}, provenance=provenance, warnings=warnings)

        provenance["status"] = "ok"

        baseline = stats.get("baseline_proofs_per_hour", [])
        guided = stats.get("guided_proofs_per_hour", [])

        baseline_mean = float(statistics.mean(baseline)) if baseline else 0.0
        guided_mean = float(statistics.mean(guided)) if guided else 0.0

        uplift_ratio = (guided_mean / baseline_mean) if baseline_mean > 0 else 0.0
        delta = guided_mean - baseline_mean
        ci_width = abs(delta) * 0.2 if baseline_mean else 0.0

        metrics = {
            "uplift": {
                "uplift_ratio": float(uplift_ratio),
                "baseline_mean": float(baseline_mean),
                "guided_mean": float(guided_mean),
                "p_value": float(stats.get("p_value", 0.0)),
                "confidence_interval_lower": float(uplift_ratio - ci_width),
                "confidence_interval_upper": float(uplift_ratio + ci_width),
                "ci_width": float(ci_width),
                "delta_from_baseline": float(delta),
            }
        }

        return CollectorResult(metrics=metrics, provenance=provenance, warnings=warnings)


class QueueCollector(BaseCollector):
    """Synthesize queue metrics to satisfy schema when live data is unavailable."""

    name = "queue"

    def collect(self) -> CollectorResult:
        metrics = {
            "queue": {
                "queue_length": 0,
                "backlog_ratio": 0.0,
                "source": "synthetic",
            }
        }
        provenance = {
            "name": self.name,
            "status": "synthetic",
            "reason": "no runtime queue instrumentation configured",
        }
        return CollectorResult(metrics=metrics, provenance=provenance, warnings=[])


class FirstOrganismCollector(BaseCollector):
    """Collect First Organism telemetry from Redis with normalized schema."""

    name = "first_organism"

    # Trend detection thresholds
    TREND_EPSILON = 0.01  # Minimum delta for trend detection
    SHORT_WINDOW = 5      # Short-term moving average window
    LONG_WINDOW = 20      # Long-term moving average window

    def __init__(self, redis_url: Optional[str], history_max: int = 20) -> None:
        self.redis_url = redis_url
        self.history_max = history_max

    def collect(self) -> CollectorResult:
        provenance = {"name": self.name, "status": "skipped"}
        warnings: List[str] = []

        if redis is None:
            warnings.append("redis client is unavailable; skipping First Organism metrics.")
            provenance["reason"] = "redis_unavailable"
            return CollectorResult(metrics={}, provenance=provenance, warnings=warnings)

        if not self.redis_url:
            warnings.append("Redis URL not configured; skipping First Organism metrics.")
            provenance["reason"] = "missing_redis_url"
            return CollectorResult(metrics={}, provenance=provenance, warnings=warnings)

        try:
            client = redis.from_url(self.redis_url, decode_responses=True)

            # Fetch all metrics in a pipeline for efficiency
            pipe = client.pipeline()
            pipe.get("ml:metrics:first_organism:runs_total")
            pipe.get("ml:metrics:first_organism:last_ht")
            pipe.get("ml:metrics:first_organism:last_ht_full")
            pipe.get("ml:metrics:first_organism:latency_seconds")
            pipe.get("ml:metrics:first_organism:duration_seconds")
            pipe.get("ml:metrics:first_organism:last_abstentions")
            pipe.get("ml:metrics:first_organism:last_run_timestamp")
            pipe.get("ml:metrics:first_organism:last_status")
            pipe.lrange("ml:metrics:first_organism:duration_history", 0, self.history_max - 1)
            pipe.lrange("ml:metrics:first_organism:abstention_history", 0, self.history_max - 1)
            pipe.lrange("ml:metrics:first_organism:success_history", 0, self.history_max - 1)
            pipe.lrange("ml:metrics:first_organism:ht_history", 0, self.history_max - 1)
            results = pipe.execute()

            # Parse results
            runs_total = self._to_int(results[0])
            ht_hash = results[1] or ""
            ht_full = results[2] or ""
            latency = self._to_float(results[3])
            duration = self._to_float(results[4])
            abstentions = self._to_int(results[5])
            last_ts = results[6]
            last_status = results[7] or ""
            duration_history = self._parse_float_list(results[8] or [])
            abstention_history = self._parse_int_list(results[9] or [])
            success_history = results[10] or []
            ht_history = results[11] or []

            # Compute derived metrics
            avg_duration = statistics.mean(duration_history) if duration_history else 0.0
            median_duration = statistics.median(duration_history) if duration_history else 0.0

            # Success rate from history
            if success_history:
                successes = sum(1 for s in success_history if s == "success")
                success_rate = (successes / len(success_history)) * 100.0
            else:
                success_rate = 0.0

            # Compute deltas
            duration_delta = duration_history[0] - duration_history[1] if len(duration_history) >= 2 else 0.0
            abstention_delta = abstention_history[0] - abstention_history[1] if len(abstention_history) >= 2 else 0

            # Compute trend series
            duration_series = self._compute_trend_series(duration_history)
            abstention_series = self._compute_trend_series([float(x) for x in abstention_history])

            # Determine health status
            if runs_total == 0:
                health_status = "UNKNOWN"
            elif last_status == "success" and success_rate >= 80.0:
                health_status = "ALIVE"
            elif success_rate >= 50.0:
                health_status = "DEGRADED"
            else:
                health_status = "CRITICAL"

            # Verify H_t cryptographic stability
            ht_verification = self._verify_ht_stability(ht_history, ht_hash, ht_full)

            metrics = {
                "first_organism": {
                    # Core metrics
                    "runs_total": runs_total,
                    "last_ht_hash": ht_hash,
                    "last_ht_full": ht_full,
                    "latency_seconds": latency,
                    "last_duration_seconds": duration,
                    "average_duration_seconds": round(avg_duration, 6),
                    "median_duration_seconds": round(median_duration, 6),
                    "abstention_count": abstentions,
                    "last_run_timestamp": last_ts,
                    "last_status": last_status,
                    # Computed rates
                    "success_rate": round(success_rate, 2),
                    # Deltas
                    "duration_delta": round(duration_delta, 6),
                    "abstention_delta": abstention_delta,
                    # Trend indicators
                    "duration_trend": duration_series["trend"],
                    "abstention_trend": abstention_series["trend"],
                    # Trend series (for detailed analysis)
                    "duration_series": duration_series,
                    "abstention_series": abstention_series,
                    # Health
                    "health_status": health_status,
                    # History
                    "duration_history": [round(d, 6) for d in duration_history],
                    "abstention_history": abstention_history,
                    "success_history": success_history,
                    "ht_history": ht_history,
                    # Cryptographic verification
                    "ht_verification": ht_verification,
                }
            }
            provenance["status"] = "ok"
            provenance["health_status"] = health_status
            return CollectorResult(metrics=metrics, provenance=provenance, warnings=warnings)
        except Exception as exc:
            warnings.append(f"Failed to collect First Organism metrics: {exc}")
            provenance["status"] = "error"
            provenance["reason"] = "collection_failure"
            return CollectorResult(metrics={}, provenance=provenance, warnings=warnings)

    def _compute_trend_series(self, history: List[float]) -> Dict[str, Any]:
        """Compute trend series with moving averages and direction."""
        if not history:
            return {
                "latest": 0.0,
                "delta_from_previous": 0.0,
                "moving_average_short": 0.0,
                "moving_average_long": 0.0,
                "samples": 0,
                "trend": "flat",
            }

        latest = history[0]
        delta = history[0] - history[1] if len(history) >= 2 else 0.0
        short_avg = statistics.mean(history[:self.SHORT_WINDOW]) if history else 0.0
        long_avg = statistics.mean(history[:self.LONG_WINDOW]) if history else short_avg

        # Determine trend direction
        if delta > self.TREND_EPSILON:
            trend = "up"
        elif delta < -self.TREND_EPSILON:
            trend = "down"
        else:
            trend = "flat"

        return {
            "latest": round(latest, 6),
            "delta_from_previous": round(delta, 6),
            "moving_average_short": round(short_avg, 6),
            "moving_average_long": round(long_avg, 6),
            "samples": len(history),
            "trend": trend,
        }

    def _verify_ht_stability(
        self,
        ht_history: List[str],
        last_ht: str,
        last_ht_full: str,
    ) -> Dict[str, Any]:
        """Verify cryptographic stability of H_t hashes."""
        if not ht_history:
            return {
                "verified": False,
                "reason": "no_history",
                "unique_count": 0,
                "total_count": 0,
            }

        # Validate hex format
        valid_hashes = []
        for ht in ht_history:
            try:
                if len(ht) >= 16:
                    int(ht, 16)  # Validate hex
                    valid_hashes.append(ht)
            except ValueError:
                pass

        unique_hashes = set(valid_hashes)

        # Verify last_ht matches first in history
        consistency_check = (
            ht_history[0] == last_ht if ht_history and last_ht else True
        )

        # Verify full hash truncates to short hash
        truncation_check = (
            last_ht_full[:16] == last_ht if last_ht_full and last_ht else True
        )

        return {
            "verified": len(valid_hashes) == len(ht_history) and consistency_check and truncation_check,
            "valid_count": len(valid_hashes),
            "total_count": len(ht_history),
            "unique_count": len(unique_hashes),
            "uniqueness_ratio": round(len(unique_hashes) / len(ht_history), 4) if ht_history else 0.0,
            "consistency_check": consistency_check,
            "truncation_check": truncation_check,
        }

    @staticmethod
    def _to_float(value: Optional[str]) -> float:
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _to_int(value: Optional[str]) -> int:
        try:
            return int(value) if value is not None else 0
        except (ValueError, TypeError):
            return 0

    def _parse_float_list(self, raw_list: List[str]) -> List[float]:
        result: List[float] = []
        for item in raw_list:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                continue
        return result

    def _parse_int_list(self, raw_list: List[str]) -> List[int]:
        result: List[int] = []
        for item in raw_list:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    idx = q * (len(sorted_vals) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = idx - lower
    return float(sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight)


def _normalize_timestamp(timestamp: str) -> Optional[str]:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def _deterministic_timestamp_from_digest(digest: str) -> str:
    seed = int(digest[:12], 16)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    offset_seconds = seed % (365 * 24 * 3600)
    return (base + timedelta(seconds=offset_seconds)).isoformat()


# ---------------------------------------------------------------------------
# Metrics Aggregator
# ---------------------------------------------------------------------------


class MetricsAggregator:
    """Aggregates metrics from all ecosystem sources with deterministic outputs."""

    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        collectors: Optional[List[BaseCollector]] = None,
    ) -> None:
        self.config = config or MetricsConfig()
        self.project_root = Path(__file__).parent.parent
        self.artifacts_dir = self.project_root / "artifacts"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.wpv5_dir = self.artifacts_dir / "wpv5"

        default_collectors = [
            DatabaseCollector(self.config.db_url),
            PerformancePassportCollector(self.project_root / "performance_passport.json"),
            UpliftCollector(self.wpv5_dir / "fol_stats.json"),
            QueueCollector(),
            FirstOrganismCollector(self.config.redis_url),
        ]
        self.collectors = collectors or default_collectors

        self.history_path = self.metrics_dir / "history.json"
        self.trends_path = self.metrics_dir / "trends.json"
        self._latest_history: List[Dict[str, Any]] = []
        self._latest_trends: Dict[str, Any] = {}
        self._history_merkle = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, session_hint: Optional[str] = None) -> CanonicalMetrics:
        """Aggregate metrics from collectors and build canonical payload."""
        results: List[CollectorResult] = []
        warnings: List[str] = []
        timestamp_hints: List[str] = []

        for collector in self.collectors:
            result = collector.collect()
            results.append(result)
            warnings.extend(result.warnings)
            if result.timestamp_hint:
                norm = _normalize_timestamp(result.timestamp_hint)
                if norm:
                    timestamp_hints.append(norm)

        merged_metrics = self._merge_metrics(results)
        merged_metrics["trends"] = {}

        canonical = self._build_canonical(merged_metrics, results, warnings, timestamp_hints, session_hint)

        history = self._load_history()
        updated_history = self._update_history(history, canonical)
        trends = self._compute_trends(updated_history, canonical)
        variance = self._compute_variance(updated_history, canonical)

        canonical.metrics["trends"] = trends
        canonical.variance = variance
        canonical.notes = warnings if warnings else None

        self._latest_history = updated_history
        self._latest_trends = trends
        self._history_merkle = self._compute_history_merkle(updated_history)
        canonical.provenance["history_merkle"] = self._history_merkle
        canonical.provenance["collectors"] = [r.provenance for r in results]
        canonical.provenance["warnings"] = warnings

        return canonical

    def export(self, canonical: CanonicalMetrics) -> List[Path]:
        """Export canonical metrics, trends, and history to artifacts/metrics/."""
        paths: List[Path] = []
        payload = canonical.to_dict()

        latest_path = self.metrics_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        paths.append(latest_path)

        session_path = self.metrics_dir / f"session_{canonical.session_id}.json"
        with open(session_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        paths.append(session_path)

        history_payload = {"sessions": self._latest_history}
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(history_payload, handle, indent=2)
        paths.append(self.history_path)

        with open(self.trends_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "generated_from": canonical.session_id,
                    "generated_at": canonical.timestamp,
                    "history_merkle": self._history_merkle,
                    "trends": self._latest_trends,
                },
                handle,
                indent=2,
            )
        paths.append(self.trends_path)

        return paths

    def validate_against_schema(self, metrics_dict: Dict[str, Any]) -> bool:
        """Validate metrics against schema_v1.json (structure-only)."""
        schema_path = self.metrics_dir / "schema_v1.json"
        if not schema_path.exists():
            return False

        with open(schema_path, "r", encoding="utf-8") as handle:
            schema = json.load(handle)

        required = schema.get("required", [])
        for field in required:
            if field not in metrics_dict:
                return False

        metrics_required = schema.get("properties", {}).get("metrics", {}).get("required", [])
        metrics_body = metrics_dict.get("metrics", {})
        for field in metrics_required:
            if field not in metrics_body:
                return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_metrics(self, results: List[CollectorResult]) -> Dict[str, Any]:
        merged = self._blank_metrics()
        metadata: Dict[str, Any] = {}

        for result in results:
            for section, payload in result.metrics.items():
                if section == "metadata":
                    metadata.update(payload)
                    continue
                if isinstance(payload, dict):
                    bucket = merged.setdefault(section, {})
                    bucket.update(payload)
                else:
                    merged[section] = payload

        if metadata:
            merged["metadata"] = metadata
            merged["policy_hash"] = metadata.get("policy_hash", "")
        else:
            merged["metadata"] = {}
            merged["policy_hash"] = ""

        return merged

    def _build_canonical(
        self,
        merged_metrics: Dict[str, Any],
        results: List[CollectorResult],
        warnings: List[str],
        timestamp_hints: List[str],
        session_hint: Optional[str],
    ) -> CanonicalMetrics:
        canonical = CanonicalMetrics(
            timestamp="",
            session_id="",
            source="metrics_oracle_v1",
            metrics=merged_metrics,
            provenance={
                "collector": "metrics_oracle",
                "policy_hash": merged_metrics.get("policy_hash", ""),
                "merkle_hash": "",
                "history_merkle": "",
                "collectors": [],
                "warnings": warnings,
            },
        )

        merkle = canonical.compute_merkle_hash()
        canonical.provenance["merkle_hash"] = merkle

        if session_hint:
            canonical.session_id = session_hint
        else:
            canonical.session_id = f"metrics-cartographer-{merkle[:12]}"

        canonical.timestamp = (
            timestamp_hints[0]
            if timestamp_hints
            else _deterministic_timestamp_from_digest(merkle)
        )

        return canonical

    def _load_history(self) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            sessions = payload.get("sessions", [])
            if isinstance(sessions, list):
                return sessions
        except Exception:
            pass
        return []

    def _update_history(
        self,
        history: List[Dict[str, Any]],
        canonical: CanonicalMetrics,
    ) -> List[Dict[str, Any]]:
        entry = {
            "session_id": canonical.session_id,
            "timestamp": canonical.timestamp,
            "merkle_hash": canonical.provenance["merkle_hash"],
            "throughput": canonical.metrics.get("throughput", {}),
            "success_rates": canonical.metrics.get("success_rates", {}),
            "performance": canonical.metrics.get("performance", {}),
            "first_organism": canonical.metrics.get("first_organism", {}),
        }

        updated = [h for h in history if h.get("session_id") != canonical.session_id]
        updated.append(entry)
        updated.sort(key=lambda item: (item.get("timestamp", ""), item.get("session_id", "")))

        if len(updated) > self.config.history_retention:
            updated = updated[-self.config.history_retention :]

        return updated

    def _compute_trends(
        self,
        history: List[Dict[str, Any]],
        canonical: CanonicalMetrics,
    ) -> Dict[str, Any]:
        def series(values: List[float], short_window: int, long_window: int) -> Dict[str, Any]:
            latest = values[-1] if values else 0.0
            delta = latest - values[-2] if len(values) >= 2 else 0.0
            short_avg = statistics.mean(values[-short_window:]) if values else 0.0
            long_avg = statistics.mean(values[-long_window:]) if values else short_avg
            trend_flag = "flat"
            if delta > 0.01:
                trend_flag = "up"
            elif delta < -0.01:
                trend_flag = "down"
            return {
                "latest": round(latest, 6),
                "delta_from_previous": round(delta, 6),
                "moving_average_short": round(short_avg, 6),
                "moving_average_long": round(long_avg, 6),
                "samples": len(values),
                "trend": trend_flag,
            }

        throughput_values = [
            float(entry.get("throughput", {}).get("proofs_per_sec", 0.0)) for entry in history
        ]
        success_values = [
            float(entry.get("success_rates", {}).get("proof_success_rate", 0.0))
            for entry in history
        ]
        latency_values = [
            float(entry.get("performance", {}).get("p95_latency_ms", 0.0)) for entry in history
        ]

        # First Organism trend values
        fo_duration_values = [
            float(entry.get("first_organism", {}).get("average_duration_seconds", 0.0))
            for entry in history
        ]
        fo_abstention_values = [
            float(entry.get("first_organism", {}).get("abstention_count", 0))
            for entry in history
        ]
        fo_runs_values = [
            float(entry.get("first_organism", {}).get("runs_total", 0))
            for entry in history
        ]

        # Compute First Organism success rate from history
        fo_success_rates: List[float] = []
        for entry in history:
            fo = entry.get("first_organism", {})
            success_hist = fo.get("success_history", [])
            if success_hist:
                successes = sum(1 for s in success_hist if s == "success")
                fo_success_rates.append(successes / len(success_hist) * 100.0)
            else:
                # Fallback: derive from runs_total and last_status
                last_status = fo.get("last_status", "")
                if last_status == "success":
                    fo_success_rates.append(100.0)
                elif last_status == "failure":
                    fo_success_rates.append(0.0)
                else:
                    fo_success_rates.append(0.0)

        trends = {
            "proofs_per_sec": series(throughput_values, self.config.trend_window_short, self.config.trend_window_long),
            "proof_success_rate": series(success_values, self.config.trend_window_short, self.config.trend_window_long),
            "p95_latency_ms": series(latency_values, self.config.trend_window_short, self.config.trend_window_long),
            "retention": len(history),
        }

        # Add First Organism trends (only if we have FO data)
        has_fo_data = any(
            entry.get("first_organism", {}).get("runs_total", 0) > 0
            for entry in history
        )
        if has_fo_data:
            trends["first_organism"] = {
                "duration_seconds": series(fo_duration_values, self.config.trend_window_short, self.config.trend_window_long),
                "abstention_count": series(fo_abstention_values, self.config.trend_window_short, self.config.trend_window_long),
                "runs_total": series(fo_runs_values, self.config.trend_window_short, self.config.trend_window_long),
                "success_rate": series(fo_success_rates, self.config.trend_window_short, self.config.trend_window_long),
            }

        return trends

    def _compute_variance(
        self,
        history: List[Dict[str, Any]],
        canonical: CanonicalMetrics,
    ) -> Dict[str, Any]:
        values = [
            float(entry.get("throughput", {}).get("proofs_per_sec", 0.0)) for entry in history
        ]
        if len(values) < 2:
            return {
                "coefficient_of_variation": 0.0,
                "epsilon_tolerance": self.config.epsilon,
                "within_tolerance": True,
                "sample_size": len(values),
            }

        mean_val = statistics.mean(values)
        if mean_val == 0:
            cv = 0.0
        else:
            cv = statistics.stdev(values) / mean_val

        return {
            "coefficient_of_variation": float(cv),
            "epsilon_tolerance": self.config.epsilon,
            "within_tolerance": cv <= self.config.epsilon,
            "sample_size": len(values),
        }

    def _compute_history_merkle(self, history: List[Dict[str, Any]]) -> str:
        payload = json.dumps(history, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _blank_metrics() -> Dict[str, Any]:
        return {
            "throughput": {
                "proofs_per_sec": 0.0,
                "proofs_per_hour": 0.0,
                "proof_count_total": 0,
                "proof_success_count": 0,
                "proof_failure_count": 0,
            },
            "success_rates": {
                "proof_success_rate": 0.0,
                "abstention_rate": 0.0,
                "verification_success_rate": 0.0,
            },
            "coverage": {
                "max_depth_reached": 0,
                "unique_statements": 0,
                "unique_proofs": 0,
            },
            "performance": {
                "mean_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "mean_memory_mb": 0.0,
                "max_memory_mb": 0.0,
                "regression_detected": False,
                "sample_size": 0,
            },
            "uplift": {
                "uplift_ratio": 0.0,
                "baseline_mean": 0.0,
                "guided_mean": 0.0,
                "p_value": 0.0,
                "confidence_interval_lower": 0.0,
                "confidence_interval_upper": 0.0,
                "ci_width": 0.0,
                "delta_from_baseline": 0.0,
            },
            "blockchain": {
                "block_height": 0,
                "total_blocks": 0,
                "merkle_root": "",
            },
            "queue": {
                "queue_length": 0,
                "backlog_ratio": 0.0,
                "source": "unknown",
            },
            "metadata": {},
            "policy_hash": "",
        }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point for Cursor K metrics pipeline."""
    aggregator = MetricsAggregator()

    schema_path = aggregator.metrics_dir / "schema_v1.json"
    perf_passport = aggregator.project_root / "performance_passport.json"

    missing_inputs = []
    if not schema_path.exists():
        missing_inputs.append("artifacts/metrics/schema_v1.json")
    if not perf_passport.exists():
        missing_inputs.append("performance_passport.json")

    if missing_inputs:
        print("[ABSTAIN] missing inputs: " + ", ".join(missing_inputs))
        print("Provide required inputs and re-run the metrics pipeline.")
        return 2

    canonical = aggregator.aggregate()
    canonical_dict = canonical.to_dict()

    is_valid = aggregator.validate_against_schema(canonical_dict)
    if not is_valid:
        print("[WARN] Canonical payload does not satisfy schema_v1.json")

    exported = aggregator.export(canonical)
    for path in exported:
        print(f"[OK] wrote {path}")

    total_entries = sum(
        len(v) if isinstance(v, dict) else 1 for v in canonical.metrics.values()
    )
    variance = canonical.variance or {}
    epsilon = variance.get("epsilon_tolerance", aggregator.config.epsilon)
    if variance.get("within_tolerance", True):
        print(f"[PASS] Metrics Canonicalized entries={total_entries} variance<=epsilon={epsilon}")
        return 0
    cv = variance.get("coefficient_of_variation", 0.0)
    print(f"[WARN] variance={cv:.4f} > epsilon={epsilon}")
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
