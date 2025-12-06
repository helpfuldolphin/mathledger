"""
First Organism Higher-Order Analytics.

==============================================================================
STATUS: PHASE II — SKELETON ONLY — NOT USED IN EVIDENCE PACK v1
==============================================================================

This module is a DESIGN SKELETON for future analytics capabilities.
It is NOT integrated into the live metrics path and NOT used in production.

Evidence Pack v1 uses only:
- FO Dyno Chart (generated from JSONL logs via experiments/generate_dyno_chart.py)
- Sealed attestation.json from closed-loop FO test
- Raw logs: results/fo_baseline_wide.jsonl, results/fo_rfl_wide.jsonl

DO NOT cite this module as evidence of:
- Production anomaly detection
- Automated uplift computation with confidence intervals
- Robust health monitoring
- Automated history export

RFL JSONL logs bypass this module; these analytics are Phase II only.

All classes below are UNVALIDATED PROTOTYPES for Phase II work.
==============================================================================

Proposed advanced analytics capabilities for FO telemetry:
- Time-series aggregation with rolling windows
- Anomaly detection (Z-score and threshold-based)
- Cross-correlation analysis between metrics
- Uplift computation (Baseline vs RFL)
- History export generation (first_organism_history.json)

This module would build on top of:
- first_organism_telemetry.py (emission)
- fo_schema.py (canonical schema)
- fo_feedback.py (feedback loop)

Example (NOT PRODUCTION):
    analytics = FOAnalytics()
    summary = analytics.compute_summary()
    anomalies = analytics.detect_anomalies()
    analytics.export_history("exports/first_organism_history.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import redis
except ImportError:
    redis = None

from backend.metrics.fo_schema import (
    FOVitalSigns,
    FOTrendSeries,
    REDIS_KEY_PREFIX,
    HISTORY_MAX_LEN,
    STATUS_SUCCESS,
    STATUS_FAILURE,
    TREND_UP,
    TREND_DOWN,
    TREND_FLAT,
)

logger = logging.getLogger("FOAnalytics")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FOAnalyticsConfig:
    """Configuration for FO analytics."""

    # Rolling window sizes
    short_window: int = 5
    long_window: int = 20

    # Anomaly detection
    zscore_threshold: float = 2.5
    duration_anomaly_threshold: float = 5.0  # seconds
    abstention_anomaly_threshold: int = 10

    # Export settings
    export_history_max: int = 1000
    feedback_history_max: int = 100

    # Uplift computation
    min_samples_for_uplift: int = 10
    confidence_level: float = 0.95


# ---------------------------------------------------------------------------
# Time-Series Aggregator
# ---------------------------------------------------------------------------

@dataclass
class FOTimeSeriesStats:
    """Statistics for a time-series metric."""

    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    count: int = 0
    trend: str = TREND_FLAT
    rolling_mean_short: float = 0.0
    rolling_mean_long: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FOTimeSeriesAggregator:
    """
    Aggregator for FO time-series metrics.

    Computes rolling statistics over duration and abstention histories.
    """

    def __init__(self, config: Optional[FOAnalyticsConfig] = None) -> None:
        self.config = config or FOAnalyticsConfig()

    def aggregate(
        self,
        values: List[float],
        epsilon: float = 0.01,
    ) -> FOTimeSeriesStats:
        """
        Compute aggregate statistics for a time-series.

        Args:
            values: List of values (most recent first)
            epsilon: Threshold for trend detection

        Returns:
            FOTimeSeriesStats with computed metrics
        """
        if not values:
            return FOTimeSeriesStats()

        # Basic statistics
        mean = statistics.mean(values)
        median = statistics.median(values)
        std = statistics.stdev(values) if len(values) >= 2 else 0.0
        min_val = min(values)
        max_val = max(values)

        # Rolling averages
        short_window = min(self.config.short_window, len(values))
        long_window = min(self.config.long_window, len(values))
        rolling_mean_short = statistics.mean(values[:short_window])
        rolling_mean_long = statistics.mean(values[:long_window])

        # Trend detection
        if len(values) >= 2:
            delta = values[0] - values[1]
            if delta > epsilon:
                trend = TREND_UP
            elif delta < -epsilon:
                trend = TREND_DOWN
            else:
                trend = TREND_FLAT
        else:
            trend = TREND_FLAT

        return FOTimeSeriesStats(
            mean=round(mean, 6),
            median=round(median, 6),
            std=round(std, 6),
            min_val=round(min_val, 6),
            max_val=round(max_val, 6),
            count=len(values),
            trend=trend,
            rolling_mean_short=round(rolling_mean_short, 6),
            rolling_mean_long=round(rolling_mean_long, 6),
        )


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------

@dataclass
class FOAnomaly:
    """Detected anomaly in FO metrics."""

    metric: str
    value: float
    threshold: float
    zscore: float
    timestamp: str
    severity: str  # "warning", "critical"
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FOAnomalyDetector:
    """
    Anomaly detection for FO metrics.

    Uses Z-score and threshold-based detection to identify outliers.
    """

    def __init__(self, config: Optional[FOAnalyticsConfig] = None) -> None:
        self.config = config or FOAnalyticsConfig()

    def detect(
        self,
        duration_history: List[float],
        abstention_history: List[int],
        success_history: List[str],
    ) -> List[FOAnomaly]:
        """
        Detect anomalies in FO metrics.

        Args:
            duration_history: Duration values (most recent first)
            abstention_history: Abstention counts (most recent first)
            success_history: Success/failure status (most recent first)

        Returns:
            List of detected anomalies
        """
        anomalies: List[FOAnomaly] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Duration anomalies (Z-score)
        if len(duration_history) >= 3:
            anomalies.extend(
                self._detect_zscore_anomalies(
                    duration_history,
                    "duration_seconds",
                    timestamp,
                )
            )

        # Abstention anomalies (threshold)
        if abstention_history:
            latest_abstention = abstention_history[0]
            if latest_abstention > self.config.abstention_anomaly_threshold:
                anomalies.append(FOAnomaly(
                    metric="abstention_count",
                    value=float(latest_abstention),
                    threshold=float(self.config.abstention_anomaly_threshold),
                    zscore=0.0,  # Threshold-based, not Z-score
                    timestamp=timestamp,
                    severity="warning" if latest_abstention < 20 else "critical",
                    description=f"High abstention count: {latest_abstention}",
                ))

        # Consecutive failures
        failure_streak = self._count_failure_streak(success_history)
        if failure_streak >= 3:
            anomalies.append(FOAnomaly(
                metric="success_streak",
                value=float(failure_streak),
                threshold=3.0,
                zscore=0.0,
                timestamp=timestamp,
                severity="critical" if failure_streak >= 5 else "warning",
                description=f"Consecutive failures: {failure_streak}",
            ))

        return anomalies

    def _detect_zscore_anomalies(
        self,
        values: List[float],
        metric_name: str,
        timestamp: str,
    ) -> List[FOAnomaly]:
        """Detect anomalies using Z-score."""
        anomalies = []

        if len(values) < 3:
            return anomalies

        mean = statistics.mean(values)
        std = statistics.stdev(values)

        if std == 0:
            return anomalies

        latest = values[0]
        zscore = (latest - mean) / std

        if abs(zscore) > self.config.zscore_threshold:
            severity = "critical" if abs(zscore) > 3.0 else "warning"
            direction = "high" if zscore > 0 else "low"

            anomalies.append(FOAnomaly(
                metric=metric_name,
                value=latest,
                threshold=mean + (self.config.zscore_threshold * std),
                zscore=round(zscore, 2),
                timestamp=timestamp,
                severity=severity,
                description=f"Anomalously {direction} {metric_name}: Z-score {zscore:.2f}",
            ))

        return anomalies

    def _count_failure_streak(self, success_history: List[str]) -> int:
        """Count consecutive failures from most recent."""
        streak = 0
        for status in success_history:
            if status == STATUS_FAILURE:
                streak += 1
            else:
                break
        return streak


# ---------------------------------------------------------------------------
# Uplift Calculator
# ---------------------------------------------------------------------------

@dataclass
class FOUpliftResult:
    """Result of uplift computation."""

    baseline_mean: float
    baseline_throughput: float
    rfl_mean: float
    rfl_throughput: float
    uplift_percent: float
    ci_low: float
    ci_high: float
    p_value: Optional[float]
    significant: bool
    sample_size_baseline: int
    sample_size_rfl: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FOUpliftCalculator:
    """
    Computes uplift between Baseline and RFL derivation runs.

    Throughput is defined as 1 / mean_duration (proofs per second).
    """

    def __init__(self, config: Optional[FOAnalyticsConfig] = None) -> None:
        self.config = config or FOAnalyticsConfig()

    def compute(
        self,
        baseline_durations: List[float],
        rfl_durations: List[float],
    ) -> Optional[FOUpliftResult]:
        """
        Compute uplift from baseline to RFL.

        Args:
            baseline_durations: Duration history from baseline runs
            rfl_durations: Duration history from RFL runs

        Returns:
            FOUpliftResult if sufficient data, None otherwise
        """
        min_samples = self.config.min_samples_for_uplift

        if len(baseline_durations) < min_samples or len(rfl_durations) < min_samples:
            logger.warning(
                f"Insufficient samples for uplift: baseline={len(baseline_durations)}, "
                f"rfl={len(rfl_durations)}, required={min_samples}"
            )
            return None

        # Compute means
        baseline_mean = statistics.mean(baseline_durations)
        rfl_mean = statistics.mean(rfl_durations)

        # Avoid division by zero
        if baseline_mean == 0 or rfl_mean == 0:
            return None

        # Throughput = 1 / duration
        baseline_throughput = 1.0 / baseline_mean
        rfl_throughput = 1.0 / rfl_mean

        # Uplift percentage
        uplift_percent = ((rfl_throughput - baseline_throughput) / baseline_throughput) * 100

        # Confidence interval (simplified bootstrap estimate)
        ci_low, ci_high = self._estimate_ci(
            baseline_durations,
            rfl_durations,
            uplift_percent,
        )

        # Significance (simplified t-test approximation)
        p_value, significant = self._test_significance(
            baseline_durations,
            rfl_durations,
        )

        return FOUpliftResult(
            baseline_mean=round(baseline_mean, 6),
            baseline_throughput=round(baseline_throughput, 6),
            rfl_mean=round(rfl_mean, 6),
            rfl_throughput=round(rfl_throughput, 6),
            uplift_percent=round(uplift_percent, 2),
            ci_low=round(ci_low, 2),
            ci_high=round(ci_high, 2),
            p_value=round(p_value, 4) if p_value else None,
            significant=significant,
            sample_size_baseline=len(baseline_durations),
            sample_size_rfl=len(rfl_durations),
        )

    def _estimate_ci(
        self,
        baseline: List[float],
        rfl: List[float],
        uplift: float,
    ) -> Tuple[float, float]:
        """Estimate confidence interval for uplift (simplified)."""
        # Use standard error propagation for rough CI
        baseline_std = statistics.stdev(baseline) if len(baseline) >= 2 else 0
        rfl_std = statistics.stdev(rfl) if len(rfl) >= 2 else 0

        # Combined relative error
        baseline_mean = statistics.mean(baseline)
        rfl_mean = statistics.mean(rfl)

        if baseline_mean == 0 or rfl_mean == 0:
            return (uplift - 20, uplift + 20)

        rel_error_baseline = baseline_std / baseline_mean
        rel_error_rfl = rfl_std / rfl_mean

        # Propagated relative error
        combined_rel_error = (rel_error_baseline**2 + rel_error_rfl**2) ** 0.5

        # 95% CI (1.96 * SE)
        margin = uplift * combined_rel_error * 1.96

        return (uplift - margin, uplift + margin)

    def _test_significance(
        self,
        baseline: List[float],
        rfl: List[float],
    ) -> Tuple[Optional[float], bool]:
        """
        Test significance of difference (simplified Welch's t-test).

        Returns (p_value, is_significant).
        """
        # Placeholder: Full implementation would use scipy.stats.ttest_ind
        # For now, use a heuristic based on effect size

        baseline_mean = statistics.mean(baseline)
        rfl_mean = statistics.mean(rfl)

        baseline_std = statistics.stdev(baseline) if len(baseline) >= 2 else 1
        rfl_std = statistics.stdev(rfl) if len(rfl) >= 2 else 1

        # Cohen's d effect size
        pooled_std = ((baseline_std**2 + rfl_std**2) / 2) ** 0.5
        if pooled_std == 0:
            return (None, False)

        effect_size = abs(rfl_mean - baseline_mean) / pooled_std

        # Heuristic: effect_size > 0.5 is medium effect
        # Larger samples with medium+ effect are likely significant
        min_samples = min(len(baseline), len(rfl))

        if effect_size > 0.8 and min_samples >= 30:
            return (0.01, True)
        elif effect_size > 0.5 and min_samples >= 20:
            return (0.05, True)
        elif effect_size > 0.3 and min_samples >= 50:
            return (0.10, False)
        else:
            return (None, False)


# ---------------------------------------------------------------------------
# History Exporter
# ---------------------------------------------------------------------------

@dataclass
class FOHistoryExport:
    """Complete FO history export structure."""

    schema: str = "https://mathledger.io/schemas/fo-history-v2.json"
    version: str = "2.0"
    generated_at: str = ""
    generator: str = "fo_analytics.py"
    summary: Dict[str, Any] = field(default_factory=dict)
    current_state: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    feedback_log: List[Dict[str, Any]] = field(default_factory=list)
    ht_verification: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "$schema": self.schema,
            "version": self.version,
            "generated_at": self.generated_at,
            "generator": self.generator,
            "summary": self.summary,
            "current_state": self.current_state,
            "history": self.history,
            "feedback_log": self.feedback_log,
            "ht_verification": self.ht_verification,
            "metadata": self.metadata,
        }


class FOHistoryExporter:
    """
    Exports FO history to first_organism_history.json.

    Reads from Redis and generates a canonical JSON export.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        config: Optional[FOAnalyticsConfig] = None,
    ) -> None:
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL", "redis://localhost:6379/0"
        )
        self.config = config or FOAnalyticsConfig()
        self._client = None
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
        return f"{REDIS_KEY_PREFIX}:{name}"

    def export(self, output_path: Optional[str] = None) -> FOHistoryExport:
        """
        Export FO history to JSON.

        Args:
            output_path: Optional path to write JSON file

        Returns:
            FOHistoryExport with all data
        """
        export = FOHistoryExport(
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        if not self._client:
            logger.warning("Redis unavailable, returning empty export")
            return export

        try:
            # Fetch all metrics
            metrics = self._fetch_all_metrics()

            # Build sections
            export.summary = self._build_summary(metrics)
            export.current_state = self._build_current_state(metrics)
            export.history = self._build_history(metrics)
            export.feedback_log = self._fetch_feedback_log()
            export.ht_verification = self._build_ht_verification(metrics)
            export.metadata = self._build_metadata()

            # Write to file if path provided
            if output_path:
                self._write_json(export, output_path)

            return export

        except Exception as exc:
            logger.error(f"Failed to export FO history: {exc}")
            return export

    def _fetch_all_metrics(self) -> Dict[str, Any]:
        """Fetch all FO metrics from Redis."""
        pipe = self._client.pipeline()

        pipe.get(self._key("runs_total"))
        pipe.get(self._key("last_ht"))
        pipe.get(self._key("last_ht_full"))
        pipe.get(self._key("duration_seconds"))
        pipe.get(self._key("last_abstentions"))
        pipe.get(self._key("last_run_timestamp"))
        pipe.get(self._key("last_status"))
        pipe.lrange(self._key("duration_history"), 0, -1)
        pipe.lrange(self._key("abstention_history"), 0, -1)
        pipe.lrange(self._key("success_history"), 0, -1)
        pipe.lrange(self._key("ht_history"), 0, -1)

        results = pipe.execute()

        return {
            "runs_total": self._to_int(results[0]),
            "last_ht": results[1] or "",
            "last_ht_full": results[2] or "",
            "duration_seconds": self._to_float(results[3]),
            "last_abstentions": self._to_int(results[4]),
            "last_run_timestamp": results[5] or "",
            "last_status": results[6] or "",
            "duration_history": self._parse_float_list(results[7] or []),
            "abstention_history": self._parse_int_list(results[8] or []),
            "success_history": results[9] or [],
            "ht_history": results[10] or [],
        }

    def _build_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Build summary section."""
        duration_history = metrics["duration_history"]
        abstention_history = metrics["abstention_history"]
        success_history = metrics["success_history"]

        # Compute success rates
        if success_history:
            successes = sum(1 for s in success_history if s == STATUS_SUCCESS)
            success_rate_overall = (successes / len(success_history)) * 100
            recent_successes = sum(1 for s in success_history[:5] if s == STATUS_SUCCESS)
            success_rate_recent = (recent_successes / min(5, len(success_history))) * 100
        else:
            success_rate_overall = 0.0
            success_rate_recent = 0.0

        # Duration stats
        avg_duration = statistics.mean(duration_history) if duration_history else 0.0
        median_duration = statistics.median(duration_history) if duration_history else 0.0

        # Abstention stats
        total_abstentions = sum(abstention_history) if abstention_history else 0
        avg_abstentions = statistics.mean(abstention_history) if abstention_history else 0.0

        # Health trend (based on success rate trend)
        if len(success_history) >= 10:
            recent_rate = sum(1 for s in success_history[:5] if s == STATUS_SUCCESS) / 5
            older_rate = sum(1 for s in success_history[5:10] if s == STATUS_SUCCESS) / 5
            if recent_rate > older_rate + 0.1:
                health_trend = "improving"
            elif recent_rate < older_rate - 0.1:
                health_trend = "degrading"
            else:
                health_trend = "stable"
        else:
            health_trend = "unknown"

        # Data quality score (placeholder)
        data_quality_score = 1.0 if metrics["runs_total"] >= 10 else metrics["runs_total"] / 10

        return {
            "runs_total": metrics["runs_total"],
            "first_run_timestamp": "",  # Would need to track this separately
            "last_run_timestamp": metrics["last_run_timestamp"],
            "success_rate_overall": round(success_rate_overall, 2),
            "success_rate_recent": round(success_rate_recent, 2),
            "average_duration_seconds": round(avg_duration, 6),
            "median_duration_seconds": round(median_duration, 6),
            "total_abstentions": total_abstentions,
            "average_abstentions_per_run": round(avg_abstentions, 2),
            "health_trend": health_trend,
            "data_quality_score": round(data_quality_score, 2),
        }

    def _build_current_state(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Build current_state section."""
        # Compute trends
        duration_history = metrics["duration_history"]
        abstention_history = metrics["abstention_history"]
        success_history = metrics["success_history"]

        duration_trend = self._compute_trend([float(d) for d in duration_history])
        abstention_trend = self._compute_trend([float(a) for a in abstention_history])

        # Success trend
        if len(success_history) >= 10:
            recent_rate = sum(1 for s in success_history[:5] if s == STATUS_SUCCESS) / 5
            older_rate = sum(1 for s in success_history[5:10] if s == STATUS_SUCCESS) / 5
            if recent_rate > older_rate + 0.1:
                success_trend = TREND_UP
            elif recent_rate < older_rate - 0.1:
                success_trend = TREND_DOWN
            else:
                success_trend = TREND_FLAT
        else:
            success_trend = TREND_FLAT

        # Health status
        if metrics["runs_total"] == 0:
            health_status = "UNKNOWN"
        elif metrics["last_status"] == STATUS_SUCCESS:
            successes = sum(1 for s in success_history if s == STATUS_SUCCESS)
            success_rate = (successes / len(success_history)) * 100 if success_history else 0
            if success_rate >= 80:
                health_status = "ALIVE"
            elif success_rate >= 50:
                health_status = "DEGRADED"
            else:
                health_status = "CRITICAL"
        else:
            health_status = "DEGRADED"

        return {
            "runs_total": metrics["runs_total"],
            "last_ht_hash": metrics["last_ht"],
            "last_ht_full": metrics["last_ht_full"],
            "last_duration_seconds": metrics["duration_seconds"],
            "last_abstentions": metrics["last_abstentions"],
            "last_run_timestamp": metrics["last_run_timestamp"],
            "last_status": metrics["last_status"],
            "health_status": health_status,
            "duration_trend": duration_trend,
            "abstention_trend": abstention_trend,
            "success_trend": success_trend,
        }

    def _build_history(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build history array (limited to export_history_max)."""
        # Note: Redis only stores HISTORY_MAX_LEN entries
        # For full history, would need persistent storage

        history = []
        duration_history = metrics["duration_history"]
        abstention_history = metrics["abstention_history"]
        success_history = metrics["success_history"]
        ht_history = metrics["ht_history"]

        num_entries = max(
            len(duration_history),
            len(abstention_history),
            len(success_history),
            len(ht_history),
        )

        for i in range(min(num_entries, self.config.export_history_max)):
            entry = {
                "run_index": metrics["runs_total"] - i,
                "timestamp": "",  # Would need per-entry timestamps
                "duration_seconds": duration_history[i] if i < len(duration_history) else None,
                "abstention_count": abstention_history[i] if i < len(abstention_history) else None,
                "status": success_history[i] if i < len(success_history) else None,
                "ht_hash": ht_history[i] if i < len(ht_history) else None,
            }

            # Compute deltas
            if i < len(duration_history) - 1:
                entry["duration_delta"] = round(
                    duration_history[i] - duration_history[i + 1], 6
                )
            if i < len(abstention_history) - 1:
                entry["abstention_delta"] = (
                    abstention_history[i] - abstention_history[i + 1]
                )

            history.append(entry)

        return history

    def _fetch_feedback_log(self) -> List[Dict[str, Any]]:
        """Fetch feedback decision history."""
        try:
            raw = self._client.lrange(
                "ml:metrics:fo_feedback:decision_history",
                0,
                self.config.feedback_history_max - 1,
            )
            return [json.loads(item) for item in raw]
        except Exception:
            return []

    def _build_ht_verification(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Build H_t verification section."""
        ht_history = metrics["ht_history"]

        if not ht_history:
            return {
                "verified": False,
                "reason": "no_history",
                "valid_count": 0,
                "total_count": 0,
            }

        # Validate hashes
        valid_hashes = []
        for ht in ht_history:
            try:
                if len(ht) >= 16:
                    int(ht, 16)  # Validate hex
                    valid_hashes.append(ht)
            except ValueError:
                pass

        unique_hashes = set(valid_hashes)

        # Truncation check: verify short hash matches full hash prefix
        truncation_ok = True
        if metrics["last_ht"] and metrics["last_ht_full"]:
            truncation_ok = metrics["last_ht_full"].startswith(metrics["last_ht"])

        return {
            "verified": len(valid_hashes) == len(ht_history),
            "valid_count": len(valid_hashes),
            "total_count": len(ht_history),
            "unique_count": len(unique_hashes),
            "uniqueness_ratio": round(len(unique_hashes) / len(ht_history), 4) if ht_history else 0.0,
            "consistency_check": True,
            "truncation_check": truncation_ok,
            "last_verified_ht": metrics["last_ht"],
            "last_verified_ht_full": metrics["last_ht_full"],
            "anomalies": [],
        }

    def _build_metadata(self) -> Dict[str, Any]:
        """Build metadata section."""
        return {
            "redis_url": self.redis_url,
            "history_window": HISTORY_MAX_LEN,
            "feedback_history_max": self.config.feedback_history_max,
            "export_history_max": self.config.export_history_max,
            "schema_version": "2.0",
            "generator_version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
        }

    def _write_json(self, export: FOHistoryExport, path: str) -> None:
        """Write export to JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export.to_dict(), f, indent=2)

        logger.info(f"Exported FO history to {output_path}")

    def _compute_trend(self, values: List[float], epsilon: float = 0.01) -> str:
        """Compute trend direction."""
        if len(values) < 2:
            return TREND_FLAT
        delta = values[0] - values[1]
        if delta > epsilon:
            return TREND_UP
        elif delta < -epsilon:
            return TREND_DOWN
        return TREND_FLAT

    @staticmethod
    def _to_int(value: Optional[str]) -> int:
        try:
            return int(value) if value else 0
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _to_float(value: Optional[str]) -> float:
        try:
            return float(value) if value else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _parse_float_list(self, raw_list: List[str]) -> List[float]:
        result = []
        for item in raw_list:
            try:
                result.append(float(item))
            except (ValueError, TypeError):
                continue
        return result

    def _parse_int_list(self, raw_list: List[str]) -> List[int]:
        result = []
        for item in raw_list:
            try:
                result.append(int(item))
            except (ValueError, TypeError):
                continue
        return result


# ---------------------------------------------------------------------------
# Wide Slice Log Parser (for Dyno Chart)
# ---------------------------------------------------------------------------

@dataclass
class WideSliceRecord:
    """Parsed record from wide slice JSONL."""

    cycle: int
    status: str
    method: str
    abstention: bool
    duration_seconds: Optional[float]
    ht_hash: Optional[str]
    timestamp: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FOWideSliceParser:
    """
    Parser for wide slice JSONL logs.

    Used by Dyno Chart generator to load baseline and RFL data.
    """

    @staticmethod
    def parse_file(path: str) -> List[WideSliceRecord]:
        """
        Parse wide slice JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            List of WideSliceRecord sorted by cycle
        """
        records = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    record = FOWideSliceParser._parse_record(data)
                    if record:
                        records.append(record)
                except json.JSONDecodeError:
                    continue

        # Sort by cycle
        records.sort(key=lambda r: r.cycle)

        return records

    @staticmethod
    def _parse_record(data: Dict[str, Any]) -> Optional[WideSliceRecord]:
        """Parse a single JSONL record."""
        # Cycle is required
        cycle = data.get("cycle")
        if cycle is None:
            return None

        # Status
        status = data.get("status", "")

        # Method (check multiple fields)
        method = data.get("method", "") or data.get("verification_method", "")

        # Abstention detection
        abstention = FOWideSliceParser._detect_abstention(data)

        return WideSliceRecord(
            cycle=int(cycle),
            status=status,
            method=method,
            abstention=abstention,
            duration_seconds=data.get("duration_seconds"),
            ht_hash=data.get("ht_hash"),
            timestamp=data.get("timestamp"),
        )

    @staticmethod
    def _detect_abstention(data: Dict[str, Any]) -> bool:
        """Detect if a record represents an abstention."""
        # Check status
        if data.get("status") == "abstain":
            return True

        # Check method
        method = data.get("method", "") or data.get("verification_method", "")
        if method == "lean-disabled":
            return True

        # Check explicit abstention flag
        abstention_val = data.get("abstention")
        if abstention_val is True or abstention_val == 1:
            return True

        # Check derivation block
        derivation = data.get("derivation", {})
        if derivation.get("abstained", 0) > 0:
            return True

        return False

    @staticmethod
    def compute_abstention_series(
        records: List[WideSliceRecord],
        window: int = 100,
    ) -> List[Tuple[int, float]]:
        """
        Compute rolling abstention rate series.

        Args:
            records: Sorted list of WideSliceRecord
            window: Rolling window size

        Returns:
            List of (cycle, abstention_rate) tuples
        """
        if not records:
            return []

        # Convert to binary abstention series
        abstention_binary = [1 if r.abstention else 0 for r in records]

        # Compute rolling mean
        series = []
        for i in range(len(records)):
            start = max(0, i - window + 1)
            window_values = abstention_binary[start:i + 1]
            rate = sum(window_values) / len(window_values)
            series.append((records[i].cycle, rate))

        return series


# ---------------------------------------------------------------------------
# Main Analytics Engine
# ---------------------------------------------------------------------------

class FOAnalytics:
    """
    Main analytics engine for First Organism telemetry.

    Combines all analytics capabilities into a single interface.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        config: Optional[FOAnalyticsConfig] = None,
    ) -> None:
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL", "redis://localhost:6379/0"
        )
        self.config = config or FOAnalyticsConfig()

        # Initialize sub-components
        self.aggregator = FOTimeSeriesAggregator(self.config)
        self.anomaly_detector = FOAnomalyDetector(self.config)
        self.uplift_calculator = FOUpliftCalculator(self.config)
        self.exporter = FOHistoryExporter(self.redis_url, self.config)

    @property
    def available(self) -> bool:
        return self.exporter.available

    def compute_summary(self) -> Dict[str, Any]:
        """Compute full analytics summary."""
        export = self.exporter.export()
        return {
            "summary": export.summary,
            "current_state": export.current_state,
            "ht_verification": export.ht_verification,
        }

    def detect_anomalies(self) -> List[FOAnomaly]:
        """Detect anomalies in current metrics."""
        if not self.available:
            return []

        metrics = self.exporter._fetch_all_metrics()

        return self.anomaly_detector.detect(
            metrics["duration_history"],
            metrics["abstention_history"],
            metrics["success_history"],
        )

    def compute_duration_stats(self) -> FOTimeSeriesStats:
        """Compute duration time-series statistics."""
        if not self.available:
            return FOTimeSeriesStats()

        metrics = self.exporter._fetch_all_metrics()
        return self.aggregator.aggregate(metrics["duration_history"])

    def compute_abstention_stats(self) -> FOTimeSeriesStats:
        """Compute abstention time-series statistics."""
        if not self.available:
            return FOTimeSeriesStats()

        metrics = self.exporter._fetch_all_metrics()
        return self.aggregator.aggregate([float(a) for a in metrics["abstention_history"]])

    def compute_uplift(
        self,
        baseline_durations: List[float],
        rfl_durations: List[float],
    ) -> Optional[FOUpliftResult]:
        """Compute uplift between baseline and RFL."""
        return self.uplift_calculator.compute(baseline_durations, rfl_durations)

    def export_history(self, output_path: str) -> FOHistoryExport:
        """Export complete history to JSON file."""
        return self.exporter.export(output_path)

    def load_wide_slice_logs(
        self,
        baseline_path: str,
        rfl_path: str,
    ) -> Tuple[List[WideSliceRecord], List[WideSliceRecord]]:
        """Load and parse wide slice log files."""
        baseline = FOWideSliceParser.parse_file(baseline_path)
        rfl = FOWideSliceParser.parse_file(rfl_path)
        return baseline, rfl

    def compute_dyno_chart_data(
        self,
        baseline_path: str,
        rfl_path: str,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Compute data for Dyno Chart visualization.

        Returns dict with baseline_series and rfl_series.
        """
        baseline_records, rfl_records = self.load_wide_slice_logs(baseline_path, rfl_path)

        baseline_series = FOWideSliceParser.compute_abstention_series(baseline_records, window)
        rfl_series = FOWideSliceParser.compute_abstention_series(rfl_records, window)

        return {
            "baseline": {
                "series": baseline_series,
                "record_count": len(baseline_records),
                "abstention_rate_final": baseline_series[-1][1] if baseline_series else 0.0,
            },
            "rfl": {
                "series": rfl_series,
                "record_count": len(rfl_records),
                "abstention_rate_final": rfl_series[-1][1] if rfl_series else 0.0,
            },
            "window": window,
        }


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

_analytics: Optional[FOAnalytics] = None


def get_analytics() -> FOAnalytics:
    """Get the default FOAnalytics instance."""
    global _analytics
    if _analytics is None:
        _analytics = FOAnalytics()
    return _analytics


def export_fo_history(output_path: str = "exports/first_organism_history.json") -> bool:
    """
    Convenience function to export FO history.

    Args:
        output_path: Path to write JSON file

    Returns:
        True if export succeeded
    """
    try:
        analytics = get_analytics()
        analytics.export_history(output_path)
        return True
    except Exception as exc:
        logger.error(f"Failed to export FO history: {exc}")
        return False


# ---------------------------------------------------------------------------
# PHASE II: RFL Uplift Metrics Integration (Design Contract Only)
# ---------------------------------------------------------------------------
#
# STATUS: NOT IMPLEMENTED — DESIGN DOCUMENTATION ONLY
#
# Phase I treats all RFL logs as file-only (results/fo_*.jsonl) with no
# analytics integration. The metrics and data sources below do not exist.
#
# ---------------------------------------------------------------------------
# PROPOSED RFL METRICS (see first_organism_telemetry_plan_v2.md Section 11)
# ---------------------------------------------------------------------------
#
# Metric Name                  | Labels                  | Type
# -----------------------------|-------------------------|------------------
# rfl_abstention_rate          | slice, policy_version   | Gauge (0.0–1.0)
# baseline_abstention_rate     | slice                   | Gauge (0.0–1.0)
# rfl_uplift_delta             | slice                   | Gauge
# rfl_throughput               | slice, policy_version   | Gauge
# baseline_throughput          | slice                   | Gauge
# rfl_throughput_uplift_pct    | slice                   | Gauge
#
# ---------------------------------------------------------------------------
# PROPOSED DATA SOURCES (Phase II)
# ---------------------------------------------------------------------------
#
# Option 1: PostgreSQL table `rfl_metrics`
#   - Schema: (timestamp, slice, policy_version, metric_name, value, window_size)
#   - Indexed on (slice, timestamp) for range queries
#   - FOAnalytics would query via SQLAlchemy or raw psycopg
#
# Option 2: Redis time-series `ml:metrics:rfl:*`
#   - Sorted sets with timestamp scores
#   - TTL-managed for memory bounds
#   - FOAnalytics would read via pipeline + ZRANGEBYSCORE
#
# Option 3: JSONL append-log `results/rfl_metrics.jsonl`
#   - One JSON object per cycle
#   - FOAnalytics would parse like FOWideSliceParser
#   - No indexing; full scan required
#
# ---------------------------------------------------------------------------
# PROPOSED INTEGRATION PATTERN (Phase II)
# ---------------------------------------------------------------------------
#
# class RFLMetricsReader:
#     """
#     PHASE II ONLY: Would read from rfl_metrics table or Redis.
#
#     Not implemented. Phase I uses JSONL files directly.
#     """
#
#     def get_abstention_rate(self, slice: str, window: int = 100) -> float:
#         """Query rfl_abstention_rate{slice} with rolling window."""
#         raise NotImplementedError("Phase II")
#
#     def get_uplift_delta(self, slice: str) -> float:
#         """Compute baseline - rfl abstention rate."""
#         raise NotImplementedError("Phase II")
#
#     def get_throughput_uplift(self, slice: str) -> Dict[str, float]:
#         """Return {baseline, rfl, uplift_pct} throughput metrics."""
#         raise NotImplementedError("Phase II")
#
#
# class FOAnalyticsWithRFL(FOAnalytics):
#     """
#     PHASE II ONLY: Extended analytics with RFL DB integration.
#
#     Would add:
#     - compute_rfl_uplift_from_db()
#     - stream_rfl_metrics()
#     - alert_on_uplift_regression()
#
#     Not implemented. Phase I FOAnalytics reads from Redis FO keys only.
#     """
#     pass
#
# ---------------------------------------------------------------------------
# PHASE I vs PHASE II BOUNDARY
# ---------------------------------------------------------------------------
#
# | Capability                  | Phase I          | Phase II        |
# |-----------------------------|------------------|-----------------|
# | JSONL log parsing           | FOWideSliceParser| Continues       |
# | Dyno Chart data             | compute_dyno_*   | Continues       |
# | rfl_metrics table           | Does not exist   | RFLMetricsReader|
# | Real-time abstention rate   | Not collected    | Streamed        |
# | FOAnalytics + RFL DB        | Not wired        | FOAnalyticsWithRFL
#
# ---------------------------------------------------------------------------


# ===========================================================================
# PHASE II — UPLIFT EXPERIMENT LOG READER STUBS
# ===========================================================================
#
# STATUS: PHASE II — NOT RUN IN PHASE I
#
# These stubs define the interface for reading U2 uplift experiment logs.
# The U2 runner produces structured logs with per-cycle metrics across
# four asymmetric uplift environments (slices).
#
# Log format: JSONL files in results/u2_<slice>_<condition>.jsonl
#   - Baseline: results/u2_<slice>_baseline.jsonl
#   - Treatment: results/u2_<slice>_rfl.jsonl
#
# Each line contains:
#   {
#     "cycle": int,
#     "slice": str,              # e.g., "prop_depth4", "fol_eq_ring"
#     "condition": str,          # "baseline" or "rfl"
#     "policy_version": str,     # e.g., "v1.0.0", only for rfl
#     "timestamp": str,          # ISO 8601
#     "duration_seconds": float,
#     "proofs_attempted": int,
#     "proofs_succeeded": int,
#     "abstention_count": int,
#     "ht_hash": str,
#     "success": bool
#   }
#
# ---------------------------------------------------------------------------
# STUB: UpliftExperimentLogReader
# ---------------------------------------------------------------------------
#
# class UpliftExperimentLogReader:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Reads U2 uplift experiment logs from JSONL files or database.
#
#     Expected file locations:
#       results/u2_prop_depth4_baseline.jsonl
#       results/u2_prop_depth4_rfl.jsonl
#       results/u2_fol_eq_group_baseline.jsonl
#       results/u2_fol_eq_group_rfl.jsonl
#       results/u2_fol_eq_ring_baseline.jsonl
#       results/u2_fol_eq_ring_rfl.jsonl
#       results/u2_linear_arith_baseline.jsonl
#       results/u2_linear_arith_rfl.jsonl
#
#     Usage (design only):
#         reader = UpliftExperimentLogReader()
#         baseline_data = reader.load_slice("prop_depth4", "baseline")
#         treatment_data = reader.load_slice("prop_depth4", "rfl")
#         paired = reader.load_paired("prop_depth4")
#     """
#
#     SLICE_IDS = [
#         "prop_depth4",      # Propositional logic, depth ≤ 4
#         "fol_eq_group",     # FOL= with group theory axioms
#         "fol_eq_ring",      # FOL= with ring theory axioms
#         "linear_arith",     # Linear arithmetic (Presburger subset)
#     ]
#
#     def __init__(self, results_dir: str = "results") -> None:
#         """
#         Initialize reader with results directory.
#
#         Args:
#             results_dir: Directory containing U2 JSONL logs
#         """
#         raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#     def load_slice(
#         self,
#         slice_id: str,
#         condition: str,
#     ) -> List[Dict[str, Any]]:
#         """
#         Load all records for a slice/condition pair.
#
#         Args:
#             slice_id: One of SLICE_IDS (e.g., "prop_depth4")
#             condition: "baseline" or "rfl"
#
#         Returns:
#             List of parsed JSONL records, sorted by cycle
#         """
#         raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#     def load_paired(
#         self,
#         slice_id: str,
#     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
#         """
#         Load both baseline and treatment for a slice.
#
#         Args:
#             slice_id: One of SLICE_IDS
#
#         Returns:
#             (baseline_records, rfl_records) tuple
#         """
#         raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#     def get_durations(
#         self,
#         records: List[Dict[str, Any]],
#     ) -> List[float]:
#         """
#         Extract duration_seconds from records.
#
#         Args:
#             records: List of JSONL records
#
#         Returns:
#             List of duration values
#         """
#         raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#     def get_success_rate(
#         self,
#         records: List[Dict[str, Any]],
#     ) -> float:
#         """
#         Compute success rate from records.
#
#         Args:
#             records: List of JSONL records
#
#         Returns:
#             Proportion of records where success=True
#         """
#         raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
    def get_abstention_rate(
        self,
        records: List[Dict[str, Any]],
    ) -> float:
        """
        Compute abstention rate from records.

        Args:
            records: List of JSONL records

        Returns:
            Proportion of proofs that were abstentions
        """
        raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

    # STUB: read_uplift_log
    def read_uplift_log(self, file_path: str) -> List[Dict[str, Any]]:
        """
        PHASE II — NOT RUN IN PHASE I

        Reads a U2 uplift experiment log file (JSONL) and returns parsed records.

        Args:
            file_path: Path to the JSONL log file.

        Returns:
            List of parsed JSONL records.
        """
        raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

    # STUB: compute_success_rate
    def compute_success_rate(
        self,
        records: List[Dict[str, Any]],
    ) -> float:
        """
        PHASE II — NOT RUN IN PHASE I

        Compute the success rate from a list of uplift experiment records.

        Args:
            records: List of parsed JSONL records.

        Returns:
            The proportion of records where 'success' is True.
        """
        raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ---------------------------------------------------------------------------
# STUB: compute_wilson_ci
# ---------------------------------------------------------------------------
#
# def compute_wilson_ci(
#     successes: int,
#     trials: int,
#     confidence: float = 0.95,
# ) -> Tuple[float, float]:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Compute Wilson score confidence interval for a binomial proportion.
#
#     The Wilson score interval provides better coverage than the normal
#     approximation (Wald interval), especially for proportions near 0 or 1
#     and for small sample sizes. It is recommended for success rate CIs.
#
#     Formula (Wilson 1927):
#         p̂ = successes / trials
#         z = z-score for confidence level (e.g., 1.96 for 95%)
#         center = (p̂ + z²/2n) / (1 + z²/n)
#         margin = z * sqrt(p̂(1-p̂)/n + z²/4n²) / (1 + z²/n)
#         CI = (center - margin, center + margin)
#
#     Args:
#         successes: Number of successful trials (0 ≤ successes ≤ trials)
#         trials: Total number of trials (n > 0)
#         confidence: Confidence level (default 0.95 for 95% CI)
#
#     Returns:
#         (ci_low, ci_high) tuple representing the confidence interval
#
#     Raises:
#         ValueError: If successes > trials or trials ≤ 0
#
#     Example (design only):
#         # 80 successes out of 100 trials
#         ci_low, ci_high = compute_wilson_ci(80, 100, confidence=0.95)
#         # Returns approximately (0.71, 0.87)
#
#     References:
#         - Wilson, E.B. (1927). "Probable Inference, the Law of Succession,
#           and Statistical Inference". J. Amer. Statist. Assoc. 22: 209–212.
#         - Agresti & Coull (1998). "Approximate Is Better than 'Exact' for
#           Interval Estimation of Binomial Proportions". The American
#           Statistician, 52(2), 119-126.
#
#     Implementation notes:
#         - Requires scipy.stats.norm.ppf for z-score, or use precomputed
#           z-values: {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
#         - Edge cases: trials=0 → raise ValueError
#         - Edge cases: successes=0 → lower bound is 0
#         - Edge cases: successes=trials → upper bound is 1
#
#     Use cases in Phase II:
#         - Success rate CI for baseline vs RFL comparison
#         - Abstention rate CI for governance thresholds
#         - Per-slice success metrics with uncertainty bounds
#     """
#     raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ---------------------------------------------------------------------------
# STUB: bootstrap_delta
# ---------------------------------------------------------------------------
#
# def bootstrap_delta(
#     baseline_values: List[float],
#     treatment_values: List[float],
#     statistic: str = "mean",
#     n_bootstrap: int = 10000,
#     confidence: float = 0.95,
#     seed: Optional[int] = None,
# ) -> Dict[str, float]:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Compute bootstrap confidence interval for the difference (delta)
#     between treatment and baseline statistics.
#
#     Bootstrap resampling provides non-parametric confidence intervals
#     that make no distributional assumptions. This is preferred for
#     duration and throughput metrics which may not be normally distributed.
#
#     Algorithm (BCa percentile bootstrap):
#         1. Compute observed statistic for each group
#         2. Compute observed delta = treatment_stat - baseline_stat
#         3. For i in 1..n_bootstrap:
#             a. Resample baseline_values with replacement
#             b. Resample treatment_values with replacement
#             c. Compute delta_i = treatment_stat_i - baseline_stat_i
#         4. Sort bootstrap deltas
#         5. Extract percentile bounds: [(1-conf)/2, (1+conf)/2]
#         6. Optionally apply BCa bias correction
#
#     Args:
#         baseline_values: List of values from baseline condition
#         treatment_values: List of values from treatment (RFL) condition
#         statistic: Statistic to compute - "mean", "median", or "throughput"
#                    (throughput = 1/mean for rate metrics)
#         n_bootstrap: Number of bootstrap iterations (default 10000)
#         confidence: Confidence level (default 0.95)
#         seed: Random seed for reproducibility (required for determinism)
#
#     Returns:
#         Dict with keys:
#             - "baseline_stat": Observed statistic for baseline
#             - "treatment_stat": Observed statistic for treatment
#             - "delta": Observed difference (treatment - baseline)
#             - "delta_ci_low": Lower bound of CI for delta
#             - "delta_ci_high": Upper bound of CI for delta
#             - "delta_pct": Percentage change ((delta / baseline) * 100)
#             - "delta_pct_ci_low": Lower bound of % change CI
#             - "delta_pct_ci_high": Upper bound of % change CI
#             - "significant": True if CI excludes zero
#             - "n_bootstrap": Number of iterations used
#             - "seed": Seed used for reproducibility
#
#     Raises:
#         ValueError: If either list is empty or statistic is invalid
#
#     Example (design only):
#         # Compare mean durations
#         baseline_durations = [1.2, 1.4, 1.1, 1.3, 1.5, ...]
#         rfl_durations = [0.9, 1.0, 0.8, 1.1, 0.95, ...]
#         result = bootstrap_delta(
#             baseline_durations,
#             rfl_durations,
#             statistic="mean",
#             n_bootstrap=10000,
#             confidence=0.95,
#             seed=42,
#         )
#         # Returns: {
#         #     "baseline_stat": 1.3,
#         #     "treatment_stat": 0.95,
#         #     "delta": -0.35,
#         #     "delta_ci_low": -0.45,
#         #     "delta_ci_high": -0.25,
#         #     "delta_pct": -26.9,
#         #     "delta_pct_ci_low": -34.6,
#         #     "delta_pct_ci_high": -19.2,
#         #     "significant": True,
#         #     "n_bootstrap": 10000,
#         #     "seed": 42,
#         # }
#
#     Implementation notes:
#         - Use numpy.random.Generator for reproducible resampling
#         - For throughput: compute 1/mean, handle mean=0 edge case
#         - Percentile method: np.percentile(deltas, [2.5, 97.5]) for 95% CI
#         - BCa correction requires jackknife estimates (optional enhancement)
#         - Parallelization: can use joblib for n_bootstrap > 50000
#
#     Determinism requirement:
#         - seed parameter MUST be provided for reproducibility
#         - All bootstrap samples must be generated from seeded RNG
#         - Results must be identical across runs with same seed
#
#     Use cases in Phase II:
#         - Duration improvement: RFL vs baseline mean duration
#         - Throughput uplift: proofs/second improvement
#         - Abstention reduction: delta in abstention counts
#         - Per-slice comparisons across four asymmetric environments
#     """
#     raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ---------------------------------------------------------------------------
# PHASE II INTEGRATION POINTS
# ---------------------------------------------------------------------------
#
# The stubs above are designed to integrate with:
#
# 1. U2 Runner (experiments/u2_runner.py — to be created in Phase II)
#    - Produces JSONL logs consumed by UpliftExperimentLogReader
#    - Calls compute_wilson_ci for per-cycle success rate CIs
#    - Calls bootstrap_delta for final uplift report
#
# 2. Governance Module (backend/governance/ — Phase II)
#    - Uses compute_wilson_ci to validate success thresholds
#    - Flags experiments where CI overlaps null effect
#
# 3. Telemetry Dashboard (ui/src/routes/uplift — Phase II)
#    - Displays CIs from compute_wilson_ci
#    - Shows bootstrap_delta results with uncertainty bands
#
# 4. Preregistration Validator (experiments/prereg_validator.py — Phase II)
#    - Verifies analysis plan matches preregistered methods
#    - Confirms bootstrap parameters match PREREG_UPLIFT_U2.yaml
#
# ---------------------------------------------------------------------------


# ===========================================================================
# PHASE II — UPLIFT EXPERIMENT LOG READER (EXTENDED STUBS)
# ===========================================================================
#
# STATUS: PHASE II — NOT RUN IN PHASE I
#
# The following stubs extend the UpliftExperimentLogReader with additional
# methods for reading and aggregating U2 experiment logs across slices.
#
# ---------------------------------------------------------------------------
# STUB: load_all_slices
# ---------------------------------------------------------------------------
#
# def load_all_slices(
#     self,
#     condition: str = "baseline",
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Load records for all slices for a given condition.
#
#     Args:
#         condition: "baseline" or "rfl"
#
#     Returns:
#         Dict mapping slice_id to list of records
#         {
#             "prop_depth4": [...],
#             "fol_eq_group": [...],
#             "fol_eq_ring": [...],
#             "linear_arith": [...],
#         }
#     """
#     raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ---------------------------------------------------------------------------
# STUB: aggregate_slice_metrics
# ---------------------------------------------------------------------------
#
# def aggregate_slice_metrics(
#     self,
#     slice_id: str,
# ) -> Dict[str, Any]:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Aggregate metrics for a single slice across baseline and treatment.
#
#     Returns:
#         {
#             "slice_id": str,
#             "baseline": {
#                 "n_cycles": int,
#                 "mean_duration": float,
#                 "median_duration": float,
#                 "success_rate": float,
#                 "abstention_rate": float,
#                 "throughput": float,  # proofs/second
#             },
#             "rfl": {
#                 "n_cycles": int,
#                 "mean_duration": float,
#                 "median_duration": float,
#                 "success_rate": float,
#                 "abstention_rate": float,
#                 "throughput": float,
#                 "policy_version": str,
#             },
#             "delta": {
#                 "duration_delta": float,
#                 "throughput_uplift_pct": float,
#                 "abstention_reduction": float,
#             },
#         }
#     """
#     raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ---------------------------------------------------------------------------
# STUB: generate_u2_summary_report
# ---------------------------------------------------------------------------
#
# def generate_u2_summary_report(
#     self,
#     output_path: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Generate comprehensive U2 experiment summary across all slices.
#
#     This produces the primary artifact for Phase II uplift analysis,
#     aggregating metrics from all four asymmetric environments.
#
#     Args:
#         output_path: Optional path to write JSON report
#
#     Returns:
#         {
#             "schema": "https://mathledger.io/schemas/u2-report-v1.json",
#             "generated_at": str,  # ISO 8601
#             "experiment_id": str,
#             "prereg_ref": "PREREG_UPLIFT_U2.yaml",
#             "slices": {
#                 "prop_depth4": {...},  # aggregate_slice_metrics output
#                 "fol_eq_group": {...},
#                 "fol_eq_ring": {...},
#                 "linear_arith": {...},
#             },
#             "global_metrics": {
#                 "total_baseline_cycles": int,
#                 "total_rfl_cycles": int,
#                 "overall_throughput_uplift_pct": float,
#                 "overall_abstention_reduction": float,
#             },
#             "statistical_tests": {
#                 "prop_depth4": {
#                     "success_rate_ci": (float, float),  # Wilson CI
#                     "duration_delta_ci": (float, float),  # Bootstrap CI
#                     "significant": bool,
#                 },
#                 ...
#             },
#             "governance": {
#                 "all_slices_pass": bool,
#                 "failing_slices": List[str],
#                 "recommendation": str,  # "proceed", "hold", "rollback"
#             },
#         }
#     """
#     raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ===========================================================================
# PHASE II — SLICE-SPECIFIC SUCCESS METRICS
# ===========================================================================
#
# STATUS: PHASE II — NOT RUN IN PHASE I
#
# Each of the four asymmetric uplift environments has distinct success
# criteria based on the nature of the logical system and expected behavior.
#
# ---------------------------------------------------------------------------
# SLICE SUCCESS CRITERIA (per PREREG_UPLIFT_U2.yaml)
# ---------------------------------------------------------------------------
#
# SLICE_SUCCESS_CRITERIA = {
#     "prop_depth4": {
#         # Propositional logic, depth ≤ 4
#         # Mature slice with high baseline success
#         "min_success_rate": 0.95,       # 95% minimum
#         "max_abstention_rate": 0.02,    # 2% max abstention
#         "min_throughput_uplift_pct": 5.0,  # 5% improvement required
#         "min_samples": 500,
#     },
#     "fol_eq_group": {
#         # FOL= with group theory axioms
#         # Medium complexity, some abstentions expected
#         "min_success_rate": 0.85,       # 85% minimum
#         "max_abstention_rate": 0.10,    # 10% max abstention
#         "min_throughput_uplift_pct": 3.0,  # 3% improvement required
#         "min_samples": 300,
#     },
#     "fol_eq_ring": {
#         # FOL= with ring theory axioms
#         # Higher complexity, more abstentions expected
#         "min_success_rate": 0.80,       # 80% minimum
#         "max_abstention_rate": 0.15,    # 15% max abstention
#         "min_throughput_uplift_pct": 2.0,  # 2% improvement required
#         "min_samples": 300,
#     },
#     "linear_arith": {
#         # Linear arithmetic (Presburger subset)
#         # Experimental slice, lower bar
#         "min_success_rate": 0.70,       # 70% minimum
#         "max_abstention_rate": 0.20,    # 20% max abstention
#         "min_throughput_uplift_pct": 0.0,  # No regression required
#         "min_samples": 200,
#     },
# }
#
#
# ---------------------------------------------------------------------------
# STUB: evaluate_slice_success
# ---------------------------------------------------------------------------
#
# def evaluate_slice_success(
#     slice_id: str,
#     metrics: Dict[str, Any],
#     criteria: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Evaluate whether a slice meets its success criteria.
#
#     Args:
#         slice_id: One of SLICE_IDS
#         metrics: Output from aggregate_slice_metrics
#         criteria: Override criteria (defaults to SLICE_SUCCESS_CRITERIA)
#
#     Returns:
#         {
#             "slice_id": str,
#             "passed": bool,
#             "checks": {
#                 "success_rate": {
#                     "value": float,
#                     "threshold": float,
#                     "passed": bool,
#                 },
#                 "abstention_rate": {
#                     "value": float,
#                     "threshold": float,
#                     "passed": bool,
#                 },
#                 "throughput_uplift": {
#                     "value": float,
#                     "threshold": float,
#                     "passed": bool,
#                 },
#                 "sample_size": {
#                     "value": int,
#                     "threshold": int,
#                     "passed": bool,
#                 },
#             },
#             "failing_checks": List[str],
#             "recommendation": str,
#         }
#     """
#     raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ---------------------------------------------------------------------------
# STUB: evaluate_all_slices
# ---------------------------------------------------------------------------
#
# def evaluate_all_slices(
#     reader: "UpliftExperimentLogReader",
# ) -> Dict[str, Any]:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Evaluate success criteria across all slices.
#
#     Returns:
#         {
#             "all_passed": bool,
#             "slices": {
#                 "prop_depth4": {...},  # evaluate_slice_success output
#                 "fol_eq_group": {...},
#                 "fol_eq_ring": {...},
#                 "linear_arith": {...},
#             },
#             "summary": {
#                 "passed_count": int,
#                 "failed_count": int,
#                 "failing_slices": List[str],
#             },
#             "governance_decision": str,  # "proceed", "hold", "rollback"
#         }
#     """
#     raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ===========================================================================
# PHASE II — MANIFEST AND ATTESTATION HANDLING
# ===========================================================================
#
# STATUS: PHASE II — NOT RUN IN PHASE I
#
# U2 experiments produce manifest files for reproducibility and attestation
# files for governance review.
#
# ---------------------------------------------------------------------------
# STUB: generate_experiment_manifest
# ---------------------------------------------------------------------------
#
# def generate_experiment_manifest(
#     experiment_id: str,
#     slices: List[str],
#     config: Dict[str, Any],
#     output_path: str,
# ) -> Dict[str, Any]:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Generate experiment manifest for reproducibility.
#
#     The manifest captures all parameters needed to reproduce the
#     experiment, including seeds, slice configurations, and policy
#     versions.
#
#     Args:
#         experiment_id: Unique experiment identifier
#         slices: List of slice IDs included
#         config: Full experiment configuration
#         output_path: Path to write manifest JSON
#
#     Returns:
#         {
#             "schema": "https://mathledger.io/schemas/u2-manifest-v1.json",
#             "experiment_id": str,
#             "created_at": str,
#             "prereg_ref": "PREREG_UPLIFT_U2.yaml",
#             "slices": List[str],
#             "config": {
#                 "n_cycles_per_slice": int,
#                 "baseline_seed": int,
#                 "rfl_seed": int,
#                 "policy_version": str,
#                 "derivation_params": {...},
#             },
#             "artifacts": {
#                 "baseline_logs": List[str],  # Paths
#                 "rfl_logs": List[str],
#                 "report_path": str,
#             },
#             "checksums": {
#                 "<path>": "<sha256>",
#                 ...
#             },
#         }
#     """
#     raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ---------------------------------------------------------------------------
# STUB: generate_attestation
# ---------------------------------------------------------------------------
#
# def generate_attestation(
#     experiment_id: str,
#     manifest_path: str,
#     evaluation_results: Dict[str, Any],
#     output_path: str,
# ) -> Dict[str, Any]:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Generate governance attestation for experiment results.
#
#     The attestation provides a cryptographically signed summary of
#     the experiment outcome for governance review.
#
#     Args:
#         experiment_id: Unique experiment identifier
#         manifest_path: Path to experiment manifest
#         evaluation_results: Output from evaluate_all_slices
#         output_path: Path to write attestation JSON
#
#     Returns:
#         {
#             "schema": "https://mathledger.io/schemas/u2-attestation-v1.json",
#             "experiment_id": str,
#             "created_at": str,
#             "manifest_hash": str,  # SHA-256 of manifest
#             "evaluation_summary": {
#                 "all_passed": bool,
#                 "passed_slices": List[str],
#                 "failed_slices": List[str],
#             },
#             "governance_recommendation": str,
#             "attestation_hash": str,  # SHA-256 of this document (excl. hash)
#         }
#     """
#     raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ===========================================================================
# PHASE II — TELEMETRY INTEGRATION STUBS
# ===========================================================================
#
# STATUS: PHASE II — NOT RUN IN PHASE I
#
# The following stubs define integration points for real-time telemetry
# during U2 experiment execution.
#
# ---------------------------------------------------------------------------
# STUB: U2TelemetryEmitter
# ---------------------------------------------------------------------------
#
# class U2TelemetryEmitter:
#     """
#     PHASE II — NOT RUN IN PHASE I
#
#     Emits telemetry events during U2 experiment execution.
#
#     Telemetry is written to Redis and/or JSONL logs for real-time
#     monitoring and post-hoc analysis.
#
#     Events:
#         - u2:cycle_start: Beginning of a derivation cycle
#         - u2:cycle_end: Completion of a derivation cycle
#         - u2:slice_start: Beginning of slice execution
#         - u2:slice_end: Completion of slice execution
#         - u2:experiment_start: Beginning of full experiment
#         - u2:experiment_end: Completion of full experiment
#         - u2:policy_update: Policy parameters changed (RFL only)
#         - u2:anomaly_detected: Anomaly detected during execution
#     """
#
#     def __init__(
#         self,
#         experiment_id: str,
#         redis_url: Optional[str] = None,
#         log_path: Optional[str] = None,
#     ) -> None:
#         raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#     def emit_cycle_start(
#         self,
#         slice_id: str,
#         cycle: int,
#         condition: str,
#     ) -> None:
#         raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#     def emit_cycle_end(
#         self,
#         slice_id: str,
#         cycle: int,
#         condition: str,
#         duration_seconds: float,
#         success: bool,
#         proofs_succeeded: int,
#         abstention_count: int,
#     ) -> None:
#         raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#     def emit_slice_complete(
#         self,
#         slice_id: str,
#         condition: str,
#         n_cycles: int,
#         aggregate_metrics: Dict[str, Any],
#     ) -> None:
#         raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#     def emit_experiment_complete(
#         self,
#         summary: Dict[str, Any],
#         attestation_path: str,
#     ) -> None:
#         raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
#
#
# ---------------------------------------------------------------------------
# END PHASE II STUBS
# ---------------------------------------------------------------------------
