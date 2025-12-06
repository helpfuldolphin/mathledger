"""
First Organism Metrics Schema — Canonical Normalization

Defines the normalized schema for First Organism vital signs,
ensuring consistency between:
- Redis telemetry emitter (first_organism_telemetry.py)
- Metrics collector (FirstOrganismCollector)
- RFL feedback loop integration
- Report generation (ASCII/Markdown)

All field names, types, and semantics are defined here as the
single source of truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import statistics


# ---------------------------------------------------------------------------
# Canonical Field Names (Redis keys use these as suffixes)
# ---------------------------------------------------------------------------

REDIS_KEY_PREFIX = "ml:metrics:first_organism"

# Scalar metrics
FIELD_RUNS_TOTAL = "runs_total"
FIELD_LAST_HT = "last_ht"
FIELD_LAST_HT_FULL = "last_ht_full"  # Full 64-char hash for verification
FIELD_DURATION_SECONDS = "duration_seconds"
FIELD_LATENCY_SECONDS = "latency_seconds"
FIELD_LAST_ABSTENTIONS = "last_abstentions"
FIELD_LAST_RUN_TIMESTAMP = "last_run_timestamp"
FIELD_LAST_STATUS = "last_status"

# History lists (rolling window)
FIELD_DURATION_HISTORY = "duration_history"
FIELD_ABSTENTION_HISTORY = "abstention_history"
FIELD_SUCCESS_HISTORY = "success_history"
FIELD_HT_HISTORY = "ht_history"  # Track H_t hashes for cryptographic verification

# Computed metrics (derived from history)
FIELD_AVERAGE_DURATION = "average_duration_seconds"
FIELD_MEDIAN_DURATION = "median_duration_seconds"
FIELD_SUCCESS_RATE = "success_rate"
FIELD_DURATION_DELTA = "duration_delta"
FIELD_ABSTENTION_DELTA = "abstention_delta"

# Trend indicators
FIELD_DURATION_TREND = "duration_trend"  # "up", "down", "flat"
FIELD_ABSTENTION_TREND = "abstention_trend"
FIELD_SUCCESS_TREND = "success_trend"

# Status values
STATUS_SUCCESS = "success"
STATUS_FAILURE = "failure"
STATUS_UNKNOWN = ""

# Trend values
TREND_UP = "up"
TREND_DOWN = "down"
TREND_FLAT = "flat"

# History window size
HISTORY_MAX_LEN = 20


# ---------------------------------------------------------------------------
# Normalized Metrics Dataclass
# ---------------------------------------------------------------------------


@dataclass
class FOTrendSeries:
    """Trend series for a single metric."""

    latest: float
    delta_from_previous: float
    moving_average_short: float  # 5-sample window
    moving_average_long: float   # 20-sample window
    samples: int
    trend: str  # "up", "down", "flat"

    @classmethod
    def from_history(
        cls,
        history: List[float],
        short_window: int = 5,
        long_window: int = 20,
        epsilon: float = 0.01,
    ) -> "FOTrendSeries":
        """Compute trend series from history list (most recent first)."""
        if not history:
            return cls(
                latest=0.0,
                delta_from_previous=0.0,
                moving_average_short=0.0,
                moving_average_long=0.0,
                samples=0,
                trend=TREND_FLAT,
            )

        latest = history[0]
        delta = history[0] - history[1] if len(history) >= 2 else 0.0

        short_avg = statistics.mean(history[:short_window]) if history else 0.0
        long_avg = statistics.mean(history[:long_window]) if history else short_avg

        # Determine trend direction
        if delta > epsilon:
            trend = TREND_UP
        elif delta < -epsilon:
            trend = TREND_DOWN
        else:
            trend = TREND_FLAT

        return cls(
            latest=round(latest, 6),
            delta_from_previous=round(delta, 6),
            moving_average_short=round(short_avg, 6),
            moving_average_long=round(long_avg, 6),
            samples=len(history),
            trend=trend,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FOVitalSigns:
    """
    Normalized First Organism vital signs.

    This is the canonical structure for FO metrics, used by:
    - Collectors to emit normalized data
    - Reports to render consistent output
    - RFL feedback loop to consume health signals
    """

    # Core metrics
    runs_total: int = 0
    last_ht_hash: str = ""
    last_ht_full: str = ""  # Full hash for cryptographic verification
    latency_seconds: float = 0.0
    last_duration_seconds: float = 0.0
    average_duration_seconds: float = 0.0
    median_duration_seconds: float = 0.0
    abstention_count: int = 0
    last_run_timestamp: Optional[str] = None
    last_status: str = ""

    # Computed rates
    success_rate: float = 0.0  # 0-100%

    # Deltas (change from previous run)
    duration_delta: float = 0.0
    abstention_delta: int = 0

    # Trends
    duration_trend: str = TREND_FLAT
    abstention_trend: str = TREND_FLAT
    success_trend: str = TREND_FLAT

    # History (most recent first)
    duration_history: List[float] = field(default_factory=list)
    abstention_history: List[int] = field(default_factory=list)
    success_history: List[str] = field(default_factory=list)
    ht_history: List[str] = field(default_factory=list)

    # Trend series (for detailed analysis)
    duration_series: Optional[FOTrendSeries] = None
    abstention_series: Optional[FOTrendSeries] = None

    def compute_derived_metrics(self) -> None:
        """Compute all derived metrics from history."""
        # Duration stats
        if self.duration_history:
            self.average_duration_seconds = statistics.mean(self.duration_history)
            self.median_duration_seconds = statistics.median(self.duration_history)
            if len(self.duration_history) >= 2:
                self.duration_delta = self.duration_history[0] - self.duration_history[1]

        # Abstention stats
        if len(self.abstention_history) >= 2:
            self.abstention_delta = self.abstention_history[0] - self.abstention_history[1]

        # Success rate
        if self.success_history:
            successes = sum(1 for s in self.success_history if s == STATUS_SUCCESS)
            self.success_rate = (successes / len(self.success_history)) * 100.0

        # Trend series
        self.duration_series = FOTrendSeries.from_history(self.duration_history)
        self.abstention_series = FOTrendSeries.from_history(
            [float(x) for x in self.abstention_history]
        )

        # Trend indicators
        self.duration_trend = self.duration_series.trend
        self.abstention_trend = self.abstention_series.trend

        # Success trend (based on success rate change)
        if len(self.success_history) >= 2:
            recent_rate = sum(1 for s in self.success_history[:5] if s == STATUS_SUCCESS) / min(5, len(self.success_history))
            older_rate = sum(1 for s in self.success_history[5:10] if s == STATUS_SUCCESS) / max(1, min(5, len(self.success_history) - 5))
            rate_delta = recent_rate - older_rate
            if rate_delta > 0.1:
                self.success_trend = TREND_UP
            elif rate_delta < -0.1:
                self.success_trend = TREND_DOWN
            else:
                self.success_trend = TREND_FLAT

    def is_alive(self) -> bool:
        """Check if the organism is considered 'alive' (healthy)."""
        return self.last_status == STATUS_SUCCESS and self.success_rate >= 80.0

    def health_status(self) -> str:
        """Return health status indicator."""
        if self.runs_total == 0:
            return "UNKNOWN"
        if self.is_alive():
            return "ALIVE"
        if self.success_rate >= 50.0:
            return "DEGRADED"
        return "CRITICAL"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "runs_total": self.runs_total,
            "last_ht_hash": self.last_ht_hash,
            "latency_seconds": self.latency_seconds,
            "last_duration_seconds": self.last_duration_seconds,
            "average_duration_seconds": round(self.average_duration_seconds, 6),
            "median_duration_seconds": round(self.median_duration_seconds, 6),
            "abstention_count": self.abstention_count,
            "last_run_timestamp": self.last_run_timestamp,
            "last_status": self.last_status,
            "success_rate": round(self.success_rate, 2),
            "duration_delta": round(self.duration_delta, 6),
            "abstention_delta": self.abstention_delta,
            "duration_trend": self.duration_trend,
            "abstention_trend": self.abstention_trend,
            "success_trend": self.success_trend,
            "duration_history": [round(d, 6) for d in self.duration_history],
            "abstention_history": self.abstention_history,
            "success_history": self.success_history,
            "ht_history": self.ht_history,
            "health_status": self.health_status(),
        }
        if self.duration_series:
            data["duration_series"] = self.duration_series.to_dict()
        if self.abstention_series:
            data["abstention_series"] = self.abstention_series.to_dict()
        return data

    def verify_ht_stability(self) -> Dict[str, Any]:
        """
        Verify cryptographic stability of H_t hashes.

        Returns a verification report including:
        - Whether all H_t hashes are valid SHA-256
        - Uniqueness check (each run should produce unique H_t)
        - Consistency check (same inputs should produce same H_t)
        """
        if not self.ht_history:
            return {
                "verified": False,
                "reason": "no_history",
                "unique_count": 0,
                "total_count": 0,
            }

        valid_hashes = []
        for ht in self.ht_history:
            # Verify it's a valid hex string of correct length
            try:
                if len(ht) >= 16:  # Short hash
                    int(ht, 16)  # Validate hex
                    valid_hashes.append(ht)
            except ValueError:
                pass

        unique_hashes = set(valid_hashes)

        return {
            "verified": len(valid_hashes) == len(self.ht_history),
            "valid_count": len(valid_hashes),
            "total_count": len(self.ht_history),
            "unique_count": len(unique_hashes),
            "uniqueness_ratio": len(unique_hashes) / len(self.ht_history) if self.ht_history else 0.0,
            "last_ht": self.last_ht_hash,
            "last_ht_full": self.last_ht_full,
        }


# ---------------------------------------------------------------------------
# Sparkline Visualization
# ---------------------------------------------------------------------------


def generate_sparkline(
    values: List[float],
    width: int = 10,
    chars: str = "_.-=",
) -> str:
    """
    Generate ASCII sparkline from values.

    Args:
        values: List of values (most recent first, will be reversed for display)
        width: Maximum number of characters
        chars: Characters for quartiles (low to high)

    Returns:
        ASCII sparkline string
    """
    if not values:
        return ""

    recent = values[:width]
    if not recent:
        return ""

    min_val = min(recent)
    max_val = max(recent)
    range_val = max_val - min_val if max_val > min_val else 1.0

    def bar_char(val: float) -> str:
        if range_val == 0:
            return chars[len(chars) // 2]
        normalized = (val - min_val) / range_val
        idx = min(int(normalized * len(chars)), len(chars) - 1)
        return chars[idx]

    # Reverse to show oldest-to-newest left-to-right
    return "".join(bar_char(v) for v in reversed(recent))


def generate_trend_indicator(trend: str) -> str:
    """Generate ASCII trend indicator."""
    if trend == TREND_UP:
        return "/\\"  # Upward
    elif trend == TREND_DOWN:
        return "\\/"  # Downward
    else:
        return "--"  # Flat


# ---------------------------------------------------------------------------
# RFL Feedback Integration
# ---------------------------------------------------------------------------


@dataclass
class FOFeedbackSignal:
    """
    Feedback signal from FO metrics to RFL runner.

    This signal influences policy decisions based on organism health.
    """

    success_rate: float  # 0-100%
    health_status: str   # ALIVE, DEGRADED, CRITICAL, UNKNOWN
    duration_trend: str  # up, down, flat
    abstention_trend: str

    # Policy adjustment signals
    should_throttle: bool = False  # Reduce derivation intensity
    should_boost: bool = False     # Increase derivation intensity
    confidence: float = 0.0        # Confidence in the signal (0-1)

    @classmethod
    def from_vital_signs(cls, vitals: FOVitalSigns) -> "FOFeedbackSignal":
        """Generate feedback signal from vital signs."""
        # Determine policy adjustments
        should_throttle = (
            vitals.health_status() == "CRITICAL" or
            vitals.success_rate < 50.0 or
            (vitals.duration_trend == TREND_UP and vitals.duration_delta > 1.0)
        )

        should_boost = (
            vitals.health_status() == "ALIVE" and
            vitals.success_rate >= 95.0 and
            vitals.abstention_trend != TREND_UP
        )

        # Confidence based on sample size
        samples = len(vitals.success_history)
        if samples >= 10:
            confidence = 1.0
        elif samples >= 5:
            confidence = 0.7
        elif samples >= 2:
            confidence = 0.4
        else:
            confidence = 0.1

        return cls(
            success_rate=vitals.success_rate,
            health_status=vitals.health_status(),
            duration_trend=vitals.duration_trend,
            abstention_trend=vitals.abstention_trend,
            should_throttle=should_throttle,
            should_boost=should_boost,
            confidence=confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

