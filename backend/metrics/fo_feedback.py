"""
First Organism → RFL Feedback Loop

Provides the integration between First Organism vital signs and the
RFL runner's policy adjustment mechanism.

The feedback loop:
1. Reads FO metrics from Redis (via FirstOrganismCollector)
2. Computes a feedback signal based on health status
3. Provides policy adjustment recommendations to RFL runner
4. Logs feedback decisions to the audit trail

This module is the bridge between Cursor K (Metrics Oracle) and
the RFL runner's policy adaptation system.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger("FOFeedback")


# ---------------------------------------------------------------------------
# Feedback Signal Dataclass
# ---------------------------------------------------------------------------


@dataclass
class FOFeedbackSignal:
    """
    Feedback signal from FO metrics to RFL runner.

    This signal influences policy decisions based on organism health.
    The RFL runner consumes this signal to adjust derivation intensity.
    """

    # Input metrics
    success_rate: float  # 0-100%
    health_status: str   # ALIVE, DEGRADED, CRITICAL, UNKNOWN
    duration_trend: str  # up, down, flat
    abstention_trend: str  # up, down, flat
    runs_total: int
    last_ht_hash: str

    # Policy adjustment signals
    should_throttle: bool = False  # Reduce derivation intensity
    should_boost: bool = False     # Increase derivation intensity
    confidence: float = 0.0        # Confidence in the signal (0-1)

    # Computed adjustments
    intensity_multiplier: float = 1.0  # Multiply derive_steps/max_breadth by this
    abstention_threshold_adjustment: float = 0.0  # Adjust abstention threshold

    # Audit trail
    timestamp: str = ""
    signal_hash: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.signal_hash:
            self.signal_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute deterministic hash of the signal for audit."""
        data = f"{self.success_rate}|{self.health_status}|{self.duration_trend}|{self.abstention_trend}|{self.runs_total}|{self.last_ht_hash}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feedback Loop Reader
# ---------------------------------------------------------------------------


class FOFeedbackReader:
    """
    Reads First Organism metrics and generates feedback signals.

    Usage:
        reader = FOFeedbackReader()
        signal = reader.get_feedback_signal()
        if signal.should_throttle:
            # Reduce derivation intensity
        elif signal.should_boost:
            # Increase derivation intensity
    """

    REDIS_KEY_PREFIX = "ml:metrics:first_organism"

    # Thresholds for policy decisions
    THROTTLE_SUCCESS_RATE = 50.0   # Below this, throttle
    BOOST_SUCCESS_RATE = 95.0      # Above this, consider boosting
    MIN_SAMPLES_FOR_CONFIDENCE = 5  # Minimum runs for high confidence

    # Intensity adjustments
    THROTTLE_MULTIPLIER = 0.5      # Reduce intensity by 50%
    BOOST_MULTIPLIER = 1.25        # Increase intensity by 25%

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL", "redis://localhost:6379/0"
        )
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
        return f"{self.REDIS_KEY_PREFIX}:{name}"

    def get_feedback_signal(self) -> Optional[FOFeedbackSignal]:
        """
        Read FO metrics and generate a feedback signal.

        Returns:
            FOFeedbackSignal if metrics available, None otherwise
        """
        if not self._client:
            logger.warning("Redis unavailable, cannot generate FO feedback signal")
            return None

        try:
            # Fetch metrics
            pipe = self._client.pipeline()
            pipe.get(self._key("runs_total"))
            pipe.get(self._key("last_ht"))
            pipe.get(self._key("last_status"))
            pipe.lrange(self._key("success_history"), 0, 19)
            pipe.lrange(self._key("duration_history"), 0, 19)
            pipe.lrange(self._key("abstention_history"), 0, 19)
            results = pipe.execute()

            runs_total = self._to_int(results[0])
            last_ht = results[1] or ""
            last_status = results[2] or ""
            success_history = results[3] or []
            duration_history = self._parse_float_list(results[4] or [])
            abstention_history = self._parse_int_list(results[5] or [])

            # Compute success rate
            if success_history:
                successes = sum(1 for s in success_history if s == "success")
                success_rate = (successes / len(success_history)) * 100.0
            else:
                success_rate = 0.0

            # Determine health status
            if runs_total == 0:
                health_status = "UNKNOWN"
            elif last_status == "success" and success_rate >= 80.0:
                health_status = "ALIVE"
            elif success_rate >= 50.0:
                health_status = "DEGRADED"
            else:
                health_status = "CRITICAL"

            # Compute trends
            duration_trend = self._compute_trend(duration_history)
            abstention_trend = self._compute_trend([float(x) for x in abstention_history])

            # Determine policy adjustments
            should_throttle = (
                health_status == "CRITICAL" or
                success_rate < self.THROTTLE_SUCCESS_RATE or
                (duration_trend == "up" and len(duration_history) >= 2 and duration_history[0] - duration_history[1] > 1.0)
            )

            should_boost = (
                health_status == "ALIVE" and
                success_rate >= self.BOOST_SUCCESS_RATE and
                abstention_trend != "up" and
                runs_total >= self.MIN_SAMPLES_FOR_CONFIDENCE
            )

            # Compute confidence
            if runs_total >= 10:
                confidence = 1.0
            elif runs_total >= 5:
                confidence = 0.7
            elif runs_total >= 2:
                confidence = 0.4
            else:
                confidence = 0.1

            # Compute intensity multiplier
            if should_throttle:
                intensity_multiplier = self.THROTTLE_MULTIPLIER
            elif should_boost:
                intensity_multiplier = self.BOOST_MULTIPLIER
            else:
                intensity_multiplier = 1.0

            # Compute abstention threshold adjustment
            # If abstentions are trending up, lower the threshold (be more strict)
            # If abstentions are trending down, raise the threshold (be more lenient)
            if abstention_trend == "up":
                abstention_threshold_adjustment = -0.05
            elif abstention_trend == "down":
                abstention_threshold_adjustment = 0.02
            else:
                abstention_threshold_adjustment = 0.0

            signal = FOFeedbackSignal(
                success_rate=round(success_rate, 2),
                health_status=health_status,
                duration_trend=duration_trend,
                abstention_trend=abstention_trend,
                runs_total=runs_total,
                last_ht_hash=last_ht,
                should_throttle=should_throttle,
                should_boost=should_boost,
                confidence=confidence,
                intensity_multiplier=intensity_multiplier,
                abstention_threshold_adjustment=abstention_threshold_adjustment,
            )

            logger.info(
                f"[FO Feedback] health={health_status} success_rate={success_rate:.1f}% "
                f"throttle={should_throttle} boost={should_boost} confidence={confidence:.1f}"
            )

            return signal

        except Exception as exc:
            logger.error(f"Failed to generate FO feedback signal: {exc}")
            return None

    def _compute_trend(self, history: List[float], epsilon: float = 0.01) -> str:
        """Compute trend direction from history."""
        if len(history) < 2:
            return "flat"
        delta = history[0] - history[1]
        if delta > epsilon:
            return "up"
        elif delta < -epsilon:
            return "down"
        return "flat"

    @staticmethod
    def _to_int(value: Optional[str]) -> int:
        try:
            return int(value) if value else 0
        except (ValueError, TypeError):
            return 0

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
# Feedback Loop Writer (for RFL runner to emit decisions)
# ---------------------------------------------------------------------------


class FOFeedbackWriter:
    """
    Writes feedback loop decisions to Redis for audit trail.

    The RFL runner calls this after consuming a feedback signal
    to record what policy adjustments were made.
    """

    REDIS_KEY_PREFIX = "ml:metrics:fo_feedback"
    HISTORY_MAX_LEN = 100

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL", "redis://localhost:6379/0"
        )
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
        return f"{self.REDIS_KEY_PREFIX}:{name}"

    def record_decision(
        self,
        signal: FOFeedbackSignal,
        action_taken: str,
        adjusted_params: Dict[str, Any],
        experiment_id: str,
    ) -> bool:
        """
        Record a feedback loop decision.

        Args:
            signal: The feedback signal that was consumed
            action_taken: Description of action (e.g., "throttled", "boosted", "none")
            adjusted_params: Dictionary of parameters that were adjusted
            experiment_id: RFL experiment ID

        Returns:
            True if recorded successfully
        """
        if not self._client:
            return False

        try:
            decision = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": experiment_id,
                "signal": signal.to_dict(),
                "action_taken": action_taken,
                "adjusted_params": adjusted_params,
            }

            # Compute decision hash for audit
            decision_hash = hashlib.sha256(
                json.dumps(decision, sort_keys=True).encode()
            ).hexdigest()[:16]
            decision["decision_hash"] = decision_hash

            pipe = self._client.pipeline()

            # Store latest decision
            pipe.set(self._key("latest_decision"), json.dumps(decision))

            # Append to history
            pipe.lpush(self._key("decision_history"), json.dumps(decision))
            pipe.ltrim(self._key("decision_history"), 0, self.HISTORY_MAX_LEN - 1)

            # Update counters
            pipe.incr(self._key("decisions_total"))
            if action_taken == "throttled":
                pipe.incr(self._key("throttle_count"))
            elif action_taken == "boosted":
                pipe.incr(self._key("boost_count"))

            pipe.execute()

            logger.info(
                f"[FO Feedback] Recorded decision: action={action_taken} "
                f"experiment={experiment_id} hash={decision_hash}"
            )

            return True

        except Exception as exc:
            logger.error(f"Failed to record FO feedback decision: {exc}")
            return False

    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent feedback decisions."""
        if not self._client:
            return []

        try:
            raw = self._client.lrange(self._key("decision_history"), 0, limit - 1)
            return [json.loads(item) for item in raw]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


_feedback_reader: Optional[FOFeedbackReader] = None
_feedback_writer: Optional[FOFeedbackWriter] = None


def get_fo_feedback_signal() -> Optional[FOFeedbackSignal]:
    """Get the current FO feedback signal."""
    global _feedback_reader
    if _feedback_reader is None:
        _feedback_reader = FOFeedbackReader()
    return _feedback_reader.get_feedback_signal()


def record_fo_feedback_decision(
    signal: FOFeedbackSignal,
    action_taken: str,
    adjusted_params: Dict[str, Any],
    experiment_id: str,
) -> bool:
    """Record a feedback loop decision."""
    global _feedback_writer
    if _feedback_writer is None:
        _feedback_writer = FOFeedbackWriter()
    return _feedback_writer.record_decision(signal, action_taken, adjusted_params, experiment_id)

