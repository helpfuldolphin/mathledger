"""
Phase-II Imperfect Verifier noise guard implementation.

The Codex-Verification-Stabilizer keeps track of timeout noise, mixed-tier
instability, queue/infra failures, and stochastic flips emitted by experimental
verifier wrappers. It maintains rolling Wilson/SPRT statistics, computes the
composite epsilon_total, and emits telemetry snapshots for downstream audit.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import yaml

from derivation.derive_utils import sha256_statement

if TYPE_CHECKING:  # pragma: no cover
    from derivation.verification import VerificationOutcome


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VerifierNoiseConfig:
    """Configurable guard rails for imperfect verifier modeling."""

    window_size: int = 1024
    persist_every: int = 64
    timeout_target: float = 0.01
    timeout_cusum_drift: float = 0.0025
    timeout_alarm: float = 0.25
    epsilon_alert: float = 0.02
    epsilon_cap: float = 0.25
    delta_h_budget: float = 0.2
    sprt_baseline: float = 0.001
    sprt_delta: float = 0.004
    sprt_eta: float = 9.0
    tier_costs: Dict[str, float] = field(
        default_factory=lambda: {"T0": 1.0, "T1": 2.0, "T2": 4.0}
    )
    tier_budget: float = 2.0
    timeout_weight: float = 1.0
    tier_weight: float = 1.0
    queue_weight: float = 1.0
    flip_weight: float = 1.0
    metrics_path: Path = Path("metrics/verifier_noise_window.json")
    state_path: Path = Path("artifacts/verifier_noise_guard.json")
    max_error_signatures: int = 256
    collapse_threshold: int = 3

    @classmethod
    def from_file(cls, path: Path | str) -> "VerifierNoiseConfig":
        data = yaml.safe_load(Path(path).read_text())
        window = data.get("window", {})
        channels = data.get("channels", {})
        timeout = data.get("timeout", {})
        epsilon = data.get("epsilon", {})
        sprt = data.get("sprt", {})
        errors = data.get("error_signatures", {})
        paths_cfg = data.get("paths", {})
        tier_costs = data.get("tier_costs", {"T0": 1.0, "T1": 2.0, "T2": 4.0})
        return cls(
            window_size=int(window.get("size", 1024)),
            persist_every=int(window.get("persist_every", 64)),
            timeout_target=float(timeout.get("target", 0.01)),
            timeout_cusum_drift=float(timeout.get("cusum_drift", 0.0025)),
            timeout_alarm=float(timeout.get("alarm", 0.25)),
            epsilon_alert=float(epsilon.get("alert", 0.02)),
            epsilon_cap=float(epsilon.get("cap", 0.25)),
            delta_h_budget=float(data.get("delta_h_budget", 0.2)),
            sprt_baseline=float(sprt.get("baseline", 0.001)),
            sprt_delta=float(sprt.get("delta", 0.004)),
            sprt_eta=float(sprt.get("eta", 9.0)),
            tier_costs={k: float(v) for k, v in tier_costs.items()},
            tier_budget=float(data.get("tier_budget", 2.0)),
            timeout_weight=float(channels.get("timeout_weight", 1.0)),
            tier_weight=float(channels.get("tier_weight", 1.0)),
            queue_weight=float(channels.get("queue_weight", 1.0)),
            flip_weight=float(channels.get("flip_weight", 1.0)),
            metrics_path=Path(paths_cfg.get("metrics", "metrics/verifier_noise_window.json")),
            state_path=Path(paths_cfg.get("state", "artifacts/verifier_noise_guard.json")),
            max_error_signatures=int(errors.get("max_tracked", 256)),
            collapse_threshold=int(errors.get("collapse_threshold", 3)),
        )


def load_config_from_env() -> Optional[VerifierNoiseConfig]:
    """Load config when the canonical YAML exists."""
    config_path = os.getenv("ML_VERIFIER_NOISE_CONFIG", "config/verifier_noise_phase2.yaml")
    path = Path(config_path)
    if not path.exists():
        return None
    try:
        return VerifierNoiseConfig.from_file(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for Binomial coverage."""
    if total <= 0:
        return (0.0, 0.0)

    phat = successes / total
    denom = 1.0 + (z ** 2) / total
    center = phat + (z ** 2) / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) / total) + (z ** 2) / (4 * total * total))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


# ---------------------------------------------------------------------------
# Error signatures
# ---------------------------------------------------------------------------


@dataclass
class ErrorSignature:
    tier_id: str
    reason: str
    statement_hash: str
    window_id: int
    attestation_ptr: Optional[str] = None
    flagged_at: float = field(default_factory=time.time)
    buckets: Counter = field(default_factory=Counter)

    def to_dict(self) -> Dict[str, object]:
        return {
            "tier_id": self.tier_id,
            "reason": self.reason,
            "statement_hash": self.statement_hash,
            "window_id": self.window_id,
            "attestation_ptr": self.attestation_ptr,
            "flagged_at": self.flagged_at,
            "buckets": dict(self.buckets),
        }


# ---------------------------------------------------------------------------
# Guard implementation
# ---------------------------------------------------------------------------


class VerifierNoiseGuard:
    """Telemetry sink plus guard rails for imperfect verifiers."""

    def __init__(self, config: VerifierNoiseConfig):
        self.config = config
        self._window: Deque[Dict[str, object]] = deque(maxlen=config.window_size)
        self._event_counter = 0
        self._timeout_events = 0
        self._fallback_events = 0
        self._queue_events = 0
        self._flip_events = 0
        self._tier_trials: Counter[str] = Counter()
        self._tier_successes: Counter[str] = Counter()
        self._timeout_cusum = 0.0
        self._tier_routing_weights: Dict[str, float] = {"T0": 1.0, "T1": 0.0, "T2": 0.0}
        self._bucket_llr: Dict[Tuple[str, str], float] = {}
        self._unstable_buckets: Dict[Tuple[str, str], float] = {}
        self._error_signatures: Dict[str, ErrorSignature] = {}
        self._window_id = 0

    # ------------- Recording -------------

    def record_verification(
        self,
        normalized: str,
        outcome: "VerificationOutcome",
        *,
        tier_hint: Optional[str] = None,
        attestation_ptr: Optional[str] = None,
    ) -> None:
        """Record a verifier outcome for telemetry and guard rails."""
        tier = tier_hint or _tier_from_method(outcome.method)
        reason = _reason_from_outcome(outcome)
        statement_hash = sha256_statement(normalized)

        self._event_counter += 1
        self._window_id = self._event_counter // max(1, self.config.window_size)
        self._tier_trials[tier] += 1
        if outcome.verified:
            self._tier_successes[tier] += 1

        if reason == "timeout":
            self._timeout_events += 1
        if tier != "T0":
            self._fallback_events += 1
        if reason == "infra":
            self._queue_events += 1
        if reason == "flip":
            self._flip_events += 1

        event = {
            "hash": statement_hash,
            "tier": tier,
            "method": outcome.method,
            "verified": outcome.verified,
            "reason": reason,
            "window_id": self._window_id,
        }
        self._window.append(event)

        bucket_key = (tier, reason)
        self._update_sprt(bucket_key, reason in {"timeout", "mismatch", "infra", "flip"})

        if not outcome.verified and reason != "timeout":
            self._track_error_signature(statement_hash, bucket_key, attestation_ptr)

        timeout_rate = self._recent_rate("timeout")
        self._update_timeout_cusum(timeout_rate)

        if self._event_counter % max(1, self.config.persist_every) == 0:
            self.persist_snapshot()

    def record_queue_anomaly(self) -> None:
        self._queue_events += 1
        self.persist_snapshot()

    # ------------- Metrics -------------

    def epsilon_channels(self) -> Dict[str, float]:
        total = max(1, self._event_counter)
        p_timeout = self._timeout_events / total
        p_tier = self._fallback_events / total
        p_queue = self._queue_events / total
        p_flip = self._flip_events / total
        return {
            "timeout": p_timeout,
            "tier": p_tier,
            "queue": p_queue,
            "flip": p_flip,
        }

    def epsilon_total(self) -> float:
        channels = self.epsilon_channels()
        total = 1.0
        for key, prob in channels.items():
            weight = getattr(self.config, f"{key}_weight", 1.0)
            total *= (1.0 - weight * prob)
        total = max(0.0, total)
        return 1.0 - total

    def tier_accuracy(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for tier, total in self._tier_trials.items():
            success = self._tier_successes.get(tier, 0)
            lower, upper = wilson_interval(success, total)
            summary[tier] = {
                "success": float(success),
                "total": float(total),
                "wilson_lower": lower,
                "wilson_upper": upper,
            }
        return summary

    def timeout_noisy(self) -> bool:
        return self._timeout_cusum >= self.config.timeout_alarm

    def unstable_hashes(self) -> List[str]:
        return [
            sig.statement_hash
            for sig in self._error_signatures.values()
            if len(sig.buckets) >= self.config.collapse_threshold
        ]

    def guard_feedback(self, delta_h: float) -> Tuple[bool, Optional[str]]:
        if self.timeout_noisy():
            return False, "timeout-cusum"
        epsilon_total = self.epsilon_total()
        if epsilon_total >= self.config.epsilon_cap:
            return False, "epsilon-cap"
        clamped = self.delta_h_bound()
        if abs(delta_h) > clamped:
            return False, "delta-h-clamp"
        if self.unstable_hashes():
            return False, "canonical-error"
        return True, None

    def should_allow_policy_update(self, delta_h: float) -> bool:
        allowed, _ = self.guard_feedback(delta_h)
        return allowed

    def delta_h_bound(self) -> float:
        return max(0.0, self.config.delta_h_budget * (1.0 - self.epsilon_total()))

    def routing_weights(self) -> Dict[str, float]:
        total_cost = sum(
            weight * self.config.tier_costs.get(tier, 1.0)
            for tier, weight in self._tier_routing_weights.items()
        )
        if total_cost <= self.config.tier_budget:
            return dict(self._tier_routing_weights)

        # Simple normalization when budget exceeded
        scale = self.config.tier_budget / max(total_cost, 1e-6)
        return {tier: weight * scale for tier, weight in self._tier_routing_weights.items()}

    # ------------- Persistence -------------

    def persist_snapshot(self) -> None:
        snapshot = {
            "timestamp": time.time(),
            "window_size": len(self._window),
            "window_id": self._window_id,
            "channels": self.epsilon_channels(),
            "epsilon_total": self.epsilon_total(),
            "timeout_cusum": self._timeout_cusum,
            "timeout_noisy": self.timeout_noisy(),
            "tier_accuracy": self.tier_accuracy(),
            "unstable_buckets": [
                {"tier": tier, "reason": reason, "llr": llr}
                for (tier, reason), llr in self._unstable_buckets.items()
            ],
            "error_signatures": [sig.to_dict() for sig in list(self._error_signatures.values())[-32:]],
            "delta_h_bound": self.delta_h_bound(),
        }

        metrics_path = self.config.metrics_path
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(snapshot, indent=2))

        state_path = self.config.state_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(snapshot, indent=2))

    # ------------- Internals -------------

    def _recent_rate(self, reason: str) -> float:
        if not self._window:
            return 0.0
        count = sum(1 for entry in self._window if entry.get("reason") == reason)
        return count / len(self._window)

    def _update_timeout_cusum(self, rate: float) -> None:
        delta = rate - self.config.timeout_target - self.config.timeout_cusum_drift
        self._timeout_cusum = max(0.0, self._timeout_cusum + delta)

    def _update_sprt(self, bucket: Tuple[str, str], is_error: bool) -> None:
        p0 = self.config.sprt_baseline
        p1 = min(1.0, self.config.sprt_baseline + self.config.sprt_delta)
        llr = self._bucket_llr.get(bucket, 0.0)
        if is_error:
            llr += math.log(max(p1, 1e-9) / max(p0, 1e-9))
        else:
            llr += math.log(max(1.0 - p1, 1e-9) / max(1.0 - p0, 1e-9))
        llr = max(0.0, llr)
        self._bucket_llr[bucket] = llr
        if llr >= self.config.sprt_eta:
            self._unstable_buckets[bucket] = llr
        elif bucket in self._unstable_buckets:
            self._unstable_buckets.pop(bucket, None)

    def _track_error_signature(
        self,
        statement_hash: str,
        bucket: Tuple[str, str],
        attestation_ptr: Optional[str],
    ) -> None:
        key = statement_hash
        sig = self._error_signatures.get(key)
        if not sig:
            if len(self._error_signatures) >= self.config.max_error_signatures:
                # Drop the oldest signature deterministically.
                oldest = min(self._error_signatures.values(), key=lambda s: s.flagged_at)
                self._error_signatures.pop(oldest.statement_hash, None)
            sig = ErrorSignature(
                tier_id=bucket[0],
                reason=bucket[1],
                statement_hash=statement_hash,
                window_id=self._window_id,
                attestation_ptr=attestation_ptr,
            )
            self._error_signatures[key] = sig
        sig.buckets[bucket] += 1
        sig.flagged_at = time.time()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tier_from_method(method: str) -> str:
    m = method or ""
    if m.startswith("pattern") or m.startswith("truth-table") or m == "timeout":
        return "T0"
    if m.startswith("lean-lite") or m.startswith("lean-disabled"):
        return "T1"
    if m.startswith("lean"):
        return "T2"
    return "T1"


def _reason_from_outcome(outcome: "VerificationOutcome") -> str:
    method = outcome.method or ""
    details = outcome.details or ""
    if "NOISY-FLIPPED" in method or "Imperfect verifier simulation" in details:
        return "flip"
    if method == "timeout":
        return "timeout"
    if method.endswith("error"):
        return "infra"
    if method.startswith("lean") and not outcome.verified:
        return "mismatch"
    if not outcome.verified:
        return "mismatch"
    return "clean"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_GLOBAL_GUARD: Optional[VerifierNoiseGuard] = None


def global_noise_guard() -> Optional[VerifierNoiseGuard]:
    """Singleton accessor used by StatementVerifier when config is present."""
    global _GLOBAL_GUARD
    if _GLOBAL_GUARD is not None:
        return _GLOBAL_GUARD
    config = load_config_from_env()
    if not config:
        return None
    _GLOBAL_GUARD = VerifierNoiseGuard(config)
    return _GLOBAL_GUARD


def summarize_noise_guard_for_global_health(window: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a health tile for dashboards / evidence packs.

    Args:
        window: Latest metrics snapshot emitted by VerifierNoiseGuard.persist_snapshot().

    Returns:
        dict with keys: status ("OK"|"ATTENTION"|"BLOCK"), epsilon_total, notes, etc.
    """
    epsilon = float(window.get("epsilon_total") or 0.0)
    timeout_noisy = bool(window.get("timeout_noisy"))
    unstable = window.get("unstable_buckets") or []
    unstable_count = len(unstable)

    status = "OK"
    notes: List[str] = []

    if timeout_noisy:
        status = "BLOCK"
        notes.append("timeout-noisy")

    if epsilon >= 0.25:
        status = "BLOCK"
        notes.append("epsilon>=0.25")
    elif epsilon >= 0.10 and status != "BLOCK":
        status = "ATTENTION"
        notes.append("epsilon>=0.10")

    if unstable_count and status != "BLOCK":
        status = "ATTENTION"
        notes.append("bucket-instability")

    if not notes:
        notes.append("stable")

    return {
        "status": status,
        "epsilon_total": round(epsilon, 6),
        "timeout_noisy": timeout_noisy,
        "unstable_bucket_count": unstable_count,
        "delta_h_bound": window.get("delta_h_bound"),
        "notes": notes,
        "window_id": window.get("window_id"),
    }


__all__ = [
    "VerifierNoiseConfig",
    "VerifierNoiseGuard",
    "global_noise_guard",
    "summarize_noise_guard_for_global_health",
]
