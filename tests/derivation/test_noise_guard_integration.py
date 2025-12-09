import math
from pathlib import Path

import pytest

from derivation.noise_guard import (
    VerifierNoiseConfig,
    VerifierNoiseGuard,
    summarize_noise_guard_for_global_health,
)
from derivation.verification import VerificationOutcome


def make_config(tmp_path: Path, **overrides) -> VerifierNoiseConfig:
    defaults = dict(
        window_size=64,
        persist_every=10_000,
        timeout_target=0.0,
        timeout_cusum_drift=0.0,
        timeout_alarm=0.2,
        epsilon_alert=0.02,
        epsilon_cap=0.3,
        delta_h_budget=0.5,
        sprt_baseline=0.01,
        sprt_delta=0.05,
        sprt_eta=4.0,
        tier_costs={"T0": 1.0, "T1": 1.0, "T2": 1.0},
        tier_budget=3.0,
        timeout_weight=1.0,
        tier_weight=1.0,
        queue_weight=1.0,
        flip_weight=1.0,
        metrics_path=tmp_path / "noise_metrics.json",
        state_path=tmp_path / "noise_state.json",
        max_error_signatures=16,
        collapse_threshold=3,
    )
    defaults.update(overrides)
    return VerifierNoiseConfig(**defaults)


def make_guard(tmp_path: Path, **overrides) -> VerifierNoiseGuard:
    return VerifierNoiseGuard(make_config(tmp_path, **overrides))


def record(guard: VerifierNoiseGuard, normalized: str, outcome: VerificationOutcome, tier: str = "T0") -> None:
    guard.record_verification(normalized, outcome, tier_hint=tier)


def test_epsilon_total_matches_expected(tmp_path: Path) -> None:
    guard = make_guard(tmp_path)

    # 6 clean T0 successes
    for _ in range(6):
        record(guard, "p -> p", VerificationOutcome(True, "pattern"), tier="T0")

    # 3 truth-table timeouts (T0)
    for _ in range(3):
        record(guard, "p -> q", VerificationOutcome(False, "timeout"), tier="T0")

    # 2 Lean fallback abstentions (tier != T0)
    for _ in range(2):
        record(guard, "q -> r", VerificationOutcome(False, "lean-disabled"), tier="T1")

    # 1 infrastructure error (queue noise)
    record(guard, "r -> s", VerificationOutcome(False, "truth-table-error"), tier="T0")

    # 1 stochastic flip from imperfect wrapper
    record(
        guard,
        "s -> t",
        VerificationOutcome(True, "truth-table-NOISY-FLIPPED", "Imperfect verifier simulation"),
        tier="T0",
    )

    total = 13
    p_timeout = 3 / total
    p_tier = 2 / total
    p_queue = 1 / total
    p_flip = 1 / total
    expected = 1 - ((1 - p_timeout) * (1 - p_tier) * (1 - p_queue) * (1 - p_flip))

    assert math.isclose(guard.epsilon_total(), expected, rel_tol=1e-9)


def test_guard_feedback_transitions(tmp_path: Path) -> None:
    # Timeout-driven BLOCK
    guard = make_guard(tmp_path, timeout_alarm=0.05, timeout_target=0.0, timeout_cusum_drift=0.0)
    for _ in range(3):
        record(guard, "p -> q", VerificationOutcome(False, "timeout"), tier="T0")
    allowed, reason = guard.guard_feedback(0.01)
    assert not allowed
    assert reason == "timeout-cusum"

    # Epsilon cap driven BLOCK
    guard2 = make_guard(tmp_path, epsilon_cap=0.15)
    for _ in range(20):
        record(guard2, "p -> p", VerificationOutcome(True, "pattern"), tier="T0")
    for _ in range(5):
        record(guard2, "p -> q", VerificationOutcome(False, "truth-table-error"), tier="T0")
    allowed2, reason2 = guard2.guard_feedback(0.0)
    assert not allowed2
    assert reason2 == "epsilon-cap"

    # Clean run should be OK
    guard3 = make_guard(tmp_path)
    for _ in range(5):
        record(guard3, "p -> p", VerificationOutcome(True, "pattern"), tier="T0")
    allowed3, reason3 = guard3.guard_feedback(0.0)
    assert allowed3
    assert reason3 is None


def test_summarize_noise_guard_for_global_health_statuses() -> None:
    base_window = {
        "epsilon_total": 0.03,
        "timeout_noisy": False,
        "unstable_buckets": [],
        "delta_h_bound": 0.15,
        "window_id": 7,
    }
    summary = summarize_noise_guard_for_global_health(base_window)
    assert summary["status"] == "OK"
    assert summary["notes"] == ["stable"]

    attention_window = {
        **base_window,
        "epsilon_total": 0.12,
        "unstable_buckets": [{"tier": "T1", "reason": "mismatch", "llr": 5.0}],
    }
    attention_summary = summarize_noise_guard_for_global_health(attention_window)
    assert attention_summary["status"] == "ATTENTION"
    assert "epsilon>=0.10" in attention_summary["notes"]

    block_window = {
        **base_window,
        "timeout_noisy": True,
        "epsilon_total": 0.30,
    }
    block_summary = summarize_noise_guard_for_global_health(block_window)
    assert block_summary["status"] == "BLOCK"
    assert "timeout-noisy" in block_summary["notes"]
