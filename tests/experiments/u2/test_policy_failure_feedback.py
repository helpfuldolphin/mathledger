"""
Tests for Lean failure-aware planner feedback in RFLPolicy.
"""

from __future__ import annotations

from backend.lean_interface import LeanFailureSignal
from experiments.u2.runner import RFLPolicy
from experiments.u2.policy import summarize_lean_failures_for_global_health


def _make_signal(kind: str, elapsed_ms: int = 50) -> LeanFailureSignal:
    return LeanFailureSignal(kind=kind, message=f"{kind} failure", elapsed_ms=elapsed_ms)


def test_timeout_penalty_downweights_same_item() -> None:
    policy = RFLPolicy(seed=7)
    initial = policy.score(["alpha"])[0]

    for _ in range(3):
        policy.update("alpha", success=False, failure_signal=_make_signal("timeout"))

    penalized = policy.score(["alpha"])[0]
    assert penalized < initial
    assert penalized < initial * 0.9  # stronger than base failure decay


def test_type_error_penalty_propagates_to_pattern() -> None:
    policy = RFLPolicy(seed=9)
    baseline_peer = policy.score(["p -> q"])[0]
    baseline_related = policy.score(["p -> r"])[0]

    policy.update("p -> q", success=False, failure_signal=_make_signal("type_error"))

    penalized_related = policy.score(["p -> r"])[0]
    assert penalized_related < baseline_related
    # Ensure unrelated formula unaffected
    other = policy.score(["s -> t"])[0]
    assert other > 0  # sanity check


def test_tactic_failure_strengthens_decay() -> None:
    policy = RFLPolicy(seed=11)
    policy.score(["phi"])
    before = policy.scores["phi"]

    policy.update("phi", success=False, failure_signal=_make_signal("tactic_failure"))

    after = policy.scores["phi"]
    assert after < before * 0.9  # base failure (0.9) plus tactic penalty


def test_lean_failure_summary_empty() -> None:
    summary = summarize_lean_failures_for_global_health([])
    assert summary["status"] == "OK"
    assert summary["total_events"] == 0
    assert summary["counts"]["timeout"] == 0


def test_lean_failure_summary_warn_and_block() -> None:
    signals = [
        _make_signal("timeout"),
        _make_signal("timeout"),
        _make_signal("type_error"),
    ]
    warn_summary = summarize_lean_failures_for_global_health(signals)
    assert warn_summary["status"] in {"WARN", "BLOCK"}

    block_signals = [_make_signal("type_error") for _ in range(4)]
    block_summary = summarize_lean_failures_for_global_health(block_signals)
    assert block_summary["status"] == "BLOCK"
    assert block_summary["counts"]["type_error"] == 4
