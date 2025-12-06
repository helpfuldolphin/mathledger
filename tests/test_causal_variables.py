"""
Unit tests for causal variable extraction and delta computation.

Tests RunMetrics, RunDelta, and delta extraction from run history.
"""

import pytest
from datetime import datetime, timedelta
from backend.causal.variables import (
    RunMetrics,
    RunDelta,
    compute_policy_delta,
    compute_abstention_delta,
    compute_throughput_delta,
    compute_run_delta,
    extract_run_deltas,
    stratify_by_policy_change,
    compute_mean_deltas,
    summary_statistics
)


def test_run_metrics_creation():
    """Test creating RunMetrics."""
    now = datetime.now()
    later = now + timedelta(seconds=100)

    metrics = RunMetrics(
        run_id=1,
        started_at=now,
        ended_at=later,
        policy_hash="abc123",
        abstain_pct=10.5,
        proofs_success=100,
        proofs_per_sec=1.0,
        depth_max_reached=4,
        system="pl",
        slice_name="test"
    )

    assert metrics.run_id == 1
    assert metrics.policy_hash == "abc123"
    assert metrics.abstain_pct == 10.5
    assert metrics.duration_seconds() == 100.0


def test_compute_policy_delta_same():
    """Test policy delta when policies are same."""
    now = datetime.now()

    baseline = RunMetrics(
        run_id=1, started_at=now, ended_at=now,
        policy_hash="abc", abstain_pct=0, proofs_success=0,
        proofs_per_sec=0, depth_max_reached=0,
        system="pl", slice_name="test"
    )

    comparison = RunMetrics(
        run_id=2, started_at=now, ended_at=now,
        policy_hash="abc", abstain_pct=0, proofs_success=0,
        proofs_per_sec=0, depth_max_reached=0,
        system="pl", slice_name="test"
    )

    delta = compute_policy_delta(baseline, comparison)
    assert delta == 0


def test_compute_policy_delta_different():
    """Test policy delta when policies differ."""
    now = datetime.now()

    baseline = RunMetrics(
        run_id=1, started_at=now, ended_at=now,
        policy_hash="abc", abstain_pct=0, proofs_success=0,
        proofs_per_sec=0, depth_max_reached=0,
        system="pl", slice_name="test"
    )

    comparison = RunMetrics(
        run_id=2, started_at=now, ended_at=now,
        policy_hash="xyz", abstain_pct=0, proofs_success=0,
        proofs_per_sec=0, depth_max_reached=0,
        system="pl", slice_name="test"
    )

    delta = compute_policy_delta(baseline, comparison)
    assert delta == 1


def test_compute_abstention_delta():
    """Test abstention delta computation."""
    now = datetime.now()

    baseline = RunMetrics(
        run_id=1, started_at=now, ended_at=now,
        policy_hash=None, abstain_pct=10.0, proofs_success=0,
        proofs_per_sec=0, depth_max_reached=0,
        system="pl", slice_name="test"
    )

    comparison = RunMetrics(
        run_id=2, started_at=now, ended_at=now,
        policy_hash=None, abstain_pct=15.5, proofs_success=0,
        proofs_per_sec=0, depth_max_reached=0,
        system="pl", slice_name="test"
    )

    delta = compute_abstention_delta(baseline, comparison)
    assert delta == 5.5


def test_compute_throughput_delta():
    """Test throughput delta computation."""
    now = datetime.now()

    baseline = RunMetrics(
        run_id=1, started_at=now, ended_at=now,
        policy_hash=None, abstain_pct=0, proofs_success=0,
        proofs_per_sec=1.0, depth_max_reached=0,
        system="pl", slice_name="test"
    )

    comparison = RunMetrics(
        run_id=2, started_at=now, ended_at=now,
        policy_hash=None, abstain_pct=0, proofs_success=0,
        proofs_per_sec=1.5, depth_max_reached=0,
        system="pl", slice_name="test"
    )

    delta = compute_throughput_delta(baseline, comparison)
    assert delta == 0.5


def test_compute_run_delta():
    """Test computing full run delta."""
    now = datetime.now()

    baseline = RunMetrics(
        run_id=1, started_at=now, ended_at=now,
        policy_hash="abc", abstain_pct=10.0, proofs_success=100,
        proofs_per_sec=1.0, depth_max_reached=3,
        system="pl", slice_name="test"
    )

    comparison = RunMetrics(
        run_id=2, started_at=now, ended_at=now,
        policy_hash="xyz", abstain_pct=15.0, proofs_success=150,
        proofs_per_sec=1.5, depth_max_reached=4,
        system="pl", slice_name="test"
    )

    delta = compute_run_delta(baseline, comparison)

    assert delta.delta_policy == 1
    assert delta.delta_abstain == 5.0
    assert delta.delta_proof == 50
    assert delta.delta_throughput == 0.5
    assert delta.delta_depth == 1
    assert delta.policy_changed is True


def test_extract_run_deltas():
    """Test extracting deltas from run sequence."""
    now = datetime.now()

    runs = [
        RunMetrics(
            run_id=i, started_at=now + timedelta(hours=i), ended_at=now + timedelta(hours=i),
            policy_hash=f"policy_{i % 2}", abstain_pct=10.0 + i,
            proofs_success=100 + i * 10, proofs_per_sec=1.0 + i * 0.1,
            depth_max_reached=3, system="pl", slice_name="test"
        )
        for i in range(5)
    ]

    deltas = extract_run_deltas(runs)

    # Should have 4 deltas (5 runs - 1)
    assert len(deltas) == 4

    # Check first delta
    assert deltas[0].baseline_run.run_id == 0
    assert deltas[0].comparison_run.run_id == 1


def test_extract_run_deltas_different_systems():
    """Test that deltas are not computed across different systems."""
    now = datetime.now()

    runs = [
        RunMetrics(
            run_id=1, started_at=now, ended_at=now,
            policy_hash="abc", abstain_pct=10.0, proofs_success=100,
            proofs_per_sec=1.0, depth_max_reached=3,
            system="pl", slice_name="test"
        ),
        RunMetrics(
            run_id=2, started_at=now + timedelta(hours=1), ended_at=now + timedelta(hours=1),
            policy_hash="xyz", abstain_pct=12.0, proofs_success=110,
            proofs_per_sec=1.1, depth_max_reached=3,
            system="fol", slice_name="test"  # Different system
        )
    ]

    deltas = extract_run_deltas(runs)

    # Should not compute delta across different systems
    assert len(deltas) == 0


def test_stratify_by_policy_change():
    """Test stratifying deltas by policy change."""
    now = datetime.now()

    baseline = RunMetrics(
        run_id=1, started_at=now, ended_at=now,
        policy_hash="abc", abstain_pct=10.0, proofs_success=100,
        proofs_per_sec=1.0, depth_max_reached=3,
        system="pl", slice_name="test"
    )

    # One with same policy
    comparison1 = RunMetrics(
        run_id=2, started_at=now, ended_at=now,
        policy_hash="abc", abstain_pct=11.0, proofs_success=105,
        proofs_per_sec=1.05, depth_max_reached=3,
        system="pl", slice_name="test"
    )

    # One with different policy
    comparison2 = RunMetrics(
        run_id=3, started_at=now, ended_at=now,
        policy_hash="xyz", abstain_pct=12.0, proofs_success=110,
        proofs_per_sec=1.1, depth_max_reached=3,
        system="pl", slice_name="test"
    )

    delta1 = compute_run_delta(baseline, comparison1)
    delta2 = compute_run_delta(baseline, comparison2)

    changed, unchanged = stratify_by_policy_change([delta1, delta2])

    assert len(changed) == 1
    assert len(unchanged) == 1
    assert delta2 in changed
    assert delta1 in unchanged


def test_compute_mean_deltas():
    """Test computing mean values across deltas."""
    now = datetime.now()

    deltas = []
    for i in range(3):
        baseline = RunMetrics(
            run_id=i * 2, started_at=now, ended_at=now,
            policy_hash="abc", abstain_pct=10.0, proofs_success=100,
            proofs_per_sec=1.0, depth_max_reached=3,
            system="pl", slice_name="test"
        )

        comparison = RunMetrics(
            run_id=i * 2 + 1, started_at=now, ended_at=now,
            policy_hash="abc", abstain_pct=10.0 + i, proofs_success=100 + i * 10,
            proofs_per_sec=1.0 + i * 0.1, depth_max_reached=3,
            system="pl", slice_name="test"
        )

        deltas.append(compute_run_delta(baseline, comparison))

    means = compute_mean_deltas(deltas)

    assert means['n_deltas'] == 3
    assert means['mean_delta_abstain'] == pytest.approx(1.0, abs=0.1)
    assert means['mean_delta_throughput'] == pytest.approx(0.1, abs=0.05)


def test_summary_statistics():
    """Test computing summary statistics."""
    now = datetime.now()

    deltas = []
    for i in range(5):
        baseline = RunMetrics(
            run_id=i * 2, started_at=now, ended_at=now,
            policy_hash="abc", abstain_pct=10.0, proofs_success=100,
            proofs_per_sec=1.0, depth_max_reached=3,
            system="pl", slice_name="test"
        )

        comparison = RunMetrics(
            run_id=i * 2 + 1, started_at=now, ended_at=now,
            policy_hash="abc", abstain_pct=10.0 + i, proofs_success=100 + i * 10,
            proofs_per_sec=1.0 + i * 0.1, depth_max_reached=3,
            system="pl", slice_name="test"
        )

        deltas.append(compute_run_delta(baseline, comparison))

    stats = summary_statistics(deltas)

    assert 'abstain' in stats
    assert 'throughput' in stats
    assert 'proof' in stats
    assert stats['n'] == 5

    # Check that mean/std/min/max are present
    assert 'mean' in stats['abstain']
    assert 'std' in stats['abstain']
    assert 'min' in stats['abstain']
    assert 'max' in stats['abstain']


def test_run_delta_to_feature_vector():
    """Test converting run delta to feature vector."""
    now = datetime.now()

    baseline = RunMetrics(
        run_id=1, started_at=now, ended_at=now,
        policy_hash="abc", abstain_pct=10.0, proofs_success=100,
        proofs_per_sec=1.0, depth_max_reached=3,
        system="pl", slice_name="test"
    )

    comparison = RunMetrics(
        run_id=2, started_at=now, ended_at=now,
        policy_hash="xyz", abstain_pct=15.0, proofs_success=150,
        proofs_per_sec=1.5, depth_max_reached=4,
        system="pl", slice_name="test"
    )

    delta = compute_run_delta(baseline, comparison)
    vector = delta.to_feature_vector()

    assert vector.shape == (4,)
    assert vector[0] == 1  # policy changed
    assert vector[1] == 5.0  # abstain delta
    assert vector[2] == 1  # depth delta
    assert vector[3] == 0.5  # throughput delta


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
