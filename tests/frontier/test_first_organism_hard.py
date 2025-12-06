import pytest
from backend.frontier.curriculum import (
    CurriculumSystem,
    GateVerdict,
    make_first_organism_pl2_hard_slice,
    should_ratchet,
    build_first_organism_metrics,
)

def test_hard_slice_instantiation():
    """Verify the hard slice can be created and has expected parameters."""
    slice_obj = make_first_organism_pl2_hard_slice()
    assert slice_obj.name == "first_organism_pl2_hard"
    assert slice_obj.params["atoms"] == 6
    assert slice_obj.params["depth_max"] == 8
    assert slice_obj.gates.abstention.max_rate_pct == 20.0

def test_hard_slice_holds_ratchet_on_abstention_failure():
    """
    Verify that if the 'hard' slice encounters too many abstentions (Lean failures),
    the ratchet stays put (advance=False).
    """
    hard_slice = make_first_organism_pl2_hard_slice()
    system_cfg = CurriculumSystem(
        slug="organism-hard",
        description="First Organism Hard Test",
        slices=[hard_slice],
        active_index=0,
        version=2,
    )

    # Metrics simulating a hard run:
    # - High abstention rate (25% > 20% limit)
    # - Good velocity (130 > 120)
    # - Good coverage (0.92 > 0.90)
    metrics = build_first_organism_metrics(
        abstention_rate=25.0,
        coverage_ci=0.92,
        proof_velocity_pph=130.0,
        velocity_cv=0.10,
        attestation_hash="beef" * 16
    )

    verdict = should_ratchet(metrics, system_cfg)

    assert verdict.advance is False
    assert "abstention" in verdict.reason
    
    abstention_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "abstention")
    assert abstention_gate["passed"] is False
    assert abstention_gate["observed"]["abstention_rate_pct"] == 25.0

def test_hard_slice_holds_ratchet_on_metrics_within_thresholds_but_incomplete():
    """
    Verify that even if metrics are 'okay' but not fully meeting advancement criteria 
    (e.g. coverage slightly low), it allows the run (doesn't crash) but doesn't advance.
    
    The prompt asks: "Metrics at or below thresholds still allow this slice."
    We interpret "allow this slice" as the system functioning correctly and evaluating gates,
    potentially staying on the slice.
    """
    hard_slice = make_first_organism_pl2_hard_slice()
    system_cfg = CurriculumSystem(
        slug="organism-hard",
        description="First Organism Hard Test",
        slices=[hard_slice],
        active_index=0,
        version=2,
    )

    # Metrics simulating a run that is "hard" but valid:
    # - Abstention 19% (just under 20% limit) -> PASS
    # - Coverage 0.85 (under 0.90 limit) -> FAIL
    metrics = build_first_organism_metrics(
        abstention_rate=19.0,
        coverage_ci=0.85,
        proof_velocity_pph=130.0
    )

    verdict = should_ratchet(metrics, system_cfg)

    assert verdict.advance is False
    assert "coverage" in verdict.reason
    
    # Abstention should pass
    abstention_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "abstention")
    assert abstention_gate["passed"] is True

    # Coverage should fail
    coverage_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "coverage")
    assert coverage_gate["passed"] is False

def test_hard_slice_advances_when_all_hard_targets_met():
    """
    Verify that if we actually conquer the hard slice, it advances.
    """
    hard_slice = make_first_organism_pl2_hard_slice()
    system_cfg = CurriculumSystem(
        slug="organism-hard",
        description="First Organism Hard Test",
        slices=[hard_slice],
        active_index=0,
        version=2,
    )

    # Metrics meeting all criteria
    metrics = build_first_organism_metrics(
        abstention_rate=15.0,  # < 20%
        coverage_ci=0.91,      # > 90%
        proof_velocity_pph=150.0, # > 120
        velocity_cv=0.10,      # < 0.15
        attestation_hash="deadbeef" * 8
    )

    verdict = should_ratchet(metrics, system_cfg)

    assert verdict.advance is True
