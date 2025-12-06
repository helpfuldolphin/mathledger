"""Verify the Derivation → Ledger → RFL closed loop via the bridge shim."""

from __future__ import annotations

import pytest

from derivation.pipeline import make_first_organism_derivation_config
from backend.bridge.shim import run_first_organism_cycle
from rfl.config import CurriculumSlice as RFLCurriculumSlice, RFLConfig
from rfl.runner import RFLRunner as RFLRunnerCanonical

@pytest.mark.integration
@pytest.mark.first_organism
def test_first_organism_bridge_closed_loop(first_organism_db) -> None:
    """
    Ensure a single derivation tick produces an AttestedRunContext that the RFL runner
    consumes directly, closing the loop within one cycle.
    """
    config = make_first_organism_derivation_config()
    slice_cfg = config.slice_cfg

    with first_organism_db.cursor() as cursor:
        context = run_first_organism_cycle(
            {
                "slice_cfg": slice_cfg,
                "cursor": cursor,
                "limit": 1,
                "policy_id": "bridge::canonical",
            }
        )

    first_organism_db.commit()

    rfl_config = RFLConfig(
        experiment_id="bridge-loop",
        num_runs=2,
        random_seed=12345,
        derive_steps=1,
        max_breadth=1,
        max_total=1,
        depth_max=1,
        bootstrap_replicates=1000,
        coverage_threshold=0.0,
        uplift_threshold=0.0,
        dual_attestation=False,
        curriculum=[
            RFLCurriculumSlice(
                name=slice_cfg.name,
                start_run=1,
                end_run=2,
                derive_steps=1,
                max_breadth=1,
                max_total=1,
                depth_max=1,
            )
        ],
    )

    runner = RFLRunnerCanonical(rfl_config)
    result = runner.run_with_attestation(context)

    assert result.source_root == context.composite_root
    assert runner.policy_ledger[-1].symbolic_descent == pytest.approx(
        -(context.abstention_rate - runner.config.abstention_tolerance)
    )
    assert runner.abstention_histogram["attestation_mass"] == round(context.abstention_mass)
    assert runner.first_organism_runs_total == 1

