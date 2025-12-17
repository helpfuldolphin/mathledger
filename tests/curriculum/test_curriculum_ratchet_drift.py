import json

from curriculum.gates import (
    CurriculumSystem,
    make_first_organism_slice,
    build_first_organism_metrics,
    should_ratchet,
)


def _make_system(slice_cfg) -> CurriculumSystem:
    return CurriculumSystem(
        slug="demo-system",
        description="Demo curriculum",
        slices=[slice_cfg],
        active_index=0,
        monotonic_axes=("atoms", "depth_max", "breadth_max", "total_max"),
    )


def _passing_metrics() -> dict:
    return build_first_organism_metrics(
        coverage_ci=0.97,
        sample_size=64,
        abstention_rate=5.0,
        attempt_mass=3200,
        proof_velocity_pph=210.0,
        velocity_cv=0.05,
        runtime_minutes=30.0,
        backlog_fraction=0.20,
    )


def test_should_ratchet_persists_drift_snapshot_and_provenance() -> None:
    slice_cfg = make_first_organism_slice()
    system = _make_system(slice_cfg)

    verdict = should_ratchet(
        _passing_metrics(),
        system,
        curriculum_fingerprint="curriculum-demo-hash",
    )

    snapshot = verdict.audit["drift_snapshot"]
    event = verdict.audit["drift_provenance_event"]

    assert snapshot["status"] == "OK"
    payload = json.loads(event)
    assert payload["curriculum_fingerprint"] == "curriculum-demo-hash"
    assert payload["slice_name"] == slice_cfg.name
    assert payload["drift_status"] == "OK"


def test_should_ratchet_detects_runtime_slice_regressions() -> None:
    slice_cfg = make_first_organism_slice()
    system = _make_system(slice_cfg)

    runtime_slice = CurriculumSystem._slice_to_dict(slice_cfg)
    runtime_slice = json.loads(json.dumps(runtime_slice))
    runtime_slice["params"]["depth_max"] -= 1

    verdict = should_ratchet(
        _passing_metrics(),
        system,
        current_slice_override=runtime_slice,
        curriculum_fingerprint="curriculum-demo-hash",
    )

    snapshot = verdict.audit["drift_snapshot"]
    assert snapshot["status"] == "BLOCK"
    assert snapshot["severity"] == "SEMANTIC"
    assert snapshot["changed_params"][0]["path"] == "params.depth_max"

    event = json.loads(verdict.audit["drift_provenance_event"])
    assert event["drift_status"] == "BLOCK"
