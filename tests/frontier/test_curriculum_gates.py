import pytest

from backend.frontier.curriculum import (
    CurriculumSystem,
    GateVerdict,
    build_first_organism_metrics,
    load,
    make_first_organism_slice,
    should_ratchet,
)


def _base_metrics():
    return {
        "metrics": {
            "rfl": {
                "coverage": {
                    "ci_lower": 0.942,
                    "sample_size": 32,
                }
            },
            "success_rates": {
                "abstention_rate": 9.5,
            },
            "curriculum": {
                "active_slice": {
                    "attempt_mass": 7800,
                    "wallclock_minutes": 36,
                    "proof_velocity_cv": 0.04,
                }
            },
            "throughput": {
                "proofs_per_hour": 240,
                "coefficient_of_variation": 0.04,
                "window_minutes": 60,
            },
            "frontier": {
                "queue_backlog": 0.22,
            },
        },
        "provenance": {
            "merkle_hash": "a" * 64,
        },
    }


def test_should_ratchet_passes_when_all_gates_satisfied():
    system_cfg = load("pl")
    metrics = _base_metrics()

    verdict = should_ratchet(metrics, system_cfg)

    assert isinstance(verdict, GateVerdict)
    assert verdict.advance is True
    assert "coverage" in verdict.reason
    coverage_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "coverage")
    assert coverage_gate["passed"] is True


@pytest.mark.parametrize(
    "mutator, expected_gate",
    [
        (lambda data: data["metrics"]["rfl"]["coverage"].update({"ci_lower": 0.91}), "coverage"),
        (lambda data: data["metrics"]["success_rates"].update({"abstention_rate": 16.0}), "abstention"),
        (lambda data: data["metrics"]["throughput"].update({"proofs_per_hour": 180}), "velocity"),
        (lambda data: data["metrics"]["frontier"].update({"queue_backlog": 0.5}), "caps"),
    ],
)
def test_should_ratchet_holds_when_gate_fails(mutator, expected_gate):
    system_cfg = load("pl")
    metrics = _base_metrics()
    mutator(metrics)

    verdict = should_ratchet(metrics, system_cfg)

    assert verdict.advance is False
    assert expected_gate in verdict.reason
    failing_gate = next(g for g in verdict.audit["gates"] if g["gate"] == expected_gate)
    assert failing_gate["passed"] is False


def test_first_organism_slice_allows_run_but_holds_ratchet():
    first_slice = make_first_organism_slice()
    system_cfg = CurriculumSystem(
        slug="organism",
        description="First Organism test curriculum",
        slices=[first_slice],
        active_index=0,
        monotonic_axes=(),
        version=2,
    )
    metrics = build_first_organism_metrics()

    verdict = should_ratchet(metrics, system_cfg)

    assert verdict.advance is False
    assert verdict.audit["attestation_hash"] == metrics["provenance"]["merkle_hash"]
    coverage_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "coverage")
    assert coverage_gate["passed"] is False
    assert "coverage" in verdict.reason
    other_gates = {
        g["gate"]: g["passed"]
        for g in verdict.audit["gates"]
        if g["gate"] != "coverage"
    }
    assert all(other_gates.values()), "Non-coverage gates should pass for first organism metrics"


# ---------------------------------------------------------------------------
# Partial / Missing Metrics Simulation Tests
# ---------------------------------------------------------------------------


def _fo_system() -> CurriculumSystem:
    """Helper to build a First Organism CurriculumSystem."""
    return CurriculumSystem(
        slug="organism",
        description="First Organism test curriculum",
        slices=[make_first_organism_slice()],
        active_index=0,
        monotonic_axes=(),
        version=2,
    )


def _passing_fo_metrics() -> dict:
    """
    Build FO metrics where ALL gates pass, so we can selectively break one.
    Coverage CI is set above the threshold (0.915) to ensure coverage passes.
    """
    return build_first_organism_metrics(coverage_ci=0.93)


def test_missing_coverage_causes_coverage_gate_failure():
    """Coverage gate fails when coverage telemetry is missing."""
    metrics = _passing_fo_metrics()
    del metrics["metrics"]["rfl"]["coverage"]

    verdict = should_ratchet(metrics, _fo_system())

    assert verdict.advance is False
    assert "coverage" in verdict.reason
    assert "coverage_ci_lower missing" in verdict.reason.lower()


def test_missing_abstention_rate_causes_abstention_gate_failure():
    """Abstention gate fails when abstention_rate is missing."""
    metrics = _passing_fo_metrics()
    del metrics["metrics"]["success_rates"]["abstention_rate"]

    verdict = should_ratchet(metrics, _fo_system())

    assert verdict.advance is False
    assert "abstention" in verdict.reason
    abstention_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "abstention")
    assert abstention_gate["passed"] is False


def test_missing_velocity_causes_velocity_gate_failure():
    """Velocity gate fails when proofs_per_hour is missing."""
    metrics = _passing_fo_metrics()
    del metrics["metrics"]["throughput"]["proofs_per_hour"]
    # Also remove window_minutes so it doesn't get used as runtime fallback
    metrics["metrics"]["throughput"].pop("window_minutes", None)

    verdict = should_ratchet(metrics, _fo_system())

    assert verdict.advance is False
    assert "velocity" in verdict.reason
    velocity_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "velocity")
    assert velocity_gate["passed"] is False


def test_missing_backlog_causes_caps_gate_failure():
    """Caps gate fails when queue_backlog is missing."""
    metrics = _passing_fo_metrics()
    del metrics["metrics"]["frontier"]["queue_backlog"]

    verdict = should_ratchet(metrics, _fo_system())

    assert verdict.advance is False
    assert "caps" in verdict.reason
    caps_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "caps")
    assert caps_gate["passed"] is False
    assert "backlog" in caps_gate["message"].lower()


def test_missing_runtime_causes_caps_gate_failure():
    """Caps gate fails when runtime (wallclock_minutes) is missing."""
    metrics = _passing_fo_metrics()
    del metrics["metrics"]["curriculum"]["active_slice"]["wallclock_minutes"]
    # Also remove throughput window_minutes fallback
    metrics["metrics"]["throughput"].pop("window_minutes", None)

    verdict = should_ratchet(metrics, _fo_system())

    assert verdict.advance is False
    assert "caps" in verdict.reason
    caps_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "caps")
    assert caps_gate["passed"] is False
    assert "runtime" in caps_gate["message"].lower()


def test_null_coverage_ci_causes_coverage_gate_failure():
    """Explicit None for coverage CI should fail the coverage gate."""
    metrics = _passing_fo_metrics()
    metrics["metrics"]["rfl"]["coverage"]["ci_lower"] = None

    verdict = should_ratchet(metrics, _fo_system())

    assert verdict.advance is False
    assert "coverage" in verdict.reason
    coverage_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "coverage")
    assert coverage_gate["passed"] is False


def test_null_velocity_cv_causes_velocity_gate_failure():
    """Explicit None for velocity CV should fail the velocity gate."""
    metrics = _passing_fo_metrics()
    metrics["metrics"]["throughput"]["coefficient_of_variation"] = None
    # Also remove fallback
    metrics["metrics"]["curriculum"]["active_slice"].pop("proof_velocity_cv", None)

    verdict = should_ratchet(metrics, _fo_system())

    assert verdict.advance is False
    assert "velocity" in verdict.reason
    velocity_gate = next(g for g in verdict.audit["gates"] if g["gate"] == "velocity")
    assert velocity_gate["passed"] is False


def test_empty_metrics_object_fails_all_gates():
    """An empty metrics payload should fail at the first gate (coverage)."""
    metrics = {"metrics": {}, "provenance": {}}

    verdict = should_ratchet(metrics, _fo_system())

    assert verdict.advance is False
    # Should fail on coverage first since it's evaluated first
    assert "coverage" in verdict.reason


def test_all_gates_pass_when_metrics_complete():
    """Verify that all gates pass when metrics exceed all thresholds."""
    metrics = _passing_fo_metrics()

    verdict = should_ratchet(metrics, _fo_system())

    assert verdict.advance is True
    for gate_status in verdict.audit["gates"]:
        assert gate_status["passed"] is True, f"Gate {gate_status['gate']} should pass"


# ---------------------------------------------------------------------------
# Config Validation Tests
# ---------------------------------------------------------------------------

from backend.frontier.curriculum import (
    CurriculumConfigError,
    validate_curriculum_config,
    make_fol_entry_slice,
    make_fol_equality_slice,
)


def test_validate_curriculum_config_detects_missing_version():
    """Validation fails when version is missing."""
    config = {"systems": {"pl": {"description": "test", "slices": []}}}
    errors = validate_curriculum_config(config)
    assert any("version" in e for e in errors)


def test_validate_curriculum_config_detects_wrong_version():
    """Validation fails when version is not 2."""
    config = {"version": 1, "systems": {"pl": {"description": "test", "slices": []}}}
    errors = validate_curriculum_config(config)
    assert any("version" in e for e in errors)


def test_validate_curriculum_config_detects_missing_gates():
    """Validation fails when a slice is missing gate definitions."""
    config = {
        "version": 2,
        "systems": {
            "pl": {
                "description": "test",
                "slices": [
                    {"name": "s1", "params": {"atoms": 4}, "gates": {"coverage": {}}}
                ],
            }
        },
    }
    errors = validate_curriculum_config(config)
    assert any("abstention" in e or "velocity" in e or "caps" in e for e in errors)


def test_validate_curriculum_config_detects_monotonicity_violation():
    """Validation fails when monotonicity invariants are violated."""
    config = {
        "version": 2,
        "systems": {
            "pl": {
                "description": "test",
                "invariants": {"monotonic_axes": ["atoms"]},
                "slices": [
                    {"name": "s1", "params": {"atoms": 5}, "gates": _stub_gates()},
                    {"name": "s2", "params": {"atoms": 4}, "gates": _stub_gates()},  # violation
                ],
            }
        },
    }
    errors = validate_curriculum_config(config)
    assert any("monotonicity" in e.lower() for e in errors)


def _stub_gates() -> dict:
    """Minimal gate structure for validation tests."""
    return {
        "coverage": {"ci_lower_min": 0.9, "sample_min": 10},
        "abstention": {"max_rate_pct": 20, "max_mass": 500},
        "velocity": {"min_pph": 100, "stability_cv_max": 0.1, "window_minutes": 60},
        "caps": {"min_attempt_mass": 1000, "min_runtime_minutes": 10, "backlog_max": 0.4},
    }


# ---------------------------------------------------------------------------
# FOL Slice Placeholder Tests (Wave 2)
# ---------------------------------------------------------------------------


def test_fol_entry_slice_has_expected_structure():
    """FOL entry slice has valid gates and params."""
    fol_slice = make_fol_entry_slice()

    assert fol_slice.name == "fol-entry"
    assert fol_slice.params.get("quantifier_depth") == 2
    assert fol_slice.gates.coverage.ci_lower_min == 0.88
    assert fol_slice.gates.abstention.max_rate_pct == 25.0
    assert fol_slice.metadata.get("wave") == 2


def test_fol_equality_slice_has_expected_structure():
    """FOL equality slice has valid gates and params."""
    fol_eq_slice = make_fol_equality_slice()

    assert fol_eq_slice.name == "fol-equality"
    assert fol_eq_slice.params.get("equality_chains_max") == 6
    assert fol_eq_slice.gates.velocity.min_pph == 60.0
    assert fol_eq_slice.metadata.get("ladder_position") == "fol_equality"


def test_fol_slices_form_valid_curriculum_system():
    """FOL slices can be assembled into a valid CurriculumSystem."""
    fol_system = CurriculumSystem(
        slug="fol",
        description="First-Order Logic curriculum",
        slices=[make_fol_entry_slice(), make_fol_equality_slice()],
        active_index=0,
        monotonic_axes=("quantifier_depth",),
        version=2,
    )

    assert fol_system.active_name == "fol-entry"
    assert fol_system.next_slice().name == "fol-equality"
