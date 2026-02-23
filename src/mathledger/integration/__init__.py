"""Integration-layer primitives for Waves A5-A7."""

from importlib import import_module

from mathledger.integration.aak_bridge import (
    AAKBridgeViolation,
    build_aak_bridge_packet,
    build_lane_b_psych_context,
    validate_lane_b_reference,
)
from mathledger.integration.cpf_adapter import (
    CPFAdapterViolation,
    cpf_snapshot_json,
    normalize_cpf_assessments,
)
from mathledger.integration.experiment_runner import (
    ExperimentRunnerViolation,
    assert_lane_boundaries,
    run_three_lane_experiment,
)

_SCENARIO_SUITE_EXPORTS = {
    "SCENARIO_SUITE_VERSION",
    "SILICON_PSYCHE_SOURCE_PATH",
    "INTEGRATION_PAPER_SOURCE_PATH",
    "ScenarioSuiteViolation",
    "scenario_catalog",
    "adversarial_scenarios",
    "benign_control_scenarios",
    "run_scenario_suite",
    "run_default_scenario_suite",
}


def __getattr__(name: str):
    if name in _SCENARIO_SUITE_EXPORTS:
        module = import_module("mathledger.integration.scenario_suite")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AAKBridgeViolation",
    "build_lane_b_psych_context",
    "build_aak_bridge_packet",
    "validate_lane_b_reference",
    "CPFAdapterViolation",
    "normalize_cpf_assessments",
    "cpf_snapshot_json",
    "ExperimentRunnerViolation",
    "run_three_lane_experiment",
    "assert_lane_boundaries",
    "SCENARIO_SUITE_VERSION",
    "SILICON_PSYCHE_SOURCE_PATH",
    "INTEGRATION_PAPER_SOURCE_PATH",
    "ScenarioSuiteViolation",
    "scenario_catalog",
    "adversarial_scenarios",
    "benign_control_scenarios",
    "run_scenario_suite",
    "run_default_scenario_suite",
]
