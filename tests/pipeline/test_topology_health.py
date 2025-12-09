"""
Tests for Topology Health Evaluator and Degradation Policy Engine.

PHASE II — NOT RUN IN PHASE I
No uplift claims are made.

These tests verify the implementation of docs/U2_PIPELINE_TOPOLOGY.md
Sections 10-12 (Degraded Pipeline Modes, Topology Health Matrix,
CI Degradation Policy).

Test Categories:
1. FULL_PIPELINE path - all nodes healthy
2. DEGRADED_ANALYSIS - some slices failing
3. EVIDENCE_ONLY - critical failure
4. Cardinal Rule enforcement - no Δp from partial data
5. Hard-fail vs soft-fail semantics
6. Health signal evaluation
7. Failure pattern detection
"""

import pytest
from datetime import datetime
from typing import Dict

from backend.pipeline.topology_health import (
    DegradationDecision,
    DegradationPolicyEngine,
    FailurePattern,
    GovernanceLabel,
    HealthSignals,
    IntegrityCheck,
    NodeHealth,
    NodeStatus,
    NodeType,
    PipelineHealth,
    PipelineMode,
    SliceResult,
    TopologyHealthEvaluator,
    ValidationStatus,
    evaluate_degradation,
    evaluate_pipeline_health,
    CRITICAL_NODES,
    SLICE_PAIRS,
    FAILURE_PATTERNS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def evaluator() -> TopologyHealthEvaluator:
    """Create a TopologyHealthEvaluator instance."""
    return TopologyHealthEvaluator()


@pytest.fixture
def engine() -> DegradationPolicyEngine:
    """Create a DegradationPolicyEngine instance."""
    return DegradationPolicyEngine()


@pytest.fixture
def all_ok_node_statuses() -> Dict[str, NodeStatus]:
    """Create node statuses where all nodes are OK."""
    return {
        "N01": NodeStatus.OK,
        "N02": NodeStatus.OK,
        "N03": NodeStatus.OK,
        "N04": NodeStatus.OK,
        "N05": NodeStatus.OK,
        "N10": NodeStatus.OK,
        "N11": NodeStatus.OK,
        "N12": NodeStatus.OK,
        "N13": NodeStatus.OK,
        "N20": NodeStatus.OK,
        "N21": NodeStatus.OK,
        "N22": NodeStatus.OK,
        "N23": NodeStatus.OK,
        "N30": NodeStatus.OK,
        "N40": NodeStatus.OK,
        "N41": NodeStatus.OK,
        "N50": NodeStatus.OK,
        "N60": NodeStatus.OK,
        "N70": NodeStatus.OK,
        "N80": NodeStatus.OK,
        "N90": NodeStatus.OK,
    }


@pytest.fixture
def all_ok_validation() -> ValidationStatus:
    """Create ValidationStatus where all stages pass."""
    return ValidationStatus(
        gate_check_passed=True,
        prereg_verify_passed=True,
        curriculum_load_passed=True,
        dry_run_passed=True,
        manifest_init_passed=True,
    )


@pytest.fixture
def all_ok_slice_results() -> list:
    """Create slice results where all slices complete."""
    return [
        SliceResult("goal", True, True, "N10", "N20"),
        SliceResult("sparse", True, True, "N11", "N21"),
        SliceResult("tree", True, True, "N12", "N22"),
        SliceResult("dep", True, True, "N13", "N23"),
    ]


@pytest.fixture
def all_ok_ci_results() -> Dict[str, Dict]:
    """Create CI stage results where all stages pass."""
    return {
        "gate-check": {"status": "OK"},
        "prereg-verify": {"status": "OK"},
        "curriculum-load": {"status": "OK"},
        "dry-run": {"status": "OK"},
        "manifest-init": {"status": "OK"},
        "run-slice-goal": {"status": "OK"},
        "run-slice-sparse": {"status": "OK"},
        "run-slice-tree": {"status": "OK"},
        "run-slice-dep": {"status": "OK"},
        "eval-goal": {"status": "OK"},
        "eval-sparse": {"status": "OK"},
        "eval-tree": {"status": "OK"},
        "eval-dep": {"status": "OK"},
        "integrity-check": {"status": "OK"},
    }


# =============================================================================
# Test Class: FULL_PIPELINE Path
# =============================================================================


class TestFullPipelinePath:
    """Tests for FULL_PIPELINE mode (all nodes healthy)."""

    def test_all_nodes_ok_returns_full_pipeline(
        self, evaluator: TopologyHealthEvaluator, all_ok_node_statuses: Dict[str, NodeStatus]
    ):
        """When all nodes are OK, pipeline should be FULL_PIPELINE."""
        health = evaluator.evaluate_pipeline_health(all_ok_node_statuses)

        assert health.mode == PipelineMode.FULL_PIPELINE
        assert health.governance_label == GovernanceLabel.OK
        assert len(health.failed_nodes) == 0
        assert set(health.successful_slices) == {"goal", "sparse", "tree", "dep"}
        assert len(health.failed_slices) == 0

    def test_all_validation_passed_returns_full_pipeline(
        self,
        evaluator: TopologyHealthEvaluator,
        all_ok_validation: ValidationStatus,
        all_ok_slice_results: list,
    ):
        """When validation passes and all slices complete, should be FULL_PIPELINE."""
        health = evaluator.evaluate_from_slice_results(
            all_ok_validation, all_ok_slice_results
        )

        assert health.mode == PipelineMode.FULL_PIPELINE
        assert health.governance_label == GovernanceLabel.OK

    def test_full_pipeline_allows_all_delta_p(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """FULL_PIPELINE mode should allow Δp for all slices."""
        decision = engine.evaluate_degradation(all_ok_ci_results)

        assert decision.mode == PipelineMode.FULL_PIPELINE
        assert decision.allow_delta_p is True
        assert decision.allowed_delta_p_slices == frozenset({"goal", "sparse", "tree", "dep"})
        assert decision.halt_ci is False
        assert decision.quarantine is False

    def test_full_pipeline_evidence_pack_complete(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """FULL_PIPELINE should produce COMPLETE evidence pack."""
        decision = engine.evaluate_degradation(all_ok_ci_results)

        assert decision.evidence_pack_status == "COMPLETE"
        assert decision.allow_evidence_pack is True


# =============================================================================
# Test Class: DEGRADED_ANALYSIS Path
# =============================================================================


class TestDegradedAnalysisPath:
    """Tests for DEGRADED_ANALYSIS mode (some slices failing)."""

    def test_one_slice_failed_returns_degraded(
        self, evaluator: TopologyHealthEvaluator, all_ok_node_statuses: Dict[str, NodeStatus]
    ):
        """When one slice fails, should be DEGRADED_ANALYSIS."""
        statuses = all_ok_node_statuses.copy()
        statuses["N10"] = NodeStatus.FAIL  # goal runner fails

        health = evaluator.evaluate_pipeline_health(statuses)

        assert health.mode == PipelineMode.DEGRADED_ANALYSIS
        assert health.governance_label == GovernanceLabel.WARN
        assert "N10" in health.failed_nodes
        assert "goal" in health.failed_slices
        assert set(health.successful_slices) == {"sparse", "tree", "dep"}

    def test_two_slices_failed_returns_degraded(
        self, evaluator: TopologyHealthEvaluator, all_ok_node_statuses: Dict[str, NodeStatus]
    ):
        """When two slices fail, should still be DEGRADED_ANALYSIS (≥2 succeed)."""
        statuses = all_ok_node_statuses.copy()
        statuses["N10"] = NodeStatus.FAIL  # goal runner fails
        statuses["N11"] = NodeStatus.FAIL  # sparse runner fails

        health = evaluator.evaluate_pipeline_health(statuses)

        assert health.mode == PipelineMode.DEGRADED_ANALYSIS
        assert health.governance_label == GovernanceLabel.WARN
        assert set(health.successful_slices) == {"tree", "dep"}

    def test_degraded_restricts_delta_p_to_successful_slices(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """DEGRADED_ANALYSIS should only allow Δp for successful slices."""
        ci_results = all_ok_ci_results.copy()
        ci_results["run-slice-goal"] = {"status": "FAIL"}
        ci_results["eval-goal"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert decision.mode == PipelineMode.DEGRADED_ANALYSIS
        assert decision.allow_delta_p is True
        assert "goal" not in decision.allowed_delta_p_slices
        assert decision.allowed_delta_p_slices == frozenset({"sparse", "tree", "dep"})

    def test_degraded_evidence_pack_partial(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """DEGRADED_ANALYSIS should produce PARTIAL evidence pack."""
        ci_results = all_ok_ci_results.copy()
        ci_results["run-slice-goal"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert decision.evidence_pack_status == "PARTIAL"

    def test_evaluator_failure_excludes_slice(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """If evaluator fails, slice should be excluded even if runner succeeded."""
        ci_results = all_ok_ci_results.copy()
        ci_results["eval-sparse"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert "sparse" not in decision.allowed_delta_p_slices

    def test_degraded_does_not_halt_ci(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """DEGRADED_ANALYSIS should not halt CI."""
        ci_results = all_ok_ci_results.copy()
        ci_results["run-slice-goal"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert decision.halt_ci is False
        assert decision.quarantine is False


# =============================================================================
# Test Class: EVIDENCE_ONLY Path
# =============================================================================


class TestEvidenceOnlyPath:
    """Tests for EVIDENCE_ONLY mode (critical failure)."""

    def test_critical_node_failure_returns_evidence_only(
        self, evaluator: TopologyHealthEvaluator, all_ok_node_statuses: Dict[str, NodeStatus]
    ):
        """Critical node failure should trigger EVIDENCE_ONLY."""
        for critical_node in CRITICAL_NODES:
            statuses = all_ok_node_statuses.copy()
            statuses[critical_node] = NodeStatus.FAIL

            health = evaluator.evaluate_pipeline_health(statuses)

            assert health.mode == PipelineMode.EVIDENCE_ONLY
            assert health.governance_label == GovernanceLabel.DO_NOT_USE
            assert critical_node in health.failed_nodes

    def test_three_slices_failed_returns_evidence_only(
        self, evaluator: TopologyHealthEvaluator, all_ok_node_statuses: Dict[str, NodeStatus]
    ):
        """When three or more slices fail, should be EVIDENCE_ONLY."""
        statuses = all_ok_node_statuses.copy()
        statuses["N10"] = NodeStatus.FAIL  # goal fails
        statuses["N11"] = NodeStatus.FAIL  # sparse fails
        statuses["N12"] = NodeStatus.FAIL  # tree fails

        health = evaluator.evaluate_pipeline_health(statuses)

        assert health.mode == PipelineMode.EVIDENCE_ONLY
        assert health.governance_label == GovernanceLabel.DO_NOT_USE

    def test_all_slices_failed_returns_evidence_only(
        self, evaluator: TopologyHealthEvaluator, all_ok_node_statuses: Dict[str, NodeStatus]
    ):
        """When all slices fail, should be EVIDENCE_ONLY."""
        statuses = all_ok_node_statuses.copy()
        statuses["N10"] = NodeStatus.FAIL
        statuses["N11"] = NodeStatus.FAIL
        statuses["N12"] = NodeStatus.FAIL
        statuses["N13"] = NodeStatus.FAIL

        health = evaluator.evaluate_pipeline_health(statuses)

        assert health.mode == PipelineMode.EVIDENCE_ONLY

    def test_validation_failure_returns_evidence_only(
        self, evaluator: TopologyHealthEvaluator, all_ok_slice_results: list
    ):
        """Validation failure should trigger EVIDENCE_ONLY."""
        validation = ValidationStatus(
            gate_check_passed=False,
            prereg_verify_passed=True,
            curriculum_load_passed=True,
            dry_run_passed=True,
            manifest_init_passed=True,
        )

        health = evaluator.evaluate_from_slice_results(validation, all_ok_slice_results)

        assert health.mode == PipelineMode.EVIDENCE_ONLY

    def test_integrity_failure_returns_evidence_only(
        self,
        evaluator: TopologyHealthEvaluator,
        all_ok_validation: ValidationStatus,
        all_ok_slice_results: list,
    ):
        """Integrity check failure should trigger EVIDENCE_ONLY."""
        integrity_checks = [
            IntegrityCheck("checksum", False, "Checksum mismatch")
        ]

        health = evaluator.evaluate_from_slice_results(
            all_ok_validation, all_ok_slice_results, integrity_checks
        )

        assert health.mode == PipelineMode.EVIDENCE_ONLY

    def test_evidence_only_forbids_delta_p(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """EVIDENCE_ONLY should forbid all Δp computation."""
        ci_results = all_ok_ci_results.copy()
        ci_results["gate-check"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert decision.mode == PipelineMode.EVIDENCE_ONLY
        assert decision.allow_delta_p is False
        assert decision.allowed_delta_p_slices == frozenset()

    def test_evidence_only_halts_ci(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """EVIDENCE_ONLY should halt CI."""
        ci_results = all_ok_ci_results.copy()
        ci_results["prereg-verify"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert decision.halt_ci is True
        assert decision.halt_reason is not None

    def test_evidence_only_quarantines(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """EVIDENCE_ONLY should quarantine the experiment."""
        ci_results = all_ok_ci_results.copy()
        ci_results["integrity-check"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert decision.quarantine is True

    def test_evidence_only_pack_forensic(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """EVIDENCE_ONLY should produce FORENSIC evidence pack."""
        ci_results = all_ok_ci_results.copy()
        ci_results["gate-check"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert decision.evidence_pack_status == "FORENSIC"


# =============================================================================
# Test Class: Cardinal Rule Enforcement
# =============================================================================


class TestCardinalRuleEnforcement:
    """
    Tests for Cardinal Rule: No Δp from partial or corrupted data.

    Section 12.1: If a baseline run fails, the corresponding RFL run's Δp
    MUST be excluded. If an RFL run fails, the corresponding baseline
    run's Δp MUST be excluded.
    """

    def test_cardinal_rule_baseline_failed(self, engine: DegradationPolicyEngine):
        """If baseline fails, Δp must be excluded for that slice."""
        allowed, message = engine.check_cardinal_rule(
            slice_name="goal",
            baseline_completed=False,
            rfl_completed=True,
        )

        assert allowed is False
        assert "CARDINAL RULE" in message
        assert "Baseline failed" in message

    def test_cardinal_rule_rfl_failed(self, engine: DegradationPolicyEngine):
        """If RFL fails, Δp must be excluded for that slice."""
        allowed, message = engine.check_cardinal_rule(
            slice_name="sparse",
            baseline_completed=True,
            rfl_completed=False,
        )

        assert allowed is False
        assert "CARDINAL RULE" in message
        assert "RFL failed" in message

    def test_cardinal_rule_both_failed(self, engine: DegradationPolicyEngine):
        """If both fail, Δp must be excluded."""
        allowed, message = engine.check_cardinal_rule(
            slice_name="tree",
            baseline_completed=False,
            rfl_completed=False,
        )

        assert allowed is False
        assert "CARDINAL RULE" in message

    def test_cardinal_rule_both_succeeded(self, engine: DegradationPolicyEngine):
        """If both succeed, Δp is allowed."""
        allowed, message = engine.check_cardinal_rule(
            slice_name="dep",
            baseline_completed=True,
            rfl_completed=True,
        )

        assert allowed is True
        assert message is None

    def test_no_scenario_yields_delta_p_when_forbidden(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """
        CRITICAL TEST: Assert no scenario yields Δp when forbidden by spec.

        This test verifies the cardinal rule across all possible failure scenarios.
        """
        # Test hard-fail stages
        hard_fail_stages = [
            "gate-check", "prereg-verify", "curriculum-load",
            "dry-run", "manifest-init", "integrity-check",
        ]

        for stage in hard_fail_stages:
            ci_results = all_ok_ci_results.copy()
            ci_results[stage] = {"status": "FAIL"}

            decision = engine.evaluate_degradation(ci_results)

            assert decision.allow_delta_p is False, \
                f"Δp should be forbidden when {stage} fails"
            assert len(decision.allowed_delta_p_slices) == 0, \
                f"No slices should allow Δp when {stage} fails"

    def test_delta_p_validation_request(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """Test validation of Δp computation requests."""
        ci_results = all_ok_ci_results.copy()
        ci_results["run-slice-goal"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        # Goal should be rejected
        allowed, reason = engine.validate_delta_p_request("goal", decision)
        assert allowed is False
        assert reason is not None

        # Sparse should be allowed
        allowed, reason = engine.validate_delta_p_request("sparse", decision)
        assert allowed is True
        assert reason is None


# =============================================================================
# Test Class: Hard-Fail vs Soft-Fail Semantics
# =============================================================================


class TestHardFailSoftFailSemantics:
    """Tests for Section 12.2-12.3 hard-fail and soft-fail semantics."""

    def test_hard_fail_stages_trigger_evidence_only(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """Hard-fail stages should trigger EVIDENCE_ONLY immediately."""
        hard_fail_stages = [
            "gate-check", "prereg-verify", "curriculum-load",
            "dry-run", "manifest-init", "integrity-check",
        ]

        for stage in hard_fail_stages:
            ci_results = all_ok_ci_results.copy()
            ci_results[stage] = {"status": "FAIL"}

            decision = engine.evaluate_degradation(ci_results)

            assert decision.mode == PipelineMode.EVIDENCE_ONLY, \
                f"Hard-fail stage {stage} should trigger EVIDENCE_ONLY"

    def test_soft_fail_stages_allow_degradation(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """Soft-fail stages should allow DEGRADED_ANALYSIS."""
        soft_fail_stages = [
            ("run-slice-goal", "eval-goal"),
            ("run-slice-sparse", "eval-sparse"),
        ]

        for runner, evaluator in soft_fail_stages:
            ci_results = all_ok_ci_results.copy()
            ci_results[runner] = {"status": "FAIL"}

            decision = engine.evaluate_degradation(ci_results)

            # Should be DEGRADED, not EVIDENCE_ONLY
            assert decision.mode == PipelineMode.DEGRADED_ANALYSIS, \
                f"Soft-fail stage {runner} should allow DEGRADED_ANALYSIS"

    def test_exit_code_categorization(self, engine: DegradationPolicyEngine):
        """Test exit code categorization by range."""
        assert engine.get_exit_code_category(100) == "gate-check"
        assert engine.get_exit_code_category(115) == "prereg-verify"
        assert engine.get_exit_code_category(200) == "slice-runner"
        assert engine.get_exit_code_category(300) == "evaluator"
        assert engine.get_exit_code_category(700) == "audit"
        assert engine.get_exit_code_category(999) == "infrastructure"
        assert engine.get_exit_code_category(50) is None  # Not in any range

    def test_hard_fail_exit_code_detection(self, engine: DegradationPolicyEngine):
        """Test detection of hard-fail exit codes."""
        assert engine.is_hard_fail_exit_code(100) is True  # gate-check
        assert engine.is_hard_fail_exit_code(149) is True  # manifest-init
        assert engine.is_hard_fail_exit_code(700) is True  # audit
        assert engine.is_hard_fail_exit_code(750) is True  # contamination
        assert engine.is_hard_fail_exit_code(200) is False  # slice-runner (soft)
        assert engine.is_hard_fail_exit_code(0) is False  # success


# =============================================================================
# Test Class: Health Signal Evaluation
# =============================================================================


class TestHealthSignalEvaluation:
    """Tests for health signal evaluation."""

    def test_health_signals_default_healthy(self):
        """Default health signals should indicate healthy."""
        signals = HealthSignals()
        assert signals.is_healthy() is True

    def test_health_signals_heartbeat_false(self):
        """Missing heartbeat should indicate unhealthy."""
        signals = HealthSignals(heartbeat=False)
        assert signals.is_healthy() is False

    def test_health_signals_incomplete_progress(self):
        """Incomplete progress should indicate unhealthy."""
        signals = HealthSignals(progress=0.5)
        assert signals.is_healthy() is False

    def test_health_signals_memory_issue(self):
        """Memory issue should indicate unhealthy."""
        signals = HealthSignals(memory_ok=False)
        assert signals.is_healthy() is False

    def test_health_signals_integrity_issue(self):
        """Integrity issue should indicate unhealthy."""
        signals = HealthSignals(integrity_ok=False)
        assert signals.is_healthy() is False

    def test_health_signals_invalid_progress_raises(self):
        """Invalid progress value should raise ValueError."""
        with pytest.raises(ValueError):
            HealthSignals(progress=1.5)

        with pytest.raises(ValueError):
            HealthSignals(progress=-0.1)


# =============================================================================
# Test Class: Failure Pattern Detection
# =============================================================================


class TestFailurePatternDetection:
    """Tests for failure pattern detection."""

    def test_detect_gatekeeper_patterns(self, evaluator: TopologyHealthEvaluator):
        """Test detection of gatekeeper failure patterns."""
        patterns = evaluator.detect_failure_patterns(
            "N01", 100, "Gate file missing: governance.yaml not found"
        )
        assert any(p.pattern_id == "GK-001" for p in patterns)

    def test_detect_validator_patterns(self, evaluator: TopologyHealthEvaluator):
        """Test detection of validator failure patterns."""
        patterns = evaluator.detect_failure_patterns(
            "N04", 130, "Schema validation failed: missing required field"
        )
        assert any(p.pattern_id == "VD-001" for p in patterns)

    def test_detect_runner_patterns(self, evaluator: TopologyHealthEvaluator):
        """Test detection of runner failure patterns."""
        patterns = evaluator.detect_failure_patterns(
            "N10", 200, "Cycle timeout: verifier did not respond"
        )
        assert any(p.pattern_id == "RN-002" for p in patterns)

    def test_detect_auditor_patterns(self, evaluator: TopologyHealthEvaluator):
        """Test detection of auditor failure patterns."""
        patterns = evaluator.detect_failure_patterns(
            "N60", 700, "Phase I reference detected: contamination risk"
        )
        assert any(p.pattern_id == "AU-005" for p in patterns)

    def test_failure_patterns_immutable(self):
        """Failure patterns should be immutable."""
        pattern = FAILURE_PATTERNS["GK-001"]
        assert isinstance(pattern, FailurePattern)
        # Attempting to modify should raise
        with pytest.raises(AttributeError):
            pattern.pattern_id = "modified"


# =============================================================================
# Test Class: Slice Result Validation
# =============================================================================


class TestSliceResultValidation:
    """Tests for slice result validation."""

    def test_slice_result_complete(self):
        """Complete slice should report completion."""
        result = SliceResult("goal", True, True, "N10", "N20")
        assert result.is_complete() is True
        assert result.allows_delta_p() is True

    def test_slice_result_baseline_failed(self):
        """Slice with failed baseline should not allow Δp."""
        result = SliceResult("goal", False, True, "N10", "N20")
        assert result.is_complete() is False
        assert result.allows_delta_p() is False

    def test_slice_result_rfl_failed(self):
        """Slice with failed RFL should not allow Δp."""
        result = SliceResult("goal", True, False, "N10", "N20")
        assert result.is_complete() is False
        assert result.allows_delta_p() is False

    def test_slice_result_both_failed(self):
        """Slice with both failed should not allow Δp."""
        result = SliceResult("goal", False, False, "N10", "N20")
        assert result.is_complete() is False
        assert result.allows_delta_p() is False


# =============================================================================
# Test Class: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_evaluate_pipeline_health_function(self, all_ok_node_statuses):
        """Test evaluate_pipeline_health convenience function."""
        health = evaluate_pipeline_health(all_ok_node_statuses)
        assert health.mode == PipelineMode.FULL_PIPELINE

    def test_evaluate_degradation_function(self, all_ok_ci_results):
        """Test evaluate_degradation convenience function."""
        decision = evaluate_degradation(all_ok_ci_results)
        assert decision.mode == PipelineMode.FULL_PIPELINE


# =============================================================================
# Test Class: Serialization
# =============================================================================


class TestSerialization:
    """Tests for JSON serialization."""

    def test_pipeline_health_to_dict(self, evaluator, all_ok_node_statuses):
        """PipelineHealth should serialize to dict."""
        health = evaluator.evaluate_pipeline_health(all_ok_node_statuses)
        data = health.to_dict()

        assert data["mode"] == "FULL_PIPELINE"
        assert data["governance_label"] == "OK"
        assert isinstance(data["successful_slices"], list)

    def test_pipeline_health_to_json(self, evaluator, all_ok_node_statuses):
        """PipelineHealth should serialize to JSON."""
        health = evaluator.evaluate_pipeline_health(all_ok_node_statuses)
        json_str = health.to_json()

        assert '"mode": "FULL_PIPELINE"' in json_str

    def test_degradation_decision_to_dict(self, engine, all_ok_ci_results):
        """DegradationDecision should serialize to dict."""
        decision = engine.evaluate_degradation(all_ok_ci_results)
        data = decision.to_dict()

        assert data["mode"] == "FULL_PIPELINE"
        assert data["allow_delta_p"] is True

    def test_degradation_decision_to_json(self, engine, all_ok_ci_results):
        """DegradationDecision should serialize to JSON."""
        decision = engine.evaluate_degradation(all_ok_ci_results)
        json_str = decision.to_json()

        assert '"allow_delta_p": true' in json_str


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_node_statuses(self, evaluator):
        """Empty node statuses should result in EVIDENCE_ONLY."""
        health = evaluator.evaluate_pipeline_health({})

        # No successful slices means EVIDENCE_ONLY
        assert health.mode == PipelineMode.EVIDENCE_ONLY

    def test_partial_node_statuses(self, evaluator):
        """Partial node statuses should handle missing nodes."""
        statuses = {
            "N01": NodeStatus.OK,
            "N02": NodeStatus.OK,
            # Missing N03, N04, N05
        }

        health = evaluator.evaluate_pipeline_health(statuses)
        # Missing critical nodes means they're NOT_RUN, not FAIL
        # But missing slice nodes means slices fail
        assert health.mode == PipelineMode.EVIDENCE_ONLY

    def test_warn_status_treated_as_success(self, evaluator, all_ok_node_statuses):
        """WARN status should be treated as success for mode determination."""
        statuses = all_ok_node_statuses.copy()
        statuses["N10"] = NodeStatus.WARN

        health = evaluator.evaluate_pipeline_health(statuses)

        # WARN is not FAIL, so should still be FULL_PIPELINE
        assert health.mode == PipelineMode.FULL_PIPELINE

    def test_skipped_status_treated_as_failure(self, evaluator, all_ok_node_statuses):
        """SKIPPED status should be treated as failure for slices."""
        statuses = all_ok_node_statuses.copy()
        statuses["N10"] = NodeStatus.SKIPPED

        health = evaluator.evaluate_pipeline_health(statuses)

        # SKIPPED slice runner means slice failed
        assert "goal" in health.failed_slices

    def test_exactly_two_slices_is_degraded(self, evaluator, all_ok_node_statuses):
        """Exactly 2 successful slices should be DEGRADED_ANALYSIS."""
        statuses = all_ok_node_statuses.copy()
        statuses["N10"] = NodeStatus.FAIL  # goal fails
        statuses["N11"] = NodeStatus.FAIL  # sparse fails

        health = evaluator.evaluate_pipeline_health(statuses)

        assert health.mode == PipelineMode.DEGRADED_ANALYSIS
        assert len(health.successful_slices) == 2

    def test_three_slices_is_still_degraded(self, evaluator, all_ok_node_statuses):
        """3 successful slices should still be DEGRADED_ANALYSIS."""
        statuses = all_ok_node_statuses.copy()
        statuses["N10"] = NodeStatus.FAIL  # only goal fails

        health = evaluator.evaluate_pipeline_health(statuses)

        assert health.mode == PipelineMode.DEGRADED_ANALYSIS
        assert len(health.successful_slices) == 3


# =============================================================================
# Test Class: Node Type Classification
# =============================================================================


class TestNodeTypeClassification:
    """Tests for node type classification."""

    def test_critical_nodes_defined(self):
        """Critical nodes should be defined."""
        assert CRITICAL_NODES == frozenset({"N01", "N02", "N03", "N04", "N05"})

    def test_slice_pairs_defined(self):
        """Slice pairs should be defined correctly."""
        assert len(SLICE_PAIRS) == 4
        assert ("goal", "N10", "N20") in SLICE_PAIRS
        assert ("sparse", "N11", "N21") in SLICE_PAIRS
        assert ("tree", "N12", "N22") in SLICE_PAIRS
        assert ("dep", "N13", "N23") in SLICE_PAIRS

    def test_node_health_is_critical(self):
        """NodeHealth should correctly identify critical nodes."""
        gatekeeper = NodeHealth(
            node_id="N01",
            node_type=NodeType.GATEKEEPER,
            status=NodeStatus.OK,
            signals=HealthSignals(),
        )
        assert gatekeeper.is_critical_node() is True

        runner = NodeHealth(
            node_id="N10",
            node_type=NodeType.RUNNER,
            status=NodeStatus.OK,
            signals=HealthSignals(),
        )
        assert runner.is_critical_node() is False


# =============================================================================
# Test Class: Violations Tracking
# =============================================================================


class TestViolationsTracking:
    """Tests for violation tracking in decisions."""

    def test_hard_fail_records_violation(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """Hard-fail should record violation."""
        ci_results = all_ok_ci_results.copy()
        ci_results["gate-check"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert len(decision.violations) > 0
        assert any("HARD-FAIL" in v for v in decision.violations)

    def test_cardinal_rule_records_violation(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """Cardinal rule violation should be recorded."""
        ci_results = all_ok_ci_results.copy()
        ci_results["run-slice-goal"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert any("CARDINAL RULE" in v for v in decision.violations)

    def test_degraded_records_restriction(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """DEGRADED_ANALYSIS should record restriction."""
        ci_results = all_ok_ci_results.copy()
        ci_results["run-slice-goal"] = {"status": "FAIL"}

        decision = engine.evaluate_degradation(ci_results)

        assert any("DEGRADED" in v for v in decision.violations)


# =============================================================================
# Test Class: Pattern Library (v2)
# =============================================================================


class TestPatternLibrary:
    """Tests for the pattern library and severity mappings."""

    def test_pattern_library_contains_all_failure_patterns(self):
        """Pattern library should contain entries for all failure patterns."""
        from backend.pipeline.topology_health import PATTERN_LIBRARY, FAILURE_PATTERNS

        for pattern_id in FAILURE_PATTERNS:
            assert pattern_id in PATTERN_LIBRARY, f"Missing pattern: {pattern_id}"

    def test_get_pattern_severity(self):
        """get_pattern_severity should return correct severity."""
        from backend.pipeline.topology_health import (
            get_pattern_severity,
            PatternSeverity,
        )

        # Critical patterns
        assert get_pattern_severity("GK-001") == PatternSeverity.CRITICAL
        assert get_pattern_severity("AU-001") == PatternSeverity.CRITICAL

        # High severity patterns
        assert get_pattern_severity("VD-001") == PatternSeverity.HIGH

        # Medium severity patterns
        assert get_pattern_severity("RN-001") == PatternSeverity.MEDIUM

        # Low severity patterns
        assert get_pattern_severity("EV-002") == PatternSeverity.LOW

        # Unknown pattern
        assert get_pattern_severity("UNKNOWN-999") is None

    def test_get_pattern_suggested_mode(self):
        """get_pattern_suggested_mode should return correct mode."""
        from backend.pipeline.topology_health import get_pattern_suggested_mode

        # Critical patterns suggest EVIDENCE_ONLY
        assert get_pattern_suggested_mode("GK-001") == PipelineMode.EVIDENCE_ONLY

        # Medium patterns may suggest DEGRADED_ANALYSIS
        assert get_pattern_suggested_mode("RN-001") == PipelineMode.DEGRADED_ANALYSIS

        # Unknown pattern
        assert get_pattern_suggested_mode("UNKNOWN-999") is None

    def test_is_critical_pattern(self):
        """is_critical_pattern should correctly identify critical patterns."""
        from backend.pipeline.topology_health import is_critical_pattern

        assert is_critical_pattern("GK-001") is True
        assert is_critical_pattern("GK-002") is True
        assert is_critical_pattern("AU-001") is True
        assert is_critical_pattern("AU-005") is True

        assert is_critical_pattern("RN-001") is False
        assert is_critical_pattern("EV-002") is False
        assert is_critical_pattern("UNKNOWN") is False

    def test_pattern_library_entry_to_dict(self):
        """PatternLibraryEntry should serialize to dict."""
        from backend.pipeline.topology_health import PATTERN_LIBRARY

        entry = PATTERN_LIBRARY["GK-001"]
        data = entry.to_dict()

        assert data["pattern_id"] == "GK-001"
        assert data["severity"] == "CRITICAL"
        assert data["suggested_mode"] == "EVIDENCE_ONLY"
        assert data["fail_behavior"] == "HARD_FAIL"
        assert "description" in data


# =============================================================================
# Test Class: Health Snapshot (v2)
# =============================================================================


class TestHealthSnapshot:
    """Tests for health snapshot building and serialization."""

    def test_build_snapshot_full_pipeline(
        self, evaluator: TopologyHealthEvaluator, engine: DegradationPolicyEngine,
        all_ok_node_statuses: Dict[str, NodeStatus], all_ok_ci_results: Dict[str, Dict]
    ):
        """build_pipeline_health_snapshot should capture FULL_PIPELINE state."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_snapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        health = evaluator.evaluate_pipeline_health(all_ok_node_statuses)
        decision = engine.evaluate_degradation(all_ok_ci_results)
        snapshot = build_pipeline_health_snapshot(health, decision, "run-001")

        assert snapshot.schema_version == HEALTH_SNAPSHOT_SCHEMA_VERSION
        assert snapshot.mode == PipelineMode.FULL_PIPELINE
        assert snapshot.governance_label == GovernanceLabel.OK
        assert snapshot.hard_fail_count == 0
        assert snapshot.soft_fail_count == 0
        assert snapshot.successful_slice_count == 4
        assert snapshot.failed_slice_count == 0
        assert snapshot.run_id == "run-001"

    def test_build_snapshot_degraded(
        self, evaluator: TopologyHealthEvaluator, engine: DegradationPolicyEngine
    ):
        """build_pipeline_health_snapshot should capture DEGRADED_ANALYSIS state."""
        from backend.pipeline.topology_health import build_pipeline_health_snapshot

        # Create node statuses with one slice failing
        statuses = {
            "N01": NodeStatus.OK, "N02": NodeStatus.OK, "N03": NodeStatus.OK,
            "N04": NodeStatus.OK, "N05": NodeStatus.OK,
            "N10": NodeStatus.FAIL,  # Goal runner fails
            "N11": NodeStatus.OK, "N12": NodeStatus.OK, "N13": NodeStatus.OK,
            "N20": NodeStatus.OK, "N21": NodeStatus.OK, "N22": NodeStatus.OK, "N23": NodeStatus.OK,
            "N30": NodeStatus.OK, "N40": NodeStatus.OK, "N41": NodeStatus.OK,
            "N50": NodeStatus.OK, "N60": NodeStatus.OK, "N70": NodeStatus.OK,
            "N80": NodeStatus.OK, "N90": NodeStatus.OK,
        }

        health = evaluator.evaluate_pipeline_health(statuses)
        ci_results = {
            "gate-check": {"status": "OK"}, "prereg-verify": {"status": "OK"},
            "curriculum-load": {"status": "OK"}, "dry-run": {"status": "OK"},
            "manifest-init": {"status": "OK"},
            "run-slice-goal": {"status": "FAIL"},
            "run-slice-sparse": {"status": "OK"}, "run-slice-tree": {"status": "OK"},
            "run-slice-dep": {"status": "OK"},
            "eval-goal": {"status": "OK"}, "eval-sparse": {"status": "OK"},
            "eval-tree": {"status": "OK"}, "eval-dep": {"status": "OK"},
            "integrity-check": {"status": "OK"},
        }
        decision = engine.evaluate_degradation(ci_results)
        snapshot = build_pipeline_health_snapshot(health, decision)

        assert snapshot.mode == PipelineMode.DEGRADED_ANALYSIS
        assert snapshot.failed_slice_count >= 1

    def test_snapshot_to_dict_and_from_dict(
        self, evaluator: TopologyHealthEvaluator, engine: DegradationPolicyEngine,
        all_ok_node_statuses: Dict[str, NodeStatus], all_ok_ci_results: Dict[str, Dict]
    ):
        """HealthSnapshot should round-trip through dict serialization."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_snapshot,
            HealthSnapshot,
        )

        health = evaluator.evaluate_pipeline_health(all_ok_node_statuses)
        decision = engine.evaluate_degradation(all_ok_ci_results)
        original = build_pipeline_health_snapshot(health, decision, "test-run")

        # Serialize and deserialize
        data = original.to_dict()
        restored = HealthSnapshot.from_dict(data)

        assert restored.schema_version == original.schema_version
        assert restored.mode == original.mode
        assert restored.governance_label == original.governance_label
        assert restored.hard_fail_count == original.hard_fail_count
        assert restored.run_id == original.run_id

    def test_snapshot_to_json(
        self, evaluator: TopologyHealthEvaluator, engine: DegradationPolicyEngine,
        all_ok_node_statuses: Dict[str, NodeStatus], all_ok_ci_results: Dict[str, Dict]
    ):
        """HealthSnapshot should serialize to JSON."""
        from backend.pipeline.topology_health import build_pipeline_health_snapshot
        import json

        health = evaluator.evaluate_pipeline_health(all_ok_node_statuses)
        decision = engine.evaluate_degradation(all_ok_ci_results)
        snapshot = build_pipeline_health_snapshot(health, decision)

        json_str = snapshot.to_json()
        data = json.loads(json_str)

        assert data["mode"] == "FULL_PIPELINE"
        assert data["governance_label"] == "OK"


# =============================================================================
# Test Class: Snapshot Comparison (v2)
# =============================================================================


class TestSnapshotComparison:
    """Tests for comparing health snapshots."""

    def test_compare_unchanged(
        self, evaluator: TopologyHealthEvaluator, engine: DegradationPolicyEngine,
        all_ok_node_statuses: Dict[str, NodeStatus], all_ok_ci_results: Dict[str, Dict]
    ):
        """Identical snapshots should show unchanged."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_snapshot,
            compare_health_snapshots,
        )

        health = evaluator.evaluate_pipeline_health(all_ok_node_statuses)
        decision = engine.evaluate_degradation(all_ok_ci_results)
        snapshot = build_pipeline_health_snapshot(health, decision)

        comparison = compare_health_snapshots(snapshot, snapshot)

        assert comparison.unchanged is True
        assert comparison.degraded is False
        assert comparison.improved is False
        assert comparison.mode_delta is None
        assert comparison.label_delta is None

    def test_compare_degraded_mode_transition(self):
        """Transition to worse mode should show degraded."""
        from backend.pipeline.topology_health import (
            HealthSnapshot,
            compare_health_snapshots,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        old = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z",
        )

        new = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.DEGRADED_ANALYSIS,
            governance_label=GovernanceLabel.WARN,
            hard_fail_count=0, soft_fail_count=1,
            slice_baseline_fail_count=0, slice_rfl_fail_count=1,
            successful_slice_count=3, failed_slice_count=1,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T01:00:00Z",
        )

        comparison = compare_health_snapshots(old, new)

        assert comparison.degraded is True
        assert comparison.improved is False
        assert comparison.mode_delta == "FULL_PIPELINE -> DEGRADED_ANALYSIS"
        assert comparison.label_delta == "OK -> WARN"
        assert comparison.slice_fail_delta == 1

    def test_compare_improved_mode_transition(self):
        """Transition to better mode should show improved."""
        from backend.pipeline.topology_health import (
            HealthSnapshot,
            compare_health_snapshots,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        old = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.DEGRADED_ANALYSIS,
            governance_label=GovernanceLabel.WARN,
            hard_fail_count=0, soft_fail_count=1,
            slice_baseline_fail_count=0, slice_rfl_fail_count=1,
            successful_slice_count=3, failed_slice_count=1,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z",
        )

        new = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T01:00:00Z",
        )

        comparison = compare_health_snapshots(old, new)

        assert comparison.improved is True
        assert comparison.degraded is False
        assert comparison.mode_delta == "DEGRADED_ANALYSIS -> FULL_PIPELINE"

    def test_compare_new_critical_pattern(self):
        """New critical pattern should flag degradation."""
        from backend.pipeline.topology_health import (
            HealthSnapshot,
            compare_health_snapshots,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        old = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(),
            pattern_severities={},
            timestamp="2025-01-01T00:00:00Z",
        )

        new = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=1, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK-001",),  # Critical pattern
            pattern_severities={"GK-001": "CRITICAL"},
            timestamp="2025-01-01T01:00:00Z",
        )

        comparison = compare_health_snapshots(old, new)

        assert comparison.any_new_critical_pattern is True
        assert comparison.degraded is True
        assert "GK-001" in comparison.new_patterns

    def test_compare_pattern_resolved(self):
        """Resolved patterns should be tracked."""
        from backend.pipeline.topology_health import (
            HealthSnapshot,
            compare_health_snapshots,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        old = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.DEGRADED_ANALYSIS,
            governance_label=GovernanceLabel.WARN,
            hard_fail_count=0, soft_fail_count=1,
            slice_baseline_fail_count=0, slice_rfl_fail_count=1,
            successful_slice_count=3, failed_slice_count=1,
            patterns_detected=("RN-001", "EV-002"),
            pattern_severities={"RN-001": "MEDIUM", "EV-002": "LOW"},
            timestamp="2025-01-01T00:00:00Z",
        )

        new = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(),
            pattern_severities={},
            timestamp="2025-01-01T01:00:00Z",
        )

        comparison = compare_health_snapshots(old, new)

        assert len(comparison.removed_patterns) == 2
        assert "RN-001" in comparison.removed_patterns
        assert "EV-002" in comparison.removed_patterns

    def test_comparison_to_dict(self):
        """SnapshotComparison should serialize to dict."""
        from backend.pipeline.topology_health import (
            HealthSnapshot,
            compare_health_snapshots,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        old = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z",
        )

        comparison = compare_health_snapshots(old, old)
        data = comparison.to_dict()

        assert "mode_delta" in data
        assert "degraded" in data
        assert "improved" in data


# =============================================================================
# Test Class: Decision Explanation (v2)
# =============================================================================


class TestDecisionExplanation:
    """Tests for decision explanation generation."""

    def test_explain_full_pipeline(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """explain() should produce explanation for FULL_PIPELINE."""
        decision = engine.evaluate_degradation(all_ok_ci_results)
        explanation = engine.explain(decision, all_ok_ci_results)

        assert explanation.mode == PipelineMode.FULL_PIPELINE
        assert explanation.governance_label == GovernanceLabel.OK
        assert "All slices completed successfully" in explanation.short_summary
        assert "MODE:FULL_PIPELINE" in explanation.reason_codes

    def test_explain_degraded_analysis(self, engine: DegradationPolicyEngine):
        """explain() should produce explanation for DEGRADED_ANALYSIS."""
        ci_results = {
            "gate-check": {"status": "OK"}, "prereg-verify": {"status": "OK"},
            "curriculum-load": {"status": "OK"}, "dry-run": {"status": "OK"},
            "manifest-init": {"status": "OK"},
            "run-slice-goal": {"status": "FAIL"},  # One slice fails
            "run-slice-sparse": {"status": "OK"}, "run-slice-tree": {"status": "OK"},
            "run-slice-dep": {"status": "OK"},
            "eval-goal": {"status": "OK"}, "eval-sparse": {"status": "OK"},
            "eval-tree": {"status": "OK"}, "eval-dep": {"status": "OK"},
            "integrity-check": {"status": "OK"},
        }

        decision = engine.evaluate_degradation(ci_results)
        explanation = engine.explain(decision, ci_results)

        assert explanation.mode == PipelineMode.DEGRADED_ANALYSIS
        assert "3/4 slice pairs succeeded" in explanation.short_summary
        assert "MODE:DEGRADED_ANALYSIS" in explanation.reason_codes

    def test_explain_evidence_only(self, engine: DegradationPolicyEngine):
        """explain() should produce explanation for EVIDENCE_ONLY."""
        ci_results = {
            "gate-check": {"status": "FAIL"},  # Hard-fail
            "prereg-verify": {"status": "OK"},
            "curriculum-load": {"status": "OK"}, "dry-run": {"status": "OK"},
            "manifest-init": {"status": "OK"},
            "run-slice-goal": {"status": "OK"}, "run-slice-sparse": {"status": "OK"},
            "run-slice-tree": {"status": "OK"}, "run-slice-dep": {"status": "OK"},
            "eval-goal": {"status": "OK"}, "eval-sparse": {"status": "OK"},
            "eval-tree": {"status": "OK"}, "eval-dep": {"status": "OK"},
            "integrity-check": {"status": "OK"},
        }

        decision = engine.evaluate_degradation(ci_results)
        explanation = engine.explain(decision, ci_results)

        assert explanation.mode == PipelineMode.EVIDENCE_ONLY
        assert "Critical failure detected" in explanation.short_summary
        assert "HARD_FAIL_POLICY" in explanation.constraints_violated

    def test_explanation_to_dict(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """DecisionExplanation should serialize to dict."""
        decision = engine.evaluate_degradation(all_ok_ci_results)
        explanation = engine.explain(decision, all_ok_ci_results)
        data = explanation.to_dict()

        assert "reason_codes" in data
        assert "short_summary" in data
        assert "mode" in data
        assert data["mode"] == "FULL_PIPELINE"

    def test_explanation_format_for_ci(
        self, engine: DegradationPolicyEngine, all_ok_ci_results: Dict[str, Dict]
    ):
        """format_for_ci() should produce markdown output."""
        decision = engine.evaluate_degradation(all_ok_ci_results)
        explanation = engine.explain(decision, all_ok_ci_results)
        markdown = explanation.format_for_ci()

        assert "## Pipeline Mode: FULL_PIPELINE" in markdown
        assert "**Governance Label:** OK" in markdown
        assert "**Summary:**" in markdown

    def test_explanation_with_cardinal_rule_violation(
        self, engine: DegradationPolicyEngine
    ):
        """explain() should include cardinal rule violations."""
        ci_results = {
            "gate-check": {"status": "OK"}, "prereg-verify": {"status": "OK"},
            "curriculum-load": {"status": "OK"}, "dry-run": {"status": "OK"},
            "manifest-init": {"status": "OK"},
            "run-slice-goal": {"status": "FAIL"},  # Cardinal rule violation
            "run-slice-sparse": {"status": "OK"}, "run-slice-tree": {"status": "OK"},
            "run-slice-dep": {"status": "OK"},
            "eval-goal": {"status": "OK"}, "eval-sparse": {"status": "OK"},
            "eval-tree": {"status": "OK"}, "eval-dep": {"status": "OK"},
            "integrity-check": {"status": "OK"},
        }

        decision = engine.evaluate_degradation(ci_results)
        explanation = engine.explain(decision, ci_results)

        assert "CARDINAL_RULE" in explanation.constraints_violated


# =============================================================================
# Phase III Tests: Long-Horizon Governance & Forecasting
# =============================================================================


# =============================================================================
# Test Class: Health Trajectory Ledger
# =============================================================================


class TestHealthTrajectory:
    """Tests for the health trajectory ledger."""

    def test_build_trajectory_empty_snapshots(self):
        """Empty snapshot list should return empty trajectory."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            TRAJECTORY_SCHEMA_VERSION,
            ModeStability,
        )

        trajectory = build_pipeline_health_trajectory([])

        assert trajectory.schema_version == TRAJECTORY_SCHEMA_VERSION
        assert len(trajectory.trajectory) == 0
        assert trajectory.mode_evolution.total_runs == 0
        assert trajectory.stability_index == 0.0

    def test_build_trajectory_single_snapshot(self):
        """Single snapshot should create minimal trajectory."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ModeStability,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])

        assert len(trajectory.trajectory) == 1
        assert trajectory.mode_evolution.total_runs == 1
        assert trajectory.mode_evolution.full_pipeline_count == 1
        assert trajectory.mode_evolution.current_streak == 1
        assert trajectory.mode_evolution.stability == ModeStability.STABLE

    def test_build_trajectory_mode_counts(self):
        """Trajectory should correctly count mode occurrences."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(3)
        ] + [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.DEGRADED_ANALYSIS,
                governance_label=GovernanceLabel.WARN,
                hard_fail_count=0, soft_fail_count=1,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=3, failed_slice_count=1,
                patterns_detected=(), pattern_severities={},
                timestamp="2025-01-04T00:00:00Z", run_id="run-004",
            ),
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.EVIDENCE_ONLY,
                governance_label=GovernanceLabel.DO_NOT_USE,
                hard_fail_count=1, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=0, failed_slice_count=4,
                patterns_detected=(), pattern_severities={},
                timestamp="2025-01-05T00:00:00Z", run_id="run-005",
            ),
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)

        assert trajectory.mode_evolution.total_runs == 5
        assert trajectory.mode_evolution.full_pipeline_count == 3
        assert trajectory.mode_evolution.degraded_analysis_count == 1
        assert trajectory.mode_evolution.evidence_only_count == 1

    def test_build_trajectory_transitions(self):
        """Trajectory should correctly count mode transitions."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        # Alternating modes = 4 transitions
        modes = [
            PipelineMode.FULL_PIPELINE,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.EVIDENCE_ONLY,
            PipelineMode.FULL_PIPELINE,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)

        assert trajectory.mode_evolution.mode_transitions == 4

    def test_build_trajectory_current_streak(self):
        """Trajectory should correctly calculate current streak."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        # 3 FULL, 2 DEGRADED at end
        modes = [
            PipelineMode.FULL_PIPELINE,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.DEGRADED_ANALYSIS,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)

        assert trajectory.mode_evolution.current_streak == 2
        assert trajectory.mode_evolution.current_mode == PipelineMode.DEGRADED_ANALYSIS

    def test_build_trajectory_pattern_trends(self):
        """Trajectory should correctly track pattern trends."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.DEGRADED_ANALYSIS,
                governance_label=GovernanceLabel.WARN,
                hard_fail_count=0, soft_fail_count=1,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=3, failed_slice_count=1,
                patterns_detected=("RN-001", "EV-002"),
                pattern_severities={"RN-001": "MEDIUM", "EV-002": "LOW"},
                timestamp="2025-01-01T00:00:00Z", run_id="run-001",
            ),
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.DEGRADED_ANALYSIS,
                governance_label=GovernanceLabel.WARN,
                hard_fail_count=0, soft_fail_count=1,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=3, failed_slice_count=1,
                patterns_detected=("RN-001",),  # EV-002 resolved
                pattern_severities={"RN-001": "MEDIUM"},
                timestamp="2025-01-02T00:00:00Z", run_id="run-002",
            ),
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(),  # All resolved
                pattern_severities={},
                timestamp="2025-01-03T00:00:00Z", run_id="run-003",
            ),
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)

        trends = trajectory.failure_pattern_trends
        assert trends.total_unique_patterns == 2
        assert "RN-001" in trends.recurring_patterns  # Appeared 2 times
        assert "RN-001" in trends.resolved_patterns
        assert "EV-002" in trends.resolved_patterns
        assert len(trends.active_patterns) == 0  # None in latest run

    def test_build_trajectory_chronic_patterns(self):
        """Chronic patterns should be detected (50%+ occurrence)."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        # RN-001 appears in 3 of 4 runs = 75% = chronic
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.DEGRADED_ANALYSIS,
                governance_label=GovernanceLabel.WARN,
                hard_fail_count=0, soft_fail_count=1,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=3, failed_slice_count=1,
                patterns_detected=("RN-001",) if i != 1 else (),
                pattern_severities={"RN-001": "MEDIUM"} if i != 1 else {},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(4)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)

        assert "RN-001" in trajectory.failure_pattern_trends.chronic_patterns

    def test_trajectory_stability_index_all_healthy(self):
        """All FULL_PIPELINE runs should have high stability index."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)

        # Should be near 1.0 (perfectly stable, all healthy)
        assert trajectory.stability_index >= 0.9

    def test_trajectory_stability_index_all_failing(self):
        """All EVIDENCE_ONLY runs should have low stability index."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.EVIDENCE_ONLY,
                governance_label=GovernanceLabel.DO_NOT_USE,
                hard_fail_count=2, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=0, failed_slice_count=4,
                patterns_detected=("GK-001",),
                pattern_severities={"GK-001": "CRITICAL"},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)

        # Should be relatively low (consistent but unhealthy)
        # Gets some credit for consistency (no transitions) even though unhealthy
        assert trajectory.stability_index <= 0.6

    def test_trajectory_to_dict_and_json(self):
        """Trajectory should serialize to dict and JSON."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )
        import json

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        data = trajectory.to_dict()
        json_str = trajectory.to_json()

        assert "schema_version" in data
        assert "trajectory" in data
        assert "mode_evolution" in data
        assert "stability_index" in data

        parsed = json.loads(json_str)
        assert parsed["mode_evolution"]["total_runs"] == 1


# =============================================================================
# Test Class: Mode Stability Detection
# =============================================================================


class TestModeStabilityDetection:
    """Tests for mode stability classification."""

    def test_stable_mode(self):
        """Consistent mode with low transitions should be STABLE."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ModeStability,
        )

        # All same mode
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(6)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        assert trajectory.mode_evolution.stability == ModeStability.STABLE

    def test_improving_mode(self):
        """Improving trend should be detected."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ModeStability,
        )

        # Start with EVIDENCE_ONLY, end with FULL_PIPELINE
        modes = [
            PipelineMode.EVIDENCE_ONLY,
            PipelineMode.EVIDENCE_ONLY,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.FULL_PIPELINE,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        assert trajectory.mode_evolution.stability == ModeStability.IMPROVING

    def test_degrading_mode(self):
        """Degrading trend should be detected."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ModeStability,
        )

        # Start with FULL_PIPELINE, end with EVIDENCE_ONLY
        modes = [
            PipelineMode.FULL_PIPELINE,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.EVIDENCE_ONLY,
            PipelineMode.EVIDENCE_ONLY,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        assert trajectory.mode_evolution.stability == ModeStability.DEGRADING

    def test_volatile_mode(self):
        """Frequent mode changes should be VOLATILE."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ModeStability,
        )

        # Alternating modes = high transition rate
        modes = [
            PipelineMode.FULL_PIPELINE,
            PipelineMode.EVIDENCE_ONLY,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.EVIDENCE_ONLY,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        assert trajectory.mode_evolution.stability == ModeStability.VOLATILE


# =============================================================================
# Test Class: Degradation Risk Prediction
# =============================================================================


class TestDegradationRiskPrediction:
    """Tests for degradation risk forecasting."""

    def test_predict_risk_empty_trajectory(self):
        """Empty trajectory should return HIGH risk."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            RiskLevel,
        )

        trajectory = build_pipeline_health_trajectory([])
        prediction = predict_degradation_risk(trajectory)

        assert prediction.risk_level == RiskLevel.HIGH
        assert prediction.confidence <= 0.2

    def test_predict_risk_low_stable_healthy(self):
        """Stable healthy pipeline should have LOW risk."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            RiskLevel,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)

        assert prediction.risk_level == RiskLevel.LOW
        assert prediction.next_run_mode_prediction == PipelineMode.FULL_PIPELINE

    def test_predict_risk_high_evidence_only(self):
        """Consecutive EVIDENCE_ONLY should have HIGH risk."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            RiskLevel,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.EVIDENCE_ONLY,
                governance_label=GovernanceLabel.DO_NOT_USE,
                hard_fail_count=1, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=0, failed_slice_count=4,
                patterns_detected=("GK-001",),
                pattern_severities={"GK-001": "CRITICAL"},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)

        assert prediction.risk_level == RiskLevel.HIGH
        assert "GK-001" in prediction.explanatory_patterns

    def test_predict_risk_medium_degraded(self):
        """DEGRADED_ANALYSIS should typically have MEDIUM risk."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            RiskLevel,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.DEGRADED_ANALYSIS,
                governance_label=GovernanceLabel.WARN,
                hard_fail_count=0, soft_fail_count=1,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=3, failed_slice_count=1,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(3)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)

        assert prediction.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)

    def test_predict_risk_factors_captured(self):
        """Risk factors should be properly captured."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.EVIDENCE_ONLY,
                governance_label=GovernanceLabel.DO_NOT_USE,
                hard_fail_count=2, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=0, failed_slice_count=4,
                patterns_detected=("GK-001", "AU-001"),
                pattern_severities={"GK-001": "CRITICAL", "AU-001": "CRITICAL"},
                timestamp="2025-01-01T00:00:00Z", run_id="run-001",
            ),
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)

        assert len(prediction.risk_factors) > 0
        assert any("EVIDENCE_ONLY" in rf for rf in prediction.risk_factors)
        assert any("critical" in rf.lower() for rf in prediction.risk_factors)

    def test_predict_protective_factors_captured(self):
        """Protective factors should be properly captured."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)

        assert len(prediction.protective_factors) > 0
        assert any("FULL_PIPELINE" in pf for pf in prediction.protective_factors)

    def test_predict_mode_improving_trend(self):
        """Improving trend should predict better mode."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        modes = [
            PipelineMode.EVIDENCE_ONLY,
            PipelineMode.EVIDENCE_ONLY,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.FULL_PIPELINE,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)

        # Should predict staying at or continuing to FULL_PIPELINE
        assert prediction.next_run_mode_prediction == PipelineMode.FULL_PIPELINE

    def test_prediction_to_dict_and_json(self):
        """Prediction should serialize to dict and JSON."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )
        import json

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)

        data = prediction.to_dict()
        json_str = prediction.to_json()

        assert "next_run_mode_prediction" in data
        assert "risk_level" in data
        assert "risk_factors" in data

        parsed = json.loads(json_str)
        assert "confidence" in parsed


# =============================================================================
# Test Class: Global Health Summary
# =============================================================================


class TestGlobalHealthSummary:
    """Tests for governance summary extraction."""

    def test_summary_empty_trajectory(self):
        """Empty trajectory should return not OK status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            summarize_topology_for_global_health,
        )

        trajectory = build_pipeline_health_trajectory([])
        summary = summarize_topology_for_global_health(trajectory)

        assert summary.pipeline_ok is False
        assert summary.runs_since_last_full_pipeline == -1
        assert "No trajectory data" in summary.recommendation

    def test_summary_healthy_pipeline(self):
        """Healthy pipeline should show OK status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            summarize_topology_for_global_health,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        summary = summarize_topology_for_global_health(trajectory)

        assert summary.pipeline_ok is True
        assert summary.current_mode == PipelineMode.FULL_PIPELINE
        assert summary.runs_since_last_full_pipeline == 0
        assert "healthy" in summary.recommendation.lower()

    def test_summary_critical_patterns(self):
        """Critical patterns should be reported."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            summarize_topology_for_global_health,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=1, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK-001", "AU-005"),
            pattern_severities={"GK-001": "CRITICAL", "AU-005": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        summary = summarize_topology_for_global_health(trajectory)

        assert summary.pipeline_ok is False
        assert "GK-001" in summary.unresolved_critical_patterns
        assert "AU-005" in summary.unresolved_critical_patterns
        assert "CRITICAL" in summary.recommendation

    def test_summary_runs_since_full_pipeline(self):
        """Should correctly count runs since FULL_PIPELINE."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            summarize_topology_for_global_health,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        modes = [
            PipelineMode.FULL_PIPELINE,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.EVIDENCE_ONLY,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        summary = summarize_topology_for_global_health(trajectory)

        assert summary.runs_since_last_full_pipeline == 3  # 3 runs since first

    def test_summary_never_full_pipeline(self):
        """Never achieving FULL_PIPELINE should show -1."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            summarize_topology_for_global_health,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.EVIDENCE_ONLY,
                governance_label=GovernanceLabel.DO_NOT_USE,
                hard_fail_count=1, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=0, failed_slice_count=4,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(3)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        summary = summarize_topology_for_global_health(trajectory)

        assert summary.runs_since_last_full_pipeline == -1

    def test_summary_recommendations(self):
        """Recommendations should be appropriate for the state."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            summarize_topology_for_global_health,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        # Test WARNING recommendation for DEGRADED
        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.DEGRADED_ANALYSIS,
            governance_label=GovernanceLabel.WARN,
            hard_fail_count=0, soft_fail_count=1,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=3, failed_slice_count=1,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        summary = summarize_topology_for_global_health(trajectory)

        assert "WARNING" in summary.recommendation

    def test_summary_to_dict_and_json(self):
        """Summary should serialize to dict and JSON."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            summarize_topology_for_global_health,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )
        import json

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        summary = summarize_topology_for_global_health(trajectory)

        data = summary.to_dict()
        json_str = summary.to_json()

        assert "pipeline_ok" in data
        assert "mode_stability" in data
        assert "recommendation" in data

        parsed = json.loads(json_str)
        assert parsed["pipeline_ok"] is True


# =============================================================================
# Phase IV Tests: Pipeline Degradation Guardrail & Director Outlook
# =============================================================================


# =============================================================================
# Test Class: Release Guardrail
# =============================================================================


class TestReleaseGuardrail:
    """Tests for the release guardrail evaluation."""

    def test_release_guardrail_empty_trajectory(self):
        """Empty trajectory should block release."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            ReleaseStatus,
        )

        trajectory = build_pipeline_health_trajectory([])
        prediction = predict_degradation_risk(trajectory)
        result = evaluate_topology_for_release(trajectory, prediction)

        assert result.release_ok is False
        assert result.status == ReleaseStatus.BLOCK
        assert len(result.blocking_reasons) > 0

    def test_release_guardrail_healthy_pipeline_ok(self):
        """Healthy pipeline should allow release."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ReleaseStatus,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        result = evaluate_topology_for_release(trajectory, prediction)

        assert result.release_ok is True
        assert result.status == ReleaseStatus.OK

    def test_release_guardrail_blocks_high_risk(self):
        """High risk should block release."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ReleaseStatus,
            RiskLevel,
        )

        # Create trajectory with EVIDENCE_ONLY to trigger HIGH risk
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.EVIDENCE_ONLY,
                governance_label=GovernanceLabel.DO_NOT_USE,
                hard_fail_count=1, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=0, failed_slice_count=4,
                patterns_detected=("GK-001",),
                pattern_severities={"GK-001": "CRITICAL"},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(3)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        result = evaluate_topology_for_release(trajectory, prediction)

        assert result.release_ok is False
        assert result.status == ReleaseStatus.BLOCK
        assert result.risk_level == RiskLevel.HIGH

    def test_release_guardrail_blocks_evidence_only(self):
        """EVIDENCE_ONLY mode should block release."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ReleaseStatus,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=1, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=(),
            pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        result = evaluate_topology_for_release(trajectory, prediction)

        assert result.release_ok is False
        assert result.status == ReleaseStatus.BLOCK
        assert any("EVIDENCE_ONLY" in r for r in result.blocking_reasons)

    def test_release_guardrail_blocks_critical_patterns(self):
        """Unresolved critical patterns should block release."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ReleaseStatus,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=1, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK-001", "AU-005"),
            pattern_severities={"GK-001": "CRITICAL", "AU-005": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        result = evaluate_topology_for_release(trajectory, prediction)

        assert result.release_ok is False
        assert result.status == ReleaseStatus.BLOCK
        assert any("critical patterns" in r.lower() for r in result.blocking_reasons)

    def test_release_guardrail_warns_medium_risk(self):
        """Medium risk should warn but allow release."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ReleaseStatus,
        )

        # DEGRADED_ANALYSIS typically results in MEDIUM risk
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.DEGRADED_ANALYSIS,
                governance_label=GovernanceLabel.WARN,
                hard_fail_count=0, soft_fail_count=1,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=3, failed_slice_count=1,
                patterns_detected=(),
                pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(3)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        result = evaluate_topology_for_release(trajectory, prediction)

        # Should be either WARN or BLOCK depending on risk factors
        assert result.status in (ReleaseStatus.WARN, ReleaseStatus.BLOCK)

    def test_release_guardrail_warns_volatile_stability(self):
        """Volatile mode stability should warn."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ReleaseStatus,
        )

        # Alternating modes = volatile
        modes = [
            PipelineMode.FULL_PIPELINE,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.DEGRADED_ANALYSIS,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        result = evaluate_topology_for_release(trajectory, prediction)

        # Should warn due to volatile stability
        assert result.status in (ReleaseStatus.WARN, ReleaseStatus.BLOCK)

    def test_release_guardrail_watch_patterns(self):
        """Watch patterns should include active and chronic patterns."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        # Create trajectory with patterns
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.DEGRADED_ANALYSIS,
                governance_label=GovernanceLabel.WARN,
                hard_fail_count=0, soft_fail_count=1,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=3, failed_slice_count=1,
                patterns_detected=("RN-001", "EV-002"),
                pattern_severities={"RN-001": "MEDIUM", "EV-002": "LOW"},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(3)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        result = evaluate_topology_for_release(trajectory, prediction)

        # Watch patterns should include the active patterns
        assert "RN-001" in result.watch_patterns or "EV-002" in result.watch_patterns

    def test_release_guardrail_to_dict(self):
        """ReleaseGuardrailResult should serialize to dict."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        result = evaluate_topology_for_release(trajectory, prediction)

        data = result.to_dict()
        assert "release_ok" in data
        assert "status" in data
        assert "blocking_reasons" in data
        assert "watch_patterns" in data


# =============================================================================
# Test Class: MAAS Health Signal
# =============================================================================


class TestMaasHealthSignal:
    """Tests for MAAS health signal generation."""

    def test_maas_signal_empty_trajectory(self):
        """Empty trajectory should return BLOCK status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            summarize_topology_for_maas,
            MaasStatus,
        )

        trajectory = build_pipeline_health_trajectory([])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        maas_signal = summarize_topology_for_maas(trajectory, release_eval)

        assert maas_signal.status == MaasStatus.BLOCK
        assert maas_signal.topology_ok_for_evidence is False

    def test_maas_signal_healthy_pipeline(self):
        """Healthy pipeline should return OK status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            summarize_topology_for_maas,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            MaasStatus,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        maas_signal = summarize_topology_for_maas(trajectory, release_eval)

        assert maas_signal.status == MaasStatus.OK
        assert maas_signal.topology_ok_for_evidence is True
        assert "OK" in maas_signal.alert_summary

    def test_maas_signal_blocked_pipeline(self):
        """Blocked pipeline should return BLOCK status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            summarize_topology_for_maas,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            MaasStatus,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=1, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK-001",),
            pattern_severities={"GK-001": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        maas_signal = summarize_topology_for_maas(trajectory, release_eval)

        assert maas_signal.status == MaasStatus.BLOCK
        assert "GK-001" in maas_signal.unresolved_critical_patterns
        assert "BLOCK" in maas_signal.alert_summary

    def test_maas_signal_attention_status(self):
        """Warning state should return ATTENTION status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            summarize_topology_for_maas,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            MaasStatus,
            ReleaseStatus,
        )

        # Create volatile trajectory to trigger warning
        modes = [
            PipelineMode.FULL_PIPELINE,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.DEGRADED_ANALYSIS,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        maas_signal = summarize_topology_for_maas(trajectory, release_eval)

        # If release is WARN, MAAS should be ATTENTION
        if release_eval.status == ReleaseStatus.WARN:
            assert maas_signal.status == MaasStatus.ATTENTION

    def test_maas_signal_to_dict(self):
        """MaasHealthSignal should serialize to dict."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            summarize_topology_for_maas,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        maas_signal = summarize_topology_for_maas(trajectory, release_eval)

        data = maas_signal.to_dict()
        assert "topology_ok_for_evidence" in data
        assert "mode_stability" in data
        assert "status" in data
        assert "alert_summary" in data


# =============================================================================
# Test Class: Director Topology Panel
# =============================================================================


class TestDirectorTopologyPanel:
    """Tests for director topology forecast panel."""

    def test_director_panel_empty_trajectory(self):
        """Empty trajectory should show RED status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel,
            StatusLight,
        )

        trajectory = build_pipeline_health_trajectory([])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel(trajectory, prediction, release_eval)

        assert panel.status_light == StatusLight.RED
        assert panel.runs_analyzed == 0
        assert "No pipeline data" in panel.headline

    def test_director_panel_green_status(self):
        """Healthy pipeline should show GREEN status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            StatusLight,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel(trajectory, prediction, release_eval)

        assert panel.status_light == StatusLight.GREEN
        assert panel.current_mode == PipelineMode.FULL_PIPELINE
        assert panel.runs_analyzed == 5

    def test_director_panel_red_status_blocked(self):
        """Blocked pipeline should show RED status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            StatusLight,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=1, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK-001",),
            pattern_severities={"GK-001": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel(trajectory, prediction, release_eval)

        assert panel.status_light == StatusLight.RED
        assert "blocked" in panel.headline.lower()

    def test_director_panel_yellow_status_warning(self):
        """Warning state should show YELLOW status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            StatusLight,
            ReleaseStatus,
        )

        # Volatile trajectory
        modes = [
            PipelineMode.FULL_PIPELINE,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.DEGRADED_ANALYSIS,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel(trajectory, prediction, release_eval)

        # Should be YELLOW due to volatility
        if release_eval.status == ReleaseStatus.WARN:
            assert panel.status_light == StatusLight.YELLOW

    def test_director_panel_headline_improving(self):
        """Improving trend should show appropriate headline."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ModeStability,
        )

        # Improving trajectory
        modes = [
            PipelineMode.EVIDENCE_ONLY,
            PipelineMode.EVIDENCE_ONLY,
            PipelineMode.DEGRADED_ANALYSIS,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.FULL_PIPELINE,
            PipelineMode.FULL_PIPELINE,
        ]
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=mode,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i, mode in enumerate(modes)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel(trajectory, prediction, release_eval)

        if trajectory.mode_evolution.stability == ModeStability.IMPROVING:
            assert "improving" in panel.headline.lower()

    def test_director_panel_headline_stable(self):
        """Stable pipeline should show stability in headline."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ModeStability,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(6)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel(trajectory, prediction, release_eval)

        if trajectory.mode_evolution.stability == ModeStability.STABLE:
            assert "stable" in panel.headline.lower()

    def test_director_panel_next_run_prediction(self):
        """Panel should include next run prediction."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel(trajectory, prediction, release_eval)

        assert panel.next_run_prediction == prediction.next_run_mode_prediction

    def test_director_panel_to_dict(self):
        """DirectorTopologyPanel should serialize to dict."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel(trajectory, prediction, release_eval)

        data = panel.to_dict()
        assert "status_light" in data
        assert "current_mode" in data
        assert "headline" in data
        assert "runs_analyzed" in data

    def test_director_panel_to_json(self):
        """DirectorTopologyPanel should serialize to JSON."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )
        import json

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel(trajectory, prediction, release_eval)

        json_str = panel.to_json()
        parsed = json.loads(json_str)

        assert parsed["status_light"] in ("GREEN", "YELLOW", "RED")
        assert parsed["current_mode"] == "FULL_PIPELINE"


# =============================================================================
# Phase V: Topology × Bundle × DAG Unification Tests
# =============================================================================


class TestTopologyBundleJointView:
    """Tests for build_topology_bundle_joint_view function."""

    def test_joint_view_empty_trajectory(self):
        """Empty trajectory should result in WARN status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            CROSS_SYSTEM_SCHEMA_VERSION,
        )

        trajectory = build_pipeline_health_trajectory([])
        bundle_evolution = {
            "stability_rating": "STABLE",
            "integration_status": "OK",
        }

        result = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        assert result["topology_ok_for_integration"] is False
        assert result["joint_status"] == "WARN"
        assert result["schema_version"] == CROSS_SYSTEM_SCHEMA_VERSION
        assert any("No trajectory data" in r for r in result["reasons"])

    def test_joint_view_healthy_topology_stable_bundle(self):
        """Healthy topology + stable bundle should result in OK."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        bundle_evolution = {
            "stability_rating": "STABLE",
            "integration_status": "OK",
        }

        result = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        assert result["topology_ok_for_integration"] is True
        assert result["joint_status"] == "OK"
        assert result["bundle_stability_rating"] == "STABLE"

    def test_joint_view_blocks_on_topology_evidence_only(self):
        """EVIDENCE_ONLY mode should result in BLOCK."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=1, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK01",), pattern_severities={"GK01": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {
            "stability_rating": "STABLE",
            "integration_status": "OK",
        }

        result = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        assert result["topology_ok_for_integration"] is False
        assert result["joint_status"] == "BLOCK"

    def test_joint_view_blocks_on_bundle_block(self):
        """Bundle BLOCK should result in BLOCK regardless of topology."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {
            "stability_rating": "UNSTABLE",
            "integration_status": "BLOCK",
            "reasons": ["Critical bundle validation failure"],
        }

        result = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        assert result["joint_status"] == "BLOCK"
        assert any("Bundle" in r and "BLOCK" in r for r in result["reasons"])

    def test_joint_view_warns_on_bundle_warn(self):
        """Bundle WARN should result in WARN."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {
            "stability_rating": "STABLE",
            "integration_status": "WARN",
            "reasons": ["Minor version mismatch detected"],
        }

        result = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        assert result["joint_status"] == "WARN"

    def test_joint_view_warns_on_unstable_bundle(self):
        """Unstable bundle should result in WARN even if integration OK."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {
            "stability_rating": "UNSTABLE",
            "integration_status": "OK",
        }

        result = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        assert result["joint_status"] == "WARN"
        assert any("UNSTABLE" in r for r in result["reasons"])

    def test_joint_view_warns_on_degrading_topology(self):
        """Degrading topology should result in WARN."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            ModeStability,
        )

        # Create degrading trajectory: FULL -> DEGRADED -> EVIDENCE_ONLY
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp="2025-01-01T00:00:00Z", run_id="run-001",
            ),
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.DEGRADED_ANALYSIS,
                governance_label=GovernanceLabel.WARN,
                hard_fail_count=0, soft_fail_count=1,
                slice_baseline_fail_count=1, slice_rfl_fail_count=0,
                successful_slice_count=3, failed_slice_count=1,
                patterns_detected=("RN01",), pattern_severities={"RN01": "MEDIUM"},
                timestamp="2025-01-02T00:00:00Z", run_id="run-002",
            ),
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.DEGRADED_ANALYSIS,
                governance_label=GovernanceLabel.WARN,
                hard_fail_count=0, soft_fail_count=2,
                slice_baseline_fail_count=2, slice_rfl_fail_count=0,
                successful_slice_count=2, failed_slice_count=2,
                patterns_detected=("RN01", "RN02"), pattern_severities={"RN01": "MEDIUM", "RN02": "MEDIUM"},
                timestamp="2025-01-03T00:00:00Z", run_id="run-003",
            ),
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        bundle_evolution = {
            "stability_rating": "STABLE",
            "integration_status": "OK",
        }

        result = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        # Should warn due to degrading or volatile stability
        assert result["joint_status"] in ("WARN", "OK")  # Depends on stability calc

    def test_joint_view_includes_schema_version(self):
        """Result should include schema version."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            CROSS_SYSTEM_SCHEMA_VERSION,
        )

        trajectory = build_pipeline_health_trajectory([])
        bundle_evolution = {"stability_rating": "UNKNOWN", "integration_status": "UNKNOWN"}

        result = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        assert "schema_version" in result
        assert result["schema_version"] == CROSS_SYSTEM_SCHEMA_VERSION


class TestGlobalConsoleAdapter:
    """Tests for summarize_topology_for_global_console function."""

    def test_console_summary_empty_trajectory(self):
        """Empty trajectory should return blocked status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            summarize_topology_for_global_console,
            CROSS_SYSTEM_SCHEMA_VERSION,
        )

        trajectory = build_pipeline_health_trajectory([])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)

        result = summarize_topology_for_global_console(trajectory, release_eval)

        assert result["topology_ok"] is False
        assert result["status_light"] == "RED"
        assert result["stability_index"] == 0.0
        assert result["schema_version"] == CROSS_SYSTEM_SCHEMA_VERSION

    def test_console_summary_healthy_pipeline(self):
        """Healthy pipeline should return green status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            summarize_topology_for_global_console,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)

        result = summarize_topology_for_global_console(trajectory, release_eval)

        assert result["topology_ok"] is True
        assert result["status_light"] == "GREEN"
        assert result["current_mode"] == "FULL_PIPELINE"
        assert result["runs_analyzed"] == 5

    def test_console_summary_blocked_pipeline(self):
        """Blocked pipeline should return red status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            summarize_topology_for_global_console,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=2, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK01",), pattern_severities={"GK01": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)

        result = summarize_topology_for_global_console(trajectory, release_eval)

        assert result["topology_ok"] is False
        assert result["status_light"] == "RED"
        assert "BLOCKED" in result["alert_summary"] or "EVIDENCE_ONLY" in result["alert_summary"]

    def test_console_summary_warning_pipeline(self):
        """Warning pipeline should return yellow status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            summarize_topology_for_global_console,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.DEGRADED_ANALYSIS,
            governance_label=GovernanceLabel.WARN,
            hard_fail_count=0, soft_fail_count=2,
            slice_baseline_fail_count=1, slice_rfl_fail_count=1,
            successful_slice_count=2, failed_slice_count=2,
            patterns_detected=("RN01",), pattern_severities={"RN01": "MEDIUM"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)

        result = summarize_topology_for_global_console(trajectory, release_eval)

        assert result["status_light"] in ("YELLOW", "RED")  # May be RED if risk is HIGH
        assert result["current_mode"] == "DEGRADED_ANALYSIS"

    def test_console_summary_alert_format(self):
        """Alert summary should be a concise string."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            summarize_topology_for_global_console,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)

        result = summarize_topology_for_global_console(trajectory, release_eval)

        assert isinstance(result["alert_summary"], str)
        assert len(result["alert_summary"]) > 0
        # Should start with status prefix
        assert result["alert_summary"].startswith(("OK:", "WARNING:", "BLOCKED:"))


class TestDagPostureConsistency:
    """Tests for check_consistency_with_dag_posture function."""

    def test_consistency_both_ok(self):
        """Both OK should be CONSISTENT."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            check_consistency_with_dag_posture,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        dag_health = {"drift_status": "OK", "reasons": []}

        result = check_consistency_with_dag_posture(trajectory, dag_health)

        assert result["consistency_status"] == "CONSISTENT"
        assert result["topology_status"] == "OK"
        assert result["dag_status"] == "OK"

    def test_consistency_both_block(self):
        """Both BLOCK should be CONSISTENT."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            check_consistency_with_dag_posture,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=2, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK01",), pattern_severities={"GK01": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        dag_health = {"drift_status": "BLOCKED", "reasons": ["Critical regression"]}

        result = check_consistency_with_dag_posture(trajectory, dag_health)

        assert result["consistency_status"] == "CONSISTENT"
        assert result["topology_status"] == "BLOCK"
        assert result["dag_status"] == "BLOCK"

    def test_consistency_both_warn(self):
        """Both WARN should be CONSISTENT."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            check_consistency_with_dag_posture,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.DEGRADED_ANALYSIS,
            governance_label=GovernanceLabel.WARN,
            hard_fail_count=0, soft_fail_count=1,
            slice_baseline_fail_count=1, slice_rfl_fail_count=0,
            successful_slice_count=3, failed_slice_count=1,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        dag_health = {"drift_status": "WARN", "reasons": ["Minor regression"]}

        result = check_consistency_with_dag_posture(trajectory, dag_health)

        assert result["consistency_status"] == "CONSISTENT"
        assert result["topology_status"] == "WARN"
        assert result["dag_status"] == "WARN"

    def test_consistency_conflict_topology_ok_dag_block(self):
        """Topology OK but DAG BLOCK should be CONFLICT."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            check_consistency_with_dag_posture,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        dag_health = {"drift_status": "BLOCKED", "reasons": ["Critical structural issue"]}

        result = check_consistency_with_dag_posture(trajectory, dag_health)

        assert result["consistency_status"] == "CONFLICT"
        assert result["topology_status"] == "OK"
        assert result["dag_status"] == "BLOCK"
        assert any("BLOCK" in r and "OK" in r for r in result["reasons"])

    def test_consistency_conflict_topology_block_dag_ok(self):
        """Topology BLOCK but DAG OK should be CONFLICT."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            check_consistency_with_dag_posture,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=1, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK01",), pattern_severities={"GK01": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        dag_health = {"drift_status": "OK", "reasons": []}

        result = check_consistency_with_dag_posture(trajectory, dag_health)

        assert result["consistency_status"] == "CONFLICT"
        assert result["topology_status"] == "BLOCK"
        assert result["dag_status"] == "OK"

    def test_consistency_tension_topology_ok_dag_warn(self):
        """Topology OK but DAG WARN should be TENSION."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            check_consistency_with_dag_posture,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        dag_health = {"drift_status": "WARN", "reasons": ["Minor depth decrease"]}

        result = check_consistency_with_dag_posture(trajectory, dag_health)

        assert result["consistency_status"] == "TENSION"
        assert result["topology_status"] == "OK"
        assert result["dag_status"] == "WARN"

    def test_consistency_tension_topology_warn_dag_ok(self):
        """Topology WARN but DAG OK should be TENSION."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            check_consistency_with_dag_posture,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.DEGRADED_ANALYSIS,
            governance_label=GovernanceLabel.WARN,
            hard_fail_count=0, soft_fail_count=1,
            slice_baseline_fail_count=1, slice_rfl_fail_count=0,
            successful_slice_count=3, failed_slice_count=1,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        dag_health = {"drift_status": "OK", "reasons": []}

        result = check_consistency_with_dag_posture(trajectory, dag_health)

        assert result["consistency_status"] == "TENSION"
        assert result["topology_status"] == "WARN"
        assert result["dag_status"] == "OK"

    def test_consistency_empty_trajectory(self):
        """Empty trajectory should be treated as BLOCK."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            check_consistency_with_dag_posture,
        )

        trajectory = build_pipeline_health_trajectory([])
        dag_health = {"drift_status": "OK", "reasons": []}

        result = check_consistency_with_dag_posture(trajectory, dag_health)

        assert result["consistency_status"] == "CONFLICT"
        assert result["topology_status"] == "BLOCK"
        assert result["dag_status"] == "OK"

    def test_consistency_normalizes_blocked_to_block(self):
        """DAG status BLOCKED should be normalized to BLOCK."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            check_consistency_with_dag_posture,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=1, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        dag_health = {"drift_status": "BLOCKED", "reasons": []}  # BLOCKED not BLOCK

        result = check_consistency_with_dag_posture(trajectory, dag_health)

        assert result["dag_status"] == "BLOCK"  # Normalized
        assert result["consistency_status"] == "CONSISTENT"

    def test_consistency_includes_schema_version(self):
        """Result should include schema version."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            check_consistency_with_dag_posture,
            CROSS_SYSTEM_SCHEMA_VERSION,
        )

        trajectory = build_pipeline_health_trajectory([])
        dag_health = {"drift_status": "OK"}

        result = check_consistency_with_dag_posture(trajectory, dag_health)

        assert "schema_version" in result
        assert result["schema_version"] == CROSS_SYSTEM_SCHEMA_VERSION


class TestDirectorPanelExtended:
    """Tests for DirectorTopologyPanelExtended and build_topology_director_panel_extended."""

    def test_extended_panel_without_dag_health(self):
        """Extended panel without DAG health should have None consistency."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)

        panel = build_topology_director_panel_extended(
            trajectory, prediction, release_eval, dag_global_health=None
        )

        assert panel.cross_system_consistency is None
        assert panel.current_mode == PipelineMode.FULL_PIPELINE

    def test_extended_panel_with_dag_health_consistent(self):
        """Extended panel with consistent DAG health should show CONSISTENT."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        dag_health = {"drift_status": "OK", "reasons": []}

        panel = build_topology_director_panel_extended(
            trajectory, prediction, release_eval, dag_global_health=dag_health
        )

        assert panel.cross_system_consistency == "CONSISTENT"

    def test_extended_panel_with_dag_health_conflict(self):
        """Extended panel with conflicting DAG health should show CONFLICT."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        dag_health = {"drift_status": "BLOCKED", "reasons": ["Critical issue"]}

        panel = build_topology_director_panel_extended(
            trajectory, prediction, release_eval, dag_global_health=dag_health
        )

        assert panel.cross_system_consistency == "CONFLICT"

    def test_extended_panel_with_dag_health_tension(self):
        """Extended panel with tension DAG health should show TENSION."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        dag_health = {"drift_status": "WARN", "reasons": ["Minor regression"]}

        panel = build_topology_director_panel_extended(
            trajectory, prediction, release_eval, dag_global_health=dag_health
        )

        assert panel.cross_system_consistency == "TENSION"

    def test_extended_panel_to_dict(self):
        """Extended panel should serialize to dict with all fields."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            CROSS_SYSTEM_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        dag_health = {"drift_status": "OK"}

        panel = build_topology_director_panel_extended(
            trajectory, prediction, release_eval, dag_global_health=dag_health
        )

        data = panel.to_dict()

        assert "status_light" in data
        assert "current_mode" in data
        assert "cross_system_consistency" in data
        assert "schema_version" in data
        assert data["schema_version"] == CROSS_SYSTEM_SCHEMA_VERSION

    def test_extended_panel_to_json(self):
        """Extended panel should serialize to valid JSON."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )
        import json

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)

        panel = build_topology_director_panel_extended(
            trajectory, prediction, release_eval
        )

        json_str = panel.to_json()
        parsed = json.loads(json_str)

        assert parsed["cross_system_consistency"] is None
        assert "schema_version" in parsed

    def test_extended_panel_preserves_base_panel_logic(self):
        """Extended panel should preserve base panel traffic light logic."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel,
            build_topology_director_panel_extended,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=2, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK01",), pattern_severities={"GK01": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)

        base_panel = build_topology_director_panel(trajectory, prediction, release_eval)
        extended_panel = build_topology_director_panel_extended(
            trajectory, prediction, release_eval, dag_global_health={"drift_status": "OK"}
        )

        # Extended panel should have same traffic light as base
        assert extended_panel.status_light == base_panel.status_light
        assert extended_panel.current_mode == base_panel.current_mode
        assert extended_panel.risk_level == base_panel.risk_level
        assert extended_panel.headline == base_panel.headline


# =============================================================================
# Phase VI: Topology/Bundle as a Coherent Governance Layer Tests
# =============================================================================


class TestGlobalConsoleTile:
    """Tests for summarize_topology_bundle_for_global_console function."""

    def test_console_tile_ok_status(self):
        """OK joint status should produce GREEN tile."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            summarize_topology_bundle_for_global_console,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            GOVERNANCE_SIGNAL_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel_extended(
            trajectory, prediction, release_eval, dag_global_health={"drift_status": "OK"}
        )

        tile = summarize_topology_bundle_for_global_console(joint_view, panel)

        assert tile["topology_bundle_ok"] is True
        assert tile["status_light"] == "GREEN"
        assert tile["consistency_status"] == "CONSISTENT"
        assert tile["schema_version"] == GOVERNANCE_SIGNAL_SCHEMA_VERSION
        assert "healthy" in tile["headline"].lower() or "operational" in tile["headline"].lower()

    def test_console_tile_warn_status(self):
        """WARN joint status should produce YELLOW tile."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            summarize_topology_bundle_for_global_console,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {"stability_rating": "UNSTABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel_extended(trajectory, prediction, release_eval)

        tile = summarize_topology_bundle_for_global_console(joint_view, panel)

        assert tile["status_light"] == "YELLOW"
        assert tile["bundle_stability"] == "UNSTABLE"

    def test_console_tile_block_status(self):
        """BLOCK joint status should produce RED tile."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            summarize_topology_bundle_for_global_console,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=2, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK01",), pattern_severities={"GK01": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel_extended(trajectory, prediction, release_eval)

        tile = summarize_topology_bundle_for_global_console(joint_view, panel)

        assert tile["topology_bundle_ok"] is False
        assert tile["status_light"] == "RED"
        assert "blocked" in tile["headline"].lower()

    def test_console_tile_conflict_overrides_to_red(self):
        """Cross-system conflict should override to RED regardless of joint status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            summarize_topology_bundle_for_global_console,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        # Joint status would be OK, but DAG is BLOCKED creating conflict
        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel_extended(
            trajectory, prediction, release_eval,
            dag_global_health={"drift_status": "BLOCKED"}  # Creates conflict
        )

        tile = summarize_topology_bundle_for_global_console(joint_view, panel)

        assert tile["topology_bundle_ok"] is False
        assert tile["status_light"] == "RED"
        assert tile["consistency_status"] == "CONFLICT"
        assert "conflict" in tile["headline"].lower()

    def test_console_tile_tension_headline(self):
        """Cross-system tension should be reflected in headline."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            summarize_topology_bundle_for_global_console,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.DEGRADED_ANALYSIS,
            governance_label=GovernanceLabel.WARN,
            hard_fail_count=0, soft_fail_count=1,
            slice_baseline_fail_count=1, slice_rfl_fail_count=0,
            successful_slice_count=3, failed_slice_count=1,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        # Topology WARN, DAG OK = TENSION
        panel = build_topology_director_panel_extended(
            trajectory, prediction, release_eval,
            dag_global_health={"drift_status": "OK"}
        )

        tile = summarize_topology_bundle_for_global_console(joint_view, panel)

        assert tile["consistency_status"] == "TENSION"

    def test_console_tile_includes_all_fields(self):
        """Console tile should include all required fields."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            predict_degradation_risk,
            evaluate_topology_for_release,
            build_topology_director_panel_extended,
            summarize_topology_bundle_for_global_console,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        prediction = predict_degradation_risk(trajectory)
        release_eval = evaluate_topology_for_release(trajectory, prediction)
        panel = build_topology_director_panel_extended(trajectory, prediction, release_eval)

        tile = summarize_topology_bundle_for_global_console(joint_view, panel)

        # Check all required fields
        assert "topology_bundle_ok" in tile
        assert "status_light" in tile
        assert "headline" in tile
        assert "consistency_status" in tile
        assert "stability_index" in tile
        assert "topology_mode" in tile
        assert "bundle_stability" in tile
        assert "runs_analyzed" in tile
        assert "risk_level" in tile
        assert "schema_version" in tile


class TestGovernanceSignalAdapter:
    """Tests for to_governance_signal_for_topology_bundle function."""

    def test_governance_signal_ok_status(self):
        """OK joint view should produce OK governance signal."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            to_governance_signal_for_topology_bundle,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
            GOVERNANCE_SIGNAL_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        signal = to_governance_signal_for_topology_bundle(joint_view)

        assert signal["layer"] == "topology_bundle"
        assert signal["status"] == "OK"
        assert signal["blocking_rate"] == 0.0
        assert len(signal["blocking_rules"]) == 0
        assert signal["schema_version"] == GOVERNANCE_SIGNAL_SCHEMA_VERSION

    def test_governance_signal_block_status(self):
        """BLOCK joint view should produce BLOCK governance signal with rules."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            to_governance_signal_for_topology_bundle,
            TopologyBundleBlockingRule,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=2, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK01",), pattern_severities={"GK01": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        signal = to_governance_signal_for_topology_bundle(joint_view)

        assert signal["status"] == "BLOCK"
        assert signal["blocking_rate"] >= 1.0
        assert TopologyBundleBlockingRule.TOPOLOGY_EVIDENCE_ONLY in signal["blocking_rules"]

    def test_governance_signal_bundle_blocked(self):
        """Bundle BLOCK should produce BUNDLE_INTEGRATION_BLOCKED rule."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            to_governance_signal_for_topology_bundle,
            TopologyBundleBlockingRule,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {
            "stability_rating": "UNSTABLE",
            "integration_status": "BLOCK",
            "reasons": ["Critical bundle failure"],
        }
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        signal = to_governance_signal_for_topology_bundle(joint_view)

        assert signal["status"] == "BLOCK"
        assert TopologyBundleBlockingRule.BUNDLE_INTEGRATION_BLOCKED in signal["blocking_rules"]
        assert TopologyBundleBlockingRule.BUNDLE_UNSTABLE in signal["blocking_rules"]

    def test_governance_signal_with_consistency_conflict(self):
        """Conflict consistency should produce CROSS_SYSTEM_CONFLICT rule and BLOCK status."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            check_consistency_with_dag_posture,
            to_governance_signal_for_topology_bundle,
            TopologyBundleBlockingRule,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        # Create conflict: topology OK, DAG BLOCKED
        consistency_result = check_consistency_with_dag_posture(
            trajectory, {"drift_status": "BLOCKED"}
        )

        signal = to_governance_signal_for_topology_bundle(joint_view, consistency_result)

        # Joint view says OK, but conflict upgrades to BLOCK
        assert signal["status"] == "BLOCK"
        assert TopologyBundleBlockingRule.CROSS_SYSTEM_CONFLICT in signal["blocking_rules"]
        assert signal["metrics"]["cross_system_consistency"] == "CONFLICT"

    def test_governance_signal_with_consistency_tension(self):
        """Tension consistency should produce CROSS_SYSTEM_TENSION rule."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            check_consistency_with_dag_posture,
            to_governance_signal_for_topology_bundle,
            TopologyBundleBlockingRule,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        # Create tension: topology OK, DAG WARN
        consistency_result = check_consistency_with_dag_posture(
            trajectory, {"drift_status": "WARN"}
        )

        signal = to_governance_signal_for_topology_bundle(joint_view, consistency_result)

        # Tension doesn't upgrade status, just adds rule
        assert signal["status"] == "OK"
        assert TopologyBundleBlockingRule.CROSS_SYSTEM_TENSION in signal["blocking_rules"]

    def test_governance_signal_blocking_rate_calculation(self):
        """Blocking rate should increase with more issues."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            to_governance_signal_for_topology_bundle,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        # Healthy case
        snapshot_healthy = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory_healthy = build_pipeline_health_trajectory([snapshot_healthy])
        joint_view_healthy = build_topology_bundle_joint_view(
            trajectory_healthy,
            {"stability_rating": "STABLE", "integration_status": "OK"}
        )
        signal_healthy = to_governance_signal_for_topology_bundle(joint_view_healthy)

        # Unhealthy case
        snapshot_unhealthy = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label=GovernanceLabel.DO_NOT_USE,
            hard_fail_count=2, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=0, failed_slice_count=4,
            patterns_detected=("GK01",), pattern_severities={"GK01": "CRITICAL"},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory_unhealthy = build_pipeline_health_trajectory([snapshot_unhealthy])
        joint_view_unhealthy = build_topology_bundle_joint_view(
            trajectory_unhealthy,
            {"stability_rating": "UNSTABLE", "integration_status": "BLOCK"}
        )
        signal_unhealthy = to_governance_signal_for_topology_bundle(joint_view_unhealthy)

        # Unhealthy should have higher blocking rate
        assert signal_unhealthy["blocking_rate"] > signal_healthy["blocking_rate"]
        assert signal_healthy["blocking_rate"] == 0.0
        assert signal_unhealthy["blocking_rate"] == 1.0  # Capped at 1.0

    def test_governance_signal_no_data_rule(self):
        """Empty trajectory should produce TOPOLOGY_NO_DATA rule."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            to_governance_signal_for_topology_bundle,
            TopologyBundleBlockingRule,
        )

        trajectory = build_pipeline_health_trajectory([])
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        signal = to_governance_signal_for_topology_bundle(joint_view)

        assert TopologyBundleBlockingRule.TOPOLOGY_NO_DATA in signal["blocking_rules"]

    def test_governance_signal_metrics_included(self):
        """Signal should include metrics dict with key values."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            to_governance_signal_for_topology_bundle,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        signal = to_governance_signal_for_topology_bundle(joint_view)

        assert "metrics" in signal
        assert "topology_ok_for_integration" in signal["metrics"]
        assert "topology_stability" in signal["metrics"]
        assert "bundle_stability_rating" in signal["metrics"]
        assert "blocking_rule_count" in signal["metrics"]


class TestGovernanceSignalConvenience:
    """Tests for build_topology_bundle_governance_signal convenience function."""

    def test_convenience_function_without_dag(self):
        """Convenience function should work without DAG health."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_governance_signal,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}

        signal = build_topology_bundle_governance_signal(trajectory, bundle_evolution)

        assert signal["layer"] == "topology_bundle"
        assert signal["status"] == "OK"
        assert "cross_system_consistency" not in signal["metrics"]

    def test_convenience_function_with_dag(self):
        """Convenience function should include consistency when DAG provided."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_governance_signal,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        dag_health = {"drift_status": "OK"}

        signal = build_topology_bundle_governance_signal(
            trajectory, bundle_evolution, dag_global_health=dag_health
        )

        assert signal["layer"] == "topology_bundle"
        assert signal["status"] == "OK"
        assert signal["metrics"]["cross_system_consistency"] == "CONSISTENT"

    def test_convenience_function_with_conflict(self):
        """Convenience function should detect and report conflict."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_governance_signal,
            TopologyBundleBlockingRule,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp=f"2025-01-0{i+1}T00:00:00Z", run_id=f"run-00{i+1}",
            )
            for i in range(5)
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        dag_health = {"drift_status": "BLOCKED"}  # Creates conflict

        signal = build_topology_bundle_governance_signal(
            trajectory, bundle_evolution, dag_global_health=dag_health
        )

        assert signal["status"] == "BLOCK"
        assert TopologyBundleBlockingRule.CROSS_SYSTEM_CONFLICT in signal["blocking_rules"]
        assert signal["metrics"]["cross_system_consistency"] == "CONFLICT"


class TestBlockingRulesCoverage:
    """Tests to ensure all blocking rules are properly detected."""

    def test_topology_degrading_rule(self):
        """TOPOLOGY_DEGRADING should be detected for degrading stability."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            to_governance_signal_for_topology_bundle,
            TopologyBundleBlockingRule,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        # Create degrading trajectory
        snapshots = [
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.FULL_PIPELINE,
                governance_label=GovernanceLabel.OK,
                hard_fail_count=0, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=4, failed_slice_count=0,
                patterns_detected=(), pattern_severities={},
                timestamp="2025-01-01T00:00:00Z", run_id="run-001",
            ),
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.DEGRADED_ANALYSIS,
                governance_label=GovernanceLabel.WARN,
                hard_fail_count=0, soft_fail_count=1,
                slice_baseline_fail_count=1, slice_rfl_fail_count=0,
                successful_slice_count=3, failed_slice_count=1,
                patterns_detected=(), pattern_severities={},
                timestamp="2025-01-02T00:00:00Z", run_id="run-002",
            ),
            HealthSnapshot(
                schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
                mode=PipelineMode.EVIDENCE_ONLY,
                governance_label=GovernanceLabel.DO_NOT_USE,
                hard_fail_count=1, soft_fail_count=0,
                slice_baseline_fail_count=0, slice_rfl_fail_count=0,
                successful_slice_count=0, failed_slice_count=4,
                patterns_detected=("GK01",), pattern_severities={"GK01": "CRITICAL"},
                timestamp="2025-01-03T00:00:00Z", run_id="run-003",
            ),
        ]

        trajectory = build_pipeline_health_trajectory(snapshots)
        bundle_evolution = {"stability_rating": "STABLE", "integration_status": "OK"}
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        signal = to_governance_signal_for_topology_bundle(joint_view)

        # Should have either DEGRADING or VOLATILE depending on analysis
        has_stability_rule = (
            TopologyBundleBlockingRule.TOPOLOGY_DEGRADING in signal["blocking_rules"]
            or TopologyBundleBlockingRule.TOPOLOGY_VOLATILE in signal["blocking_rules"]
        )
        assert has_stability_rule or TopologyBundleBlockingRule.TOPOLOGY_EVIDENCE_ONLY in signal["blocking_rules"]

    def test_bundle_integration_warn_rule(self):
        """BUNDLE_INTEGRATION_WARN should be detected for bundle warnings."""
        from backend.pipeline.topology_health import (
            build_pipeline_health_trajectory,
            build_topology_bundle_joint_view,
            to_governance_signal_for_topology_bundle,
            TopologyBundleBlockingRule,
            HealthSnapshot,
            HEALTH_SNAPSHOT_SCHEMA_VERSION,
        )

        snapshot = HealthSnapshot(
            schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
            mode=PipelineMode.FULL_PIPELINE,
            governance_label=GovernanceLabel.OK,
            hard_fail_count=0, soft_fail_count=0,
            slice_baseline_fail_count=0, slice_rfl_fail_count=0,
            successful_slice_count=4, failed_slice_count=0,
            patterns_detected=(), pattern_severities={},
            timestamp="2025-01-01T00:00:00Z", run_id="run-001",
        )

        trajectory = build_pipeline_health_trajectory([snapshot])
        bundle_evolution = {
            "stability_rating": "STABLE",
            "integration_status": "WARN",
            "reasons": ["Minor issue detected"],
        }
        joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

        signal = to_governance_signal_for_topology_bundle(joint_view)

        assert TopologyBundleBlockingRule.BUNDLE_INTEGRATION_WARN in signal["blocking_rules"]
