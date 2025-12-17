"""
Tests for curriculum integration module.

Tests cover:
- P3 stability hook attachment
- Evidence pack attachment
- Council classification (OK/WARN/BLOCK)
- Batch summarization
"""

import pytest
from typing import Any, Dict, List

from curriculum.enforcement import (
    DriftSeverity,
    DriftStatus,
    GovernanceSignalType,
    GovernanceSignal,
    DriftTimelineEvent,
    Violation,
    ChangedParam,
    MonotonicityViolation,
    GateEvolutionViolation,
)
from curriculum.integration import (
    attach_curriculum_governance_to_p3,
    attach_curriculum_timeline_to_p3,
    attach_curriculum_to_evidence,
    council_classify_curriculum,
    council_classify_curriculum_from_dict,
    summarize_curriculum_for_council,
    # CTRPK functions
    compute_ctrpk,
    ctrpk_to_status_light,
    compute_ctrpk_trend,
    council_classify_ctrpk,
    build_ctrpk_summary,
    build_ctrpk_compact,
    attach_ctrpk_to_evidence,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def ok_governance_signal() -> GovernanceSignal:
    """Return a governance signal with OK status."""
    return GovernanceSignal(
        signal_id="sig-ok-001",
        timestamp="2025-01-01T00:00:00+00:00",
        phase="P3",
        mode="SHADOW",
        signal_type=GovernanceSignalType.SNAPSHOT_VERIFIED,
        curriculum_fingerprint="abc123",
        active_slice="slice_1",
        severity=DriftSeverity.NONE,
        status=DriftStatus.OK,
        violations=[],
        governance_action="LOGGED_ONLY",
        hypothetical={
            "would_allow_transition": True,
            "would_trigger_alert": False,
            "blocking_violations": [],
        },
    )


@pytest.fixture
def warn_governance_signal() -> GovernanceSignal:
    """Return a governance signal with WARN status."""
    return GovernanceSignal(
        signal_id="sig-warn-001",
        timestamp="2025-01-01T00:00:00+00:00",
        phase="P4",
        mode="SHADOW",
        signal_type=GovernanceSignalType.DRIFT_DETECTED,
        curriculum_fingerprint="def456",
        active_slice="slice_1",
        severity=DriftSeverity.PARAMETRIC,
        status=DriftStatus.WARN,
        violations=[
            Violation(
                code="PARAM_CHANGE_ATOMS",
                message="Atoms parameter changed",
                details={"before": 4, "after": 5},
            ),
        ],
        governance_action="LOGGED_ONLY",
        hypothetical={
            "would_allow_transition": True,
            "would_trigger_alert": True,
            "blocking_violations": [],
        },
    )


@pytest.fixture
def block_governance_signal() -> GovernanceSignal:
    """Return a governance signal with BLOCK status."""
    return GovernanceSignal(
        signal_id="sig-block-001",
        timestamp="2025-01-01T00:00:00+00:00",
        phase="P4",
        mode="SHADOW",
        signal_type=GovernanceSignalType.INVARIANT_VIOLATION,
        curriculum_fingerprint="ghi789",
        active_slice="slice_1",
        severity=DriftSeverity.SEMANTIC,
        status=DriftStatus.BLOCK,
        violations=[
            Violation(
                code="MONO_REGRESSION_ATOMS",
                message="Monotonicity regression on atoms",
                details={"axis": "atoms", "before": 5, "after": 4},
            ),
            Violation(
                code="GATE_REGRESS_COV_CI",
                message="Gate regression on coverage ci_lower_min",
                details={"before": 0.92, "after": 0.85},
            ),
        ],
        governance_action="LOGGED_ONLY",
        hypothetical={
            "would_allow_transition": False,
            "would_trigger_alert": True,
            "blocking_violations": ["MONO_REGRESSION_ATOMS", "GATE_REGRESS_COV_CI"],
        },
    )


@pytest.fixture
def ok_drift_event() -> DriftTimelineEvent:
    """Return a drift event with no drift."""
    return DriftTimelineEvent(
        event_id="evt-ok-001",
        timestamp="2025-01-01T00:00:00+00:00",
        phase="P4",
        mode="SHADOW",
        curriculum_fingerprint="abc123",
        slice_name="slice_1",
        baseline_slice_name="slice_1",
        drift_status=DriftStatus.OK,
        drift_severity=DriftSeverity.NONE,
        changed_params=[],
        monotonicity_violations=[],
        gate_evolution_violations=[],
        action_taken="LOGGED_ONLY",
    )


@pytest.fixture
def parametric_drift_event() -> DriftTimelineEvent:
    """Return a drift event with parametric change."""
    return DriftTimelineEvent(
        event_id="evt-param-001",
        timestamp="2025-01-01T00:01:00+00:00",
        phase="P4",
        mode="SHADOW",
        curriculum_fingerprint="def456",
        slice_name="slice_1",
        baseline_slice_name="slice_1",
        drift_status=DriftStatus.WARN,
        drift_severity=DriftSeverity.PARAMETRIC,
        changed_params=[
            ChangedParam(
                path="params.atoms",
                baseline=4,
                current=5,
                classification=DriftSeverity.PARAMETRIC,
                constraint="increasing",
                delta=1.0,
            ),
        ],
        monotonicity_violations=[],
        gate_evolution_violations=[],
        action_taken="LOGGED_ONLY",
    )


@pytest.fixture
def semantic_drift_event() -> DriftTimelineEvent:
    """Return a drift event with semantic violation."""
    return DriftTimelineEvent(
        event_id="evt-semantic-001",
        timestamp="2025-01-01T00:02:00+00:00",
        phase="P4",
        mode="SHADOW",
        curriculum_fingerprint="ghi789",
        slice_name="slice_1",
        baseline_slice_name="slice_1",
        drift_status=DriftStatus.BLOCK,
        drift_severity=DriftSeverity.SEMANTIC,
        changed_params=[
            ChangedParam(
                path="params.atoms",
                baseline=5,
                current=4,
                classification=DriftSeverity.SEMANTIC,
                constraint="increasing",
                delta=-1.0,
            ),
        ],
        monotonicity_violations=[
            MonotonicityViolation(
                axis="atoms",
                type="REGRESSION",
                before=5,
                after=4,
                delta=-1.0,
            ),
        ],
        gate_evolution_violations=[],
        action_taken="LOGGED_ONLY",
    )


@pytest.fixture
def basic_stability_report() -> Dict[str, Any]:
    """Return a basic stability report."""
    return {
        "schema_version": "1.0.0",
        "run_id": "run-001",
        "cycles_completed": 100,
        "metrics": {
            "success_rate": 0.85,
            "mean_rsi": 0.78,
        },
    }


@pytest.fixture
def basic_evidence() -> Dict[str, Any]:
    """Return a basic evidence pack."""
    return {
        "schema_version": "1.0.0",
        "bundle_id": "bundle-001",
        "artifacts": [],
    }


# -----------------------------------------------------------------------------
# P3 Stability Hook Tests
# -----------------------------------------------------------------------------

class TestP3StabilityHook:
    """Tests for P3 stability hook attachment."""

    def test_attach_ok_signal(
        self,
        basic_stability_report: Dict[str, Any],
        ok_governance_signal: GovernanceSignal,
    ) -> None:
        """OK signal should attach with GREEN status light."""
        result = attach_curriculum_governance_to_p3(
            basic_stability_report, ok_governance_signal
        )

        assert "curriculum_governance" in result
        cg = result["curriculum_governance"]

        assert cg["status_light"] == "GREEN"
        assert cg["curriculum_health_score"] == 1.0
        assert cg["severity"] == "NONE"
        assert cg["status"] == "OK"
        assert cg["violation_count"] == 0

    def test_attach_warn_signal(
        self,
        basic_stability_report: Dict[str, Any],
        warn_governance_signal: GovernanceSignal,
    ) -> None:
        """WARN signal should attach with YELLOW status light."""
        result = attach_curriculum_governance_to_p3(
            basic_stability_report, warn_governance_signal
        )

        assert "curriculum_governance" in result
        cg = result["curriculum_governance"]

        assert cg["status_light"] == "YELLOW"
        assert cg["curriculum_health_score"] == 0.7
        assert cg["severity"] == "PARAMETRIC"
        assert cg["status"] == "WARN"
        assert cg["violation_count"] == 1

    def test_attach_block_signal(
        self,
        basic_stability_report: Dict[str, Any],
        block_governance_signal: GovernanceSignal,
    ) -> None:
        """BLOCK signal should attach with RED status light."""
        result = attach_curriculum_governance_to_p3(
            basic_stability_report, block_governance_signal
        )

        assert "curriculum_governance" in result
        cg = result["curriculum_governance"]

        assert cg["status_light"] == "RED"
        assert cg["curriculum_health_score"] == 0.3
        assert cg["severity"] == "SEMANTIC"
        assert cg["status"] == "BLOCK"
        assert cg["violation_count"] == 2

    def test_non_mutating(
        self,
        basic_stability_report: Dict[str, Any],
        ok_governance_signal: GovernanceSignal,
    ) -> None:
        """Attachment should not mutate original report."""
        original_keys = set(basic_stability_report.keys())

        result = attach_curriculum_governance_to_p3(
            basic_stability_report, ok_governance_signal
        )

        # Original should be unchanged
        assert set(basic_stability_report.keys()) == original_keys
        assert "curriculum_governance" not in basic_stability_report

        # Result should have new key
        assert "curriculum_governance" in result

    def test_attach_timeline_empty(
        self,
        basic_stability_report: Dict[str, Any],
    ) -> None:
        """Empty timeline should attach with OK status."""
        result = attach_curriculum_timeline_to_p3(basic_stability_report, [])

        assert "curriculum_drift_timeline" in result
        tl = result["curriculum_drift_timeline"]

        assert tl["event_count"] == 0
        assert tl["max_severity"] == "NONE"
        assert tl["overall_status"] == "OK"

    def test_attach_timeline_with_events(
        self,
        basic_stability_report: Dict[str, Any],
        ok_drift_event: DriftTimelineEvent,
        parametric_drift_event: DriftTimelineEvent,
    ) -> None:
        """Timeline with events should compute summary."""
        events = [ok_drift_event, parametric_drift_event]
        result = attach_curriculum_timeline_to_p3(basic_stability_report, events)

        assert "curriculum_drift_timeline" in result
        tl = result["curriculum_drift_timeline"]

        assert tl["event_count"] == 2
        assert tl["max_severity"] == "PARAMETRIC"
        assert tl["overall_status"] == "WARN"
        assert tl["parametric_changes"] == 1
        assert len(tl["events"]) == 2


# -----------------------------------------------------------------------------
# Evidence Attachment Tests
# -----------------------------------------------------------------------------

class TestEvidenceAttachment:
    """Tests for evidence pack attachment."""

    def test_attach_to_evidence(
        self,
        basic_evidence: Dict[str, Any],
        ok_governance_signal: GovernanceSignal,
    ) -> None:
        """Signal should attach under governance.curriculum."""
        result = attach_curriculum_to_evidence(basic_evidence, ok_governance_signal)

        assert "governance" in result
        assert "curriculum" in result["governance"]

        curr = result["governance"]["curriculum"]
        assert "signal" in curr
        assert "timeline_summary" in curr
        assert "council_status" in curr
        assert curr["council_status"] == "OK"

    def test_attach_with_timeline(
        self,
        basic_evidence: Dict[str, Any],
        warn_governance_signal: GovernanceSignal,
        parametric_drift_event: DriftTimelineEvent,
    ) -> None:
        """Timeline should be included in attachment."""
        timeline = [parametric_drift_event]
        result = attach_curriculum_to_evidence(
            basic_evidence, warn_governance_signal, timeline
        )

        curr = result["governance"]["curriculum"]
        assert curr["timeline_summary"]["event_count"] == 1
        assert curr["timeline_summary"]["parametric_count"] == 1
        assert curr["council_status"] == "WARN"

    def test_attach_blocking_signal(
        self,
        basic_evidence: Dict[str, Any],
        block_governance_signal: GovernanceSignal,
        semantic_drift_event: DriftTimelineEvent,
    ) -> None:
        """Blocking signal should produce BLOCK council status."""
        timeline = [semantic_drift_event]
        result = attach_curriculum_to_evidence(
            basic_evidence, block_governance_signal, timeline
        )

        curr = result["governance"]["curriculum"]
        assert curr["council_status"] == "BLOCK"
        assert curr["council_classification"]["would_block_uplift"] is True

    def test_preserves_existing_governance(
        self,
        ok_governance_signal: GovernanceSignal,
    ) -> None:
        """Attachment should preserve existing governance keys."""
        evidence = {
            "governance": {
                "budget": {"status": "OK"},
            },
        }

        result = attach_curriculum_to_evidence(evidence, ok_governance_signal)

        assert "budget" in result["governance"]
        assert result["governance"]["budget"]["status"] == "OK"
        assert "curriculum" in result["governance"]

    def test_non_mutating(
        self,
        basic_evidence: Dict[str, Any],
        ok_governance_signal: GovernanceSignal,
    ) -> None:
        """Attachment should not mutate original evidence."""
        result = attach_curriculum_to_evidence(basic_evidence, ok_governance_signal)

        assert "governance" not in basic_evidence
        assert "governance" in result


# -----------------------------------------------------------------------------
# Council Classification Tests
# -----------------------------------------------------------------------------

class TestCouncilClassification:
    """Tests for council classification."""

    def test_classify_ok(self, ok_governance_signal: GovernanceSignal) -> None:
        """OK signal should classify as OK."""
        status = council_classify_curriculum(ok_governance_signal)
        assert status == "OK"

    def test_classify_warn_from_parametric(
        self, warn_governance_signal: GovernanceSignal
    ) -> None:
        """PARAMETRIC severity should classify as WARN."""
        status = council_classify_curriculum(warn_governance_signal)
        assert status == "WARN"

    def test_classify_block_from_semantic(
        self, block_governance_signal: GovernanceSignal
    ) -> None:
        """SEMANTIC severity should classify as BLOCK."""
        status = council_classify_curriculum(block_governance_signal)
        assert status == "BLOCK"

    def test_classify_block_from_invariant_violation(
        self, ok_governance_signal: GovernanceSignal
    ) -> None:
        """INVARIANT_VIOLATION signal type should classify as BLOCK."""
        # Modify signal type
        signal = GovernanceSignal(
            signal_id=ok_governance_signal.signal_id,
            timestamp=ok_governance_signal.timestamp,
            phase=ok_governance_signal.phase,
            mode=ok_governance_signal.mode,
            signal_type=GovernanceSignalType.INVARIANT_VIOLATION,
            curriculum_fingerprint=ok_governance_signal.curriculum_fingerprint,
            active_slice=ok_governance_signal.active_slice,
            severity=DriftSeverity.NONE,
            status=DriftStatus.OK,
            violations=[],
            governance_action="LOGGED_ONLY",
        )
        status = council_classify_curriculum(signal)
        assert status == "BLOCK"

    def test_classify_block_from_hypothetical(
        self, ok_governance_signal: GovernanceSignal
    ) -> None:
        """Non-allowing hypothetical should classify as BLOCK."""
        signal = GovernanceSignal(
            signal_id=ok_governance_signal.signal_id,
            timestamp=ok_governance_signal.timestamp,
            phase=ok_governance_signal.phase,
            mode=ok_governance_signal.mode,
            signal_type=ok_governance_signal.signal_type,
            curriculum_fingerprint=ok_governance_signal.curriculum_fingerprint,
            active_slice=ok_governance_signal.active_slice,
            severity=DriftSeverity.NONE,
            status=DriftStatus.OK,
            violations=[],
            governance_action="LOGGED_ONLY",
            hypothetical={"would_allow_transition": False},
        )
        status = council_classify_curriculum(signal)
        assert status == "BLOCK"

    def test_classify_block_from_timeline_semantic(
        self,
        ok_governance_signal: GovernanceSignal,
        semantic_drift_event: DriftTimelineEvent,
    ) -> None:
        """Semantic event in timeline should classify as BLOCK."""
        status = council_classify_curriculum(ok_governance_signal, [semantic_drift_event])
        assert status == "BLOCK"

    def test_classify_warn_from_timeline_parametric(
        self,
        ok_governance_signal: GovernanceSignal,
        parametric_drift_event: DriftTimelineEvent,
    ) -> None:
        """Parametric event in timeline should classify as WARN."""
        status = council_classify_curriculum(ok_governance_signal, [parametric_drift_event])
        assert status == "WARN"

    def test_classify_from_dict_ok(self, ok_governance_signal: GovernanceSignal) -> None:
        """Dict-based classification should work for OK."""
        signal_dict = ok_governance_signal.to_dict()
        status = council_classify_curriculum_from_dict(signal_dict)
        assert status == "OK"

    def test_classify_from_dict_block(
        self, block_governance_signal: GovernanceSignal
    ) -> None:
        """Dict-based classification should work for BLOCK."""
        signal_dict = block_governance_signal.to_dict()
        status = council_classify_curriculum_from_dict(signal_dict)
        assert status == "BLOCK"

    def test_classify_from_dict_with_timeline(
        self,
        ok_governance_signal: GovernanceSignal,
        semantic_drift_event: DriftTimelineEvent,
    ) -> None:
        """Dict-based classification should check timeline."""
        signal_dict = ok_governance_signal.to_dict()
        timeline_dicts = [semantic_drift_event.to_dict()]
        status = council_classify_curriculum_from_dict(signal_dict, timeline_dicts)
        assert status == "BLOCK"


# -----------------------------------------------------------------------------
# Batch Summarization Tests
# -----------------------------------------------------------------------------

class TestBatchSummarization:
    """Tests for batch summarization."""

    def test_summarize_empty(self) -> None:
        """Empty signals should produce OK summary."""
        summary = summarize_curriculum_for_council([])

        assert summary["signal_count"] == 0
        assert summary["overall_status"] == "OK"
        assert summary["curriculum_health"] == "HEALTHY"

    def test_summarize_all_ok(
        self, ok_governance_signal: GovernanceSignal
    ) -> None:
        """All OK signals should produce OK summary."""
        signals = [ok_governance_signal, ok_governance_signal]
        summary = summarize_curriculum_for_council(signals)

        assert summary["signal_count"] == 2
        assert summary["overall_status"] == "OK"
        assert summary["curriculum_health"] == "HEALTHY"
        assert summary["ok_signals"] == 2

    def test_summarize_mixed_warn(
        self,
        ok_governance_signal: GovernanceSignal,
        warn_governance_signal: GovernanceSignal,
    ) -> None:
        """Mixed OK/WARN should produce WARN summary."""
        signals = [ok_governance_signal, warn_governance_signal]
        summary = summarize_curriculum_for_council(signals)

        assert summary["signal_count"] == 2
        assert summary["overall_status"] == "WARN"
        assert summary["curriculum_health"] == "DEGRADED"
        assert summary["ok_signals"] == 1
        assert summary["warn_signals"] == 1

    def test_summarize_any_block(
        self,
        ok_governance_signal: GovernanceSignal,
        warn_governance_signal: GovernanceSignal,
        block_governance_signal: GovernanceSignal,
    ) -> None:
        """Any BLOCK signal should produce BLOCK summary."""
        signals = [ok_governance_signal, warn_governance_signal, block_governance_signal]
        summary = summarize_curriculum_for_council(signals)

        assert summary["signal_count"] == 3
        assert summary["overall_status"] == "BLOCK"
        assert summary["curriculum_health"] == "CRITICAL"
        assert summary["block_signals"] == 1

    def test_summarize_counts_violations(
        self,
        warn_governance_signal: GovernanceSignal,
        block_governance_signal: GovernanceSignal,
    ) -> None:
        """Summary should count violations correctly."""
        signals = [warn_governance_signal, block_governance_signal]
        summary = summarize_curriculum_for_council(signals)

        assert summary["violations_total"] == 3  # 1 + 2
        assert summary["parametric_changes"] == 1
        assert summary["semantic_violations"] == 1


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases."""

    def test_attach_to_empty_stability_report(
        self, ok_governance_signal: GovernanceSignal
    ) -> None:
        """Attachment to empty report should work."""
        result = attach_curriculum_governance_to_p3({}, ok_governance_signal)
        assert "curriculum_governance" in result

    def test_attach_to_empty_evidence(
        self, ok_governance_signal: GovernanceSignal
    ) -> None:
        """Attachment to empty evidence should work."""
        result = attach_curriculum_to_evidence({}, ok_governance_signal)
        assert "governance" in result
        assert "curriculum" in result["governance"]

    def test_classify_signal_with_no_hypothetical(self) -> None:
        """Classification should handle missing hypothetical."""
        signal = GovernanceSignal(
            signal_id="sig-001",
            timestamp="2025-01-01T00:00:00+00:00",
            phase="P3",
            mode="SHADOW",
            signal_type=GovernanceSignalType.SNAPSHOT_CAPTURED,
            curriculum_fingerprint="abc",
            active_slice="slice_1",
            severity=DriftSeverity.NONE,
            status=DriftStatus.OK,
            violations=[],
            governance_action="LOGGED_ONLY",
            hypothetical=None,
        )
        status = council_classify_curriculum(signal)
        assert status == "OK"

    def test_classify_with_regress_violation_code(self) -> None:
        """REGRESS in violation code should trigger BLOCK."""
        signal = GovernanceSignal(
            signal_id="sig-001",
            timestamp="2025-01-01T00:00:00+00:00",
            phase="P3",
            mode="SHADOW",
            signal_type=GovernanceSignalType.DRIFT_DETECTED,
            curriculum_fingerprint="abc",
            active_slice="slice_1",
            severity=DriftSeverity.PARAMETRIC,  # Not SEMANTIC
            status=DriftStatus.WARN,
            violations=[
                Violation(
                    code="PARAM_REGRESS_ATOMS",  # Contains REGRESS
                    message="Parameter regression",
                ),
            ],
            governance_action="LOGGED_ONLY",
        )
        status = council_classify_curriculum(signal)
        assert status == "BLOCK"

    def test_timeline_limits_events(
        self, basic_stability_report: Dict[str, Any]
    ) -> None:
        """Timeline should limit to last 10 events."""
        events = [
            DriftTimelineEvent(
                event_id=f"evt-{i}",
                timestamp=f"2025-01-01T00:{i:02d}:00+00:00",
                phase="P4",
                mode="SHADOW",
                curriculum_fingerprint="abc",
                slice_name="slice_1",
                baseline_slice_name="slice_1",
                drift_status=DriftStatus.OK,
                drift_severity=DriftSeverity.NONE,
                action_taken="LOGGED_ONLY",
            )
            for i in range(15)
        ]

        result = attach_curriculum_timeline_to_p3(basic_stability_report, events)
        tl = result["curriculum_drift_timeline"]

        assert tl["event_count"] == 15
        assert len(tl["events"]) == 10  # Limited to last 10


# -----------------------------------------------------------------------------
# CTRPK Tests
# -----------------------------------------------------------------------------

class TestCTRPKBandClassification:
    """Tests for CTRPK band classification (GREEN/YELLOW/RED)."""

    def test_ctrpk_green_threshold(self) -> None:
        """CTRPK < 1.0 should produce GREEN status light."""
        # 5 transition requests in 10000 cycles = 0.5 CTRPK
        ctrpk = compute_ctrpk(transition_requests=5, total_cycles=10000)
        assert ctrpk == 0.5

        status_light = ctrpk_to_status_light(ctrpk)
        assert status_light == "GREEN"

        # Edge case: exactly at boundary (0.99)
        ctrpk_edge = compute_ctrpk(transition_requests=99, total_cycles=100000)
        assert ctrpk_edge == 0.99
        assert ctrpk_to_status_light(ctrpk_edge) == "GREEN"

        # Zero transitions = 0.0 CTRPK = GREEN
        ctrpk_zero = compute_ctrpk(transition_requests=0, total_cycles=10000)
        assert ctrpk_zero == 0.0
        assert ctrpk_to_status_light(ctrpk_zero) == "GREEN"

    def test_ctrpk_yellow_threshold(self) -> None:
        """CTRPK 1.0-5.0 should produce YELLOW status light."""
        # 20 transition requests in 10000 cycles = 2.0 CTRPK
        ctrpk = compute_ctrpk(transition_requests=20, total_cycles=10000)
        assert ctrpk == 2.0

        status_light = ctrpk_to_status_light(ctrpk)
        assert status_light == "YELLOW"

        # Lower bound (exactly 1.0)
        ctrpk_lower = compute_ctrpk(transition_requests=10, total_cycles=10000)
        assert ctrpk_lower == 1.0
        assert ctrpk_to_status_light(ctrpk_lower) == "YELLOW"

        # Upper bound (exactly 5.0)
        ctrpk_upper = compute_ctrpk(transition_requests=50, total_cycles=10000)
        assert ctrpk_upper == 5.0
        assert ctrpk_to_status_light(ctrpk_upper) == "YELLOW"

    def test_ctrpk_red_threshold(self) -> None:
        """CTRPK > 5.0 should produce RED status light."""
        # 60 transition requests in 10000 cycles = 6.0 CTRPK
        ctrpk = compute_ctrpk(transition_requests=60, total_cycles=10000)
        assert ctrpk == 6.0

        status_light = ctrpk_to_status_light(ctrpk)
        assert status_light == "RED"

        # Just above threshold (5.01)
        ctrpk_edge = compute_ctrpk(transition_requests=501, total_cycles=100000)
        assert ctrpk_edge == 5.01
        assert ctrpk_to_status_light(ctrpk_edge) == "RED"

        # High stress scenario (10.0 CTRPK)
        ctrpk_high = compute_ctrpk(transition_requests=100, total_cycles=10000)
        assert ctrpk_high == 10.0
        assert ctrpk_to_status_light(ctrpk_high) == "RED"

    def test_council_semantic_override_forces_block(self) -> None:
        """Semantic violations should force BLOCK regardless of CTRPK value."""
        # Low CTRPK (0.5 = GREEN) but semantic violation present
        council_status = council_classify_ctrpk(
            ctrpk=0.5,
            semantic_violations=1,  # Forces BLOCK
            blocked_requests=0,
            trend_direction="STABLE",
        )
        assert council_status == "BLOCK"

        # Medium CTRPK (2.0 = YELLOW) with semantic violation
        council_status = council_classify_ctrpk(
            ctrpk=2.0,
            semantic_violations=2,
            blocked_requests=0,
            trend_direction="STABLE",
        )
        assert council_status == "BLOCK"

        # Verify without semantic violations it would be OK/WARN
        council_status_no_semantic = council_classify_ctrpk(
            ctrpk=0.5,
            semantic_violations=0,
            blocked_requests=0,
            trend_direction="STABLE",
        )
        assert council_status_no_semantic == "OK"

    def test_ctrpk_trend_computation(self) -> None:
        """Trend should compute IMPROVING/STABLE/DEGRADING correctly."""
        # IMPROVING: 1h CTRPK much lower than 24h (delta < -0.5)
        trend = compute_ctrpk_trend(ctrpk_1h=1.0, ctrpk_24h=3.0)
        assert trend == "IMPROVING"

        # STABLE: 1h and 24h CTRPK similar (|delta| <= 0.5)
        trend = compute_ctrpk_trend(ctrpk_1h=2.0, ctrpk_24h=2.3)
        assert trend == "STABLE"

        trend = compute_ctrpk_trend(ctrpk_1h=2.5, ctrpk_24h=2.0)
        assert trend == "STABLE"

        # DEGRADING: 1h CTRPK much higher than 24h (delta > 0.5)
        trend = compute_ctrpk_trend(ctrpk_1h=4.0, ctrpk_24h=2.0)
        assert trend == "DEGRADING"

        # Edge cases at threshold boundaries
        trend_improving_edge = compute_ctrpk_trend(ctrpk_1h=1.0, ctrpk_24h=1.51)
        assert trend_improving_edge == "IMPROVING"

        trend_degrading_edge = compute_ctrpk_trend(ctrpk_1h=1.51, ctrpk_24h=1.0)
        assert trend_degrading_edge == "DEGRADING"


class TestCTRPKIntegration:
    """Tests for CTRPK integration with evidence and summaries."""

    def test_build_ctrpk_summary_includes_all_fields(self) -> None:
        """Summary should include all required fields."""
        summary = build_ctrpk_summary(
            transition_requests=30,
            total_cycles=10000,
            measurement_window_minutes=60,
            blocked_requests=2,
            successful_transitions=28,
            ctrpk_1h=3.0,
            ctrpk_24h=2.0,
            semantic_violations=0,
        )

        assert summary["ctrpk"] == 3.0
        assert summary["status_light"] == "YELLOW"
        assert summary["transition_requests"] == 30
        assert summary["total_cycles"] == 10000
        assert summary["measurement_window_minutes"] == 60
        assert summary["blocked_requests"] == 2
        assert summary["successful_transitions"] == 28
        assert summary["trend"]["direction"] == "DEGRADING"
        assert summary["trend"]["ctrpk_1h"] == 3.0
        assert summary["trend"]["ctrpk_24h"] == 2.0
        assert summary["trend"]["delta_vs_baseline"] == 1.0  # 3.0 - 2.0

    def test_build_ctrpk_compact_minimal(self) -> None:
        """Compact CTRPK should have minimal fields."""
        compact = build_ctrpk_compact(
            transition_requests=5,
            total_cycles=10000,
            trend_direction="STABLE",
        )

        assert compact["value"] == 0.5
        assert compact["status"] == "OK"
        assert compact["window_cycles"] == 10000
        assert compact["transition_requests"] == 5
        assert compact["trend"] == "STABLE"

    def test_attach_ctrpk_to_evidence_creates_path(self) -> None:
        """CTRPK attachment should create governance.curriculum.ctrpk path."""
        evidence: Dict[str, Any] = {"bundle_id": "test-001"}

        result = attach_ctrpk_to_evidence(
            evidence=evidence,
            transition_requests=20,
            total_cycles=10000,
            trend_direction="STABLE",
        )

        assert "governance" in result
        assert "curriculum" in result["governance"]
        assert "ctrpk" in result["governance"]["curriculum"]

        ctrpk = result["governance"]["curriculum"]["ctrpk"]
        assert ctrpk["value"] == 2.0
        assert ctrpk["status"] == "WARN"

    def test_attach_ctrpk_preserves_existing_evidence(self) -> None:
        """CTRPK attachment should preserve existing evidence fields."""
        evidence: Dict[str, Any] = {
            "bundle_id": "test-001",
            "governance": {
                "curriculum": {
                    "signal": {"status": "OK"},
                },
                "budget": {"remaining": 100},
            },
        }

        result = attach_ctrpk_to_evidence(
            evidence=evidence,
            transition_requests=5,
            total_cycles=10000,
        )

        # Original fields preserved
        assert result["governance"]["curriculum"]["signal"]["status"] == "OK"
        assert result["governance"]["budget"]["remaining"] == 100
        # CTRPK added
        assert "ctrpk" in result["governance"]["curriculum"]

    def test_ctrpk_zero_cycles_returns_zero(self) -> None:
        """Zero cycles should return 0.0 CTRPK (not division error)."""
        ctrpk = compute_ctrpk(transition_requests=10, total_cycles=0)
        assert ctrpk == 0.0

        ctrpk_negative = compute_ctrpk(transition_requests=10, total_cycles=-5)
        assert ctrpk_negative == 0.0

    def test_council_ctrpk_degrading_trend_block(self) -> None:
        """DEGRADING trend with CTRPK > 3.0 should BLOCK."""
        # CTRPK 4.0 with DEGRADING trend
        status = council_classify_ctrpk(
            ctrpk=4.0,
            semantic_violations=0,
            blocked_requests=0,
            trend_direction="DEGRADING",
        )
        assert status == "BLOCK"

        # Same CTRPK with STABLE trend = WARN (not BLOCK)
        status_stable = council_classify_ctrpk(
            ctrpk=4.0,
            semantic_violations=0,
            blocked_requests=0,
            trend_direction="STABLE",
        )
        assert status_stable == "WARN"

    def test_council_ctrpk_blocked_requests_warn(self) -> None:
        """Blocked requests should trigger WARN even with low CTRPK."""
        status = council_classify_ctrpk(
            ctrpk=0.5,  # GREEN threshold
            semantic_violations=0,
            blocked_requests=1,  # But has blocked requests
            trend_direction="STABLE",
        )
        assert status == "WARN"
