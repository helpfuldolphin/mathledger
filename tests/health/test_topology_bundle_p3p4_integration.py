"""Integration tests for topology bundle P3/P4 and evidence binding.

STATUS: PHASE X — TOPOLOGY/BUNDLE P3/P4 INTEGRATION TESTS

Tests:
1. P3 stability summary correctness
2. P3 stability report attachment
3. P4 calibration summary correctness
4. P4 calibration report attachment
5. Evidence attachment with governance signal
6. First Light signal extraction
7. Non-mutation guarantees
8. JSON serialization safety

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No tests depend on governance enforcement
- Tests validate read-only, side-effect-free behavior
"""

import copy
import json
from typing import Any, Dict

import pytest


class TestP3StabilitySummary:
    """Test P3 stability summary building."""

    def _make_mock_tile(
        self,
        topology_stability: str = "STABLE",
        bundle_stability: str = "VALID",
        joint_status: str = "ALIGNED",
        conflict_codes: list = None,
        status_light: str = "GREEN",
    ) -> Dict[str, Any]:
        """Create mock console tile for testing."""
        return {
            "schema_version": "1.0.0",
            "status_light": status_light,
            "topology_stability": topology_stability,
            "bundle_stability": bundle_stability,
            "cross_system_consistency": True,
            "joint_status": joint_status,
            "conflict_codes": conflict_codes or [],
            "headline": f"Topology: {topology_stability} | Bundle: {bundle_stability}",
        }

    def test_01_p3_summary_has_required_fields(self) -> None:
        """P3 summary contains all required fields."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_summary_for_p3,
        )

        tile = self._make_mock_tile()
        summary = build_topology_bundle_summary_for_p3(tile)

        assert "topology_stability" in summary
        assert "bundle_stability" in summary
        assert "joint_status" in summary
        assert "conflict_codes" in summary
        assert "status_light" in summary

    def test_02_p3_summary_extracts_values_correctly(self) -> None:
        """P3 summary extracts correct values from tile."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_summary_for_p3,
        )

        tile = self._make_mock_tile(
            topology_stability="DRIFTING",
            bundle_stability="ATTENTION",
            joint_status="TENSION",
            conflict_codes=["XCOR-WARN-001"],
            status_light="YELLOW",
        )
        summary = build_topology_bundle_summary_for_p3(tile)

        assert summary["topology_stability"] == "DRIFTING"
        assert summary["bundle_stability"] == "ATTENTION"
        assert summary["joint_status"] == "TENSION"
        assert summary["conflict_codes"] == ["XCOR-WARN-001"]
        assert summary["status_light"] == "YELLOW"

    def test_03_p3_summary_handles_missing_fields(self) -> None:
        """P3 summary handles missing fields with defaults."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_summary_for_p3,
        )

        tile = {}  # Empty tile
        summary = build_topology_bundle_summary_for_p3(tile)

        assert summary["topology_stability"] == "UNKNOWN"
        assert summary["bundle_stability"] == "UNKNOWN"
        assert summary["joint_status"] == "UNKNOWN"
        assert summary["conflict_codes"] == []
        assert summary["status_light"] == "GREEN"


class TestP3StabilityReportAttachment:
    """Test P3 stability report attachment."""

    def _make_mock_tile(self) -> Dict[str, Any]:
        """Create mock console tile for testing."""
        return {
            "schema_version": "1.0.0",
            "status_light": "YELLOW",
            "topology_stability": "DRIFTING",
            "bundle_stability": "ATTENTION",
            "cross_system_consistency": True,
            "joint_status": "TENSION",
            "conflict_codes": ["XCOR-WARN-001"],
            "headline": "Topology: DRIFTING | Bundle: ATTENTION",
        }

    def test_01_p3_report_attachment_adds_summary(self) -> None:
        """P3 report attachment adds topology_bundle_summary."""
        from backend.health.topology_bundle_adapter import (
            add_topology_bundle_to_p3_stability_report,
        )

        report = {"schema_version": "1.0.0", "run_id": "test_run_001"}
        tile = self._make_mock_tile()

        enriched = add_topology_bundle_to_p3_stability_report(report, tile)

        assert "topology_bundle_summary" in enriched
        assert enriched["topology_bundle_summary"]["topology_stability"] == "DRIFTING"

    def test_02_p3_report_attachment_is_non_mutating(self) -> None:
        """P3 report attachment does not modify original report."""
        from backend.health.topology_bundle_adapter import (
            add_topology_bundle_to_p3_stability_report,
        )

        report = {"schema_version": "1.0.0", "run_id": "test_run_001"}
        report_copy = copy.deepcopy(report)
        tile = self._make_mock_tile()

        add_topology_bundle_to_p3_stability_report(report, tile)

        assert report == report_copy
        assert "topology_bundle_summary" not in report

    def test_03_p3_report_preserves_existing_fields(self) -> None:
        """P3 report attachment preserves existing fields."""
        from backend.health.topology_bundle_adapter import (
            add_topology_bundle_to_p3_stability_report,
        )

        report = {
            "schema_version": "1.0.0",
            "run_id": "test_run_001",
            "existing_field": "existing_value",
        }
        tile = self._make_mock_tile()

        enriched = add_topology_bundle_to_p3_stability_report(report, tile)

        assert enriched["existing_field"] == "existing_value"
        assert enriched["run_id"] == "test_run_001"


class TestP4CalibrationSummary:
    """Test P4 calibration summary building."""

    def _make_mock_tile(
        self,
        topology_stability: str = "STABLE",
        bundle_stability: str = "VALID",
        status_light: str = "GREEN",
        conflict_codes: list = None,
        cross_system_consistency: bool = True,
    ) -> Dict[str, Any]:
        """Create mock console tile for testing."""
        return {
            "schema_version": "1.0.0",
            "status_light": status_light,
            "topology_stability": topology_stability,
            "bundle_stability": bundle_stability,
            "cross_system_consistency": cross_system_consistency,
            "joint_status": "ALIGNED",
            "conflict_codes": conflict_codes or [],
            "headline": f"Topology: {topology_stability} | Bundle: {bundle_stability}",
        }

    def _make_mock_joint_view(
        self,
        topology_mode: str = "STABLE",
    ) -> Dict[str, Any]:
        """Create mock joint view for testing."""
        return {
            "topology_snapshot": {"topology_mode": topology_mode},
            "bundle_snapshot": {"bundle_status": "VALID"},
            "alignment_status": {"overall_status": "ALIGNED"},
        }

    def _make_mock_consistency_result(
        self,
        consistent: bool = True,
    ) -> Dict[str, Any]:
        """Create mock consistency result for testing."""
        return {"consistent": consistent, "status": "OK" if consistent else "WARN"}

    def test_01_p4_calibration_has_required_fields(self) -> None:
        """P4 calibration contains all required fields."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_calibration_for_p4,
        )

        tile = self._make_mock_tile()
        calibration = build_topology_bundle_calibration_for_p4(tile)

        assert "topology_mode" in calibration
        assert "bundle_integration_status" in calibration
        assert "cross_system_consistency" in calibration
        assert "xcor_codes" in calibration
        assert "status_light" in calibration
        assert "structural_notes" in calibration

    def test_02_p4_calibration_extracts_topology_mode_from_joint_view(self) -> None:
        """P4 calibration extracts raw topology_mode from joint_view."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_calibration_for_p4,
        )

        tile = self._make_mock_tile(topology_stability="DRIFTING")
        joint_view = self._make_mock_joint_view(topology_mode="DRIFT")

        calibration = build_topology_bundle_calibration_for_p4(tile, joint_view=joint_view)

        assert calibration["topology_mode"] == "DRIFT"

    def test_03_p4_calibration_derives_topology_mode_from_stability(self) -> None:
        """P4 calibration derives topology_mode from stability if no joint_view."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_calibration_for_p4,
        )

        tile = self._make_mock_tile(topology_stability="TURBULENT")

        calibration = build_topology_bundle_calibration_for_p4(tile)

        assert calibration["topology_mode"] == "TURBULENT"

    def test_04_p4_calibration_maps_bundle_integration_status(self) -> None:
        """P4 calibration maps bundle stability to integration status."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_calibration_for_p4,
        )

        # VALID -> INTEGRATED
        tile = self._make_mock_tile(bundle_stability="VALID")
        calibration = build_topology_bundle_calibration_for_p4(tile)
        assert calibration["bundle_integration_status"] == "INTEGRATED"

        # ATTENTION -> PARTIAL
        tile = self._make_mock_tile(bundle_stability="ATTENTION")
        calibration = build_topology_bundle_calibration_for_p4(tile)
        assert calibration["bundle_integration_status"] == "PARTIAL"

        # BROKEN -> BROKEN
        tile = self._make_mock_tile(bundle_stability="BROKEN")
        calibration = build_topology_bundle_calibration_for_p4(tile)
        assert calibration["bundle_integration_status"] == "BROKEN"

    def test_05_p4_calibration_extracts_xcor_codes_only(self) -> None:
        """P4 calibration extracts only XCOR-* codes."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_calibration_for_p4,
        )

        tile = self._make_mock_tile(
            conflict_codes=["XCOR-WARN-001", "TOPO-WARN-001", "XCOR-CRIT-001", "BNDL-OK-001"]
        )

        calibration = build_topology_bundle_calibration_for_p4(tile)

        assert calibration["xcor_codes"] == ["XCOR-WARN-001", "XCOR-CRIT-001"]

    def test_06_p4_calibration_builds_structural_notes(self) -> None:
        """P4 calibration builds structural notes for anomalies."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_calibration_for_p4,
        )

        tile = self._make_mock_tile(
            topology_stability="DRIFTING",
            bundle_stability="ATTENTION",
            cross_system_consistency=False,
            conflict_codes=["XCOR-WARN-001"],
        )
        joint_view = self._make_mock_joint_view(topology_mode="DRIFT")
        consistency = self._make_mock_consistency_result(consistent=False)

        calibration = build_topology_bundle_calibration_for_p4(
            tile, joint_view=joint_view, consistency_result=consistency
        )

        notes = calibration["structural_notes"]
        assert any("drift" in note.lower() for note in notes)
        assert any("incomplete" in note.lower() for note in notes)
        assert any("consistency" in note.lower() for note in notes)
        assert any("XCOR-WARN-001" in note for note in notes)


class TestP4CalibrationReportAttachment:
    """Test P4 calibration report attachment."""

    def _make_mock_tile(self) -> Dict[str, Any]:
        """Create mock console tile for testing."""
        return {
            "schema_version": "1.0.0",
            "status_light": "YELLOW",
            "topology_stability": "DRIFTING",
            "bundle_stability": "ATTENTION",
            "cross_system_consistency": False,
            "joint_status": "TENSION",
            "conflict_codes": ["XCOR-WARN-001"],
            "headline": "Topology: DRIFTING | Bundle: ATTENTION",
        }

    def _make_mock_joint_view(self) -> Dict[str, Any]:
        """Create mock joint view for testing."""
        return {
            "topology_snapshot": {"topology_mode": "DRIFT"},
            "bundle_snapshot": {"bundle_status": "WARN"},
            "alignment_status": {"overall_status": "TENSION"},
        }

    def test_01_p4_report_attachment_adds_calibration(self) -> None:
        """P4 report attachment adds topology_bundle_calibration."""
        from backend.health.topology_bundle_adapter import (
            add_topology_bundle_to_p4_calibration_report,
        )

        report = {"schema_version": "1.0.0", "run_id": "test_run_001"}
        tile = self._make_mock_tile()
        joint_view = self._make_mock_joint_view()

        enriched = add_topology_bundle_to_p4_calibration_report(
            report, tile, joint_view=joint_view
        )

        assert "topology_bundle_calibration" in enriched
        assert enriched["topology_bundle_calibration"]["topology_mode"] == "DRIFT"

    def test_02_p4_report_attachment_is_non_mutating(self) -> None:
        """P4 report attachment does not modify original report."""
        from backend.health.topology_bundle_adapter import (
            add_topology_bundle_to_p4_calibration_report,
        )

        report = {"schema_version": "1.0.0", "run_id": "test_run_001"}
        report_copy = copy.deepcopy(report)
        tile = self._make_mock_tile()

        add_topology_bundle_to_p4_calibration_report(report, tile)

        assert report == report_copy
        assert "topology_bundle_calibration" not in report


class TestEvidenceAttachmentWithSignal:
    """Test evidence attachment with governance signal."""

    def _make_mock_tile(self) -> Dict[str, Any]:
        """Create mock console tile for testing."""
        return {
            "schema_version": "1.0.0",
            "status_light": "YELLOW",
            "topology_stability": "DRIFTING",
            "bundle_stability": "ATTENTION",
            "cross_system_consistency": True,
            "joint_status": "TENSION",
            "conflict_codes": ["XCOR-WARN-001"],
            "headline": "Topology: DRIFTING | Bundle: ATTENTION",
        }

    def _make_mock_signal(self) -> Dict[str, Any]:
        """Create mock governance signal for testing."""
        return {
            "schema_version": "1.0.0",
            "signal_type": "topology_bundle",
            "status": "WARN",
            "governance_status": "WARN",
            "governance_alignment": "TENSION",
            "topology_status": "WARN",
            "bundle_status": "WARN",
            "conflict": False,
            "reasons": [
                "[Topology] Topology drift detected",
                "[Bundle] Bundle chain warning detected",
                "[Topology] Topology-bundle alignment: TENSION",
            ],
            "safe_for_policy_update": True,
            "safe_for_promotion": True,
        }

    def test_01_evidence_attachment_includes_tile_fields(self) -> None:
        """Evidence attachment includes tile fields."""
        from backend.health.topology_bundle_adapter import (
            attach_topology_bundle_to_evidence,
        )

        evidence = {"timestamp": "2025-12-10T00:00:00Z"}
        tile = self._make_mock_tile()

        enriched = attach_topology_bundle_to_evidence(evidence, tile)

        summary = enriched["governance"]["topology_bundle"]
        assert summary["status_light"] == "YELLOW"
        assert summary["topology_stability"] == "DRIFTING"
        assert summary["bundle_stability"] == "ATTENTION"
        assert summary["joint_status"] == "TENSION"
        assert summary["conflict_codes"] == ["XCOR-WARN-001"]

    def test_02_evidence_attachment_includes_signal_fields(self) -> None:
        """Evidence attachment includes governance signal fields."""
        from backend.health.topology_bundle_adapter import (
            attach_topology_bundle_to_evidence,
        )

        evidence = {"timestamp": "2025-12-10T00:00:00Z"}
        tile = self._make_mock_tile()
        signal = self._make_mock_signal()

        enriched = attach_topology_bundle_to_evidence(evidence, tile, governance_signal=signal)

        summary = enriched["governance"]["topology_bundle"]
        assert summary["governance_status"] == "WARN"
        assert summary["governance_alignment"] == "TENSION"
        assert summary["topology_signal_status"] == "WARN"
        assert summary["bundle_signal_status"] == "WARN"
        assert summary["conflict"] is False
        assert len(summary["reasons"]) == 3
        assert all(r.startswith("[Topology]") or r.startswith("[Bundle]") for r in summary["reasons"])

    def test_03_evidence_attachment_without_signal(self) -> None:
        """Evidence attachment works without governance signal."""
        from backend.health.topology_bundle_adapter import (
            attach_topology_bundle_to_evidence,
        )

        evidence = {"timestamp": "2025-12-10T00:00:00Z"}
        tile = self._make_mock_tile()

        enriched = attach_topology_bundle_to_evidence(evidence, tile, governance_signal=None)

        summary = enriched["governance"]["topology_bundle"]
        # Should not have signal-specific fields
        assert "governance_status" not in summary
        assert "reasons" not in summary

    def test_04_evidence_attachment_is_non_mutating(self) -> None:
        """Evidence attachment does not modify original evidence."""
        from backend.health.topology_bundle_adapter import (
            attach_topology_bundle_to_evidence,
        )

        evidence = {"timestamp": "2025-12-10T00:00:00Z", "data": {"key": "value"}}
        evidence_copy = copy.deepcopy(evidence)
        tile = self._make_mock_tile()
        signal = self._make_mock_signal()

        attach_topology_bundle_to_evidence(evidence, tile, governance_signal=signal)

        assert evidence == evidence_copy
        assert "governance" not in evidence

    def test_05_evidence_attachment_serializes_to_json(self) -> None:
        """Evidence with attachment serializes to JSON."""
        from backend.health.topology_bundle_adapter import (
            attach_topology_bundle_to_evidence,
        )

        evidence = {"timestamp": "2025-12-10T00:00:00Z"}
        tile = self._make_mock_tile()
        signal = self._make_mock_signal()

        enriched = attach_topology_bundle_to_evidence(evidence, tile, governance_signal=signal)

        # Should not raise
        json_str = json.dumps(enriched)
        assert len(json_str) > 0

        # Round-trip
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "topology_bundle" in parsed["governance"]


class TestFirstLightSignalExtraction:
    """Test First Light signal extraction."""

    def _make_mock_tile(
        self,
        status_light: str = "GREEN",
        topology_stability: str = "STABLE",
        bundle_stability: str = "VALID",
        conflict_codes: list = None,
    ) -> Dict[str, Any]:
        """Create mock console tile for testing."""
        return {
            "schema_version": "1.0.0",
            "status_light": status_light,
            "topology_stability": topology_stability,
            "bundle_stability": bundle_stability,
            "cross_system_consistency": True,
            "joint_status": "ALIGNED",
            "conflict_codes": conflict_codes or [],
            "headline": f"Topology: {topology_stability} | Bundle: {bundle_stability}",
        }

    def _make_mock_signal(self, status: str = "OK") -> Dict[str, Any]:
        """Create mock governance signal for testing."""
        return {
            "schema_version": "1.0.0",
            "signal_type": "topology_bundle",
            "status": status,
            "governance_status": status,
            "governance_alignment": "ALIGNED",
            "topology_status": "OK",
            "bundle_status": "OK",
            "conflict": False,
            "reasons": [],
            "safe_for_policy_update": True,
            "safe_for_promotion": True,
        }

    def test_01_first_light_signal_has_required_fields(self) -> None:
        """First Light signal contains all required fields."""
        from backend.health.topology_bundle_adapter import (
            extract_topology_bundle_signal_for_first_light,
        )

        tile = self._make_mock_tile()
        signal = extract_topology_bundle_signal_for_first_light(tile)

        assert "status" in signal
        assert "topology_mode" in signal
        assert "bundle_status" in signal
        assert "conflict_count" in signal

    def test_02_first_light_signal_uses_governance_signal_status(self) -> None:
        """First Light signal uses status from governance signal if provided."""
        from backend.health.topology_bundle_adapter import (
            extract_topology_bundle_signal_for_first_light,
        )

        tile = self._make_mock_tile(status_light="YELLOW")
        gov_signal = self._make_mock_signal(status="WARN")

        signal = extract_topology_bundle_signal_for_first_light(tile, governance_signal=gov_signal)

        assert signal["status"] == "WARN"

    def test_03_first_light_signal_derives_status_from_status_light(self) -> None:
        """First Light signal derives status from status_light if no signal."""
        from backend.health.topology_bundle_adapter import (
            extract_topology_bundle_signal_for_first_light,
        )

        # GREEN -> OK
        tile = self._make_mock_tile(status_light="GREEN")
        signal = extract_topology_bundle_signal_for_first_light(tile)
        assert signal["status"] == "OK"

        # YELLOW -> WARN
        tile = self._make_mock_tile(status_light="YELLOW")
        signal = extract_topology_bundle_signal_for_first_light(tile)
        assert signal["status"] == "WARN"

        # RED -> BLOCK
        tile = self._make_mock_tile(status_light="RED")
        signal = extract_topology_bundle_signal_for_first_light(tile)
        assert signal["status"] == "BLOCK"

    def test_04_first_light_signal_maps_stability_to_mode(self) -> None:
        """First Light signal maps stability back to mode."""
        from backend.health.topology_bundle_adapter import (
            extract_topology_bundle_signal_for_first_light,
        )

        # DRIFTING -> DRIFT
        tile = self._make_mock_tile(topology_stability="DRIFTING")
        signal = extract_topology_bundle_signal_for_first_light(tile)
        assert signal["topology_mode"] == "DRIFT"

    def test_05_first_light_signal_counts_conflicts(self) -> None:
        """First Light signal counts conflict codes."""
        from backend.health.topology_bundle_adapter import (
            extract_topology_bundle_signal_for_first_light,
        )

        tile = self._make_mock_tile(
            conflict_codes=["XCOR-WARN-001", "XCOR-CRIT-001", "TOPO-WARN-001"]
        )
        signal = extract_topology_bundle_signal_for_first_light(tile)

        assert signal["conflict_count"] == 3

    def test_06_first_light_signal_serializes_to_json(self) -> None:
        """First Light signal serializes to JSON."""
        from backend.health.topology_bundle_adapter import (
            extract_topology_bundle_signal_for_first_light,
        )

        tile = self._make_mock_tile(
            status_light="YELLOW",
            topology_stability="DRIFTING",
            bundle_stability="ATTENTION",
            conflict_codes=["XCOR-WARN-001"],
        )
        signal = extract_topology_bundle_signal_for_first_light(tile)

        # Should not raise
        json_str = json.dumps(signal)
        assert len(json_str) > 0

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed == signal
