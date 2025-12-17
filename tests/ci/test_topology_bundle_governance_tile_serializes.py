"""Integration tests for topology bundle governance tile.

STATUS: PHASE X — TOPOLOGY/BUNDLE GOVERNANCE TILE CI TESTS

Tests:
1. Console tile correctness - tile fields are correctly extracted
2. Status mapping correctness - topology modes and bundle statuses map correctly
3. Determinism - same inputs produce same outputs
4. Evidence-safe JSON - tile serializes without error
5. SHADOW mode validation - shadow mode contract maintained

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No tests depend on governance enforcement
- Tests validate read-only, side-effect-free behavior
"""

import copy
import json
from typing import Any, Dict

import pytest


class TestTopologyBundleConsoleTile:
    """Test topology bundle console tile correctness."""

    def _make_mock_joint_view(
        self,
        topology_mode: str = "STABLE",
        bundle_status: str = "VALID",
        alignment_status: str = "ALIGNED",
    ) -> Dict[str, Any]:
        """Create mock joint view for testing."""
        return {
            "schema_version": "1.0.0",
            "cycle": 100,
            "timestamp": "2025-12-10T00:00:00Z",
            "mode": "SHADOW",
            "topology_snapshot": {
                "topology_mode": topology_mode,
                "betti_0": 1,
                "betti_1": 0 if topology_mode == "STABLE" else 2,
                "persistence_stability": 0.03 if topology_mode == "STABLE" else 0.12,
                "in_omega": True,
                "topology_hash": "sha256:abc123...",
            },
            "bundle_snapshot": {
                "bundle_id": "sha256:def456...",
                "bundle_status": bundle_status,
                "bundle_timestamp": "2025-12-10T00:00:00Z",
                "manifest_valid": bundle_status == "VALID",
                "source_file_count": 15,
                "trace_file_count": 3,
            },
            "alignment_status": {
                "topology_bundle_aligned": alignment_status == "ALIGNED",
                "overall_status": alignment_status,
                "topology_status": "OK" if topology_mode == "STABLE" else "WARN",
                "bundle_status_contribution": "OK" if bundle_status == "VALID" else "WARN",
                "tension_reasons": [] if alignment_status == "ALIGNED" else ["[TOPO] Test reason"],
            },
            "governance_signals": {
                "topology_codes": ["TOPO-OK-001"] if topology_mode == "STABLE" else ["TOPO-WARN-001"],
                "bundle_codes": ["BNDL-OK-001"] if bundle_status == "VALID" else ["BNDL-WARN-001"],
                "correlation_codes": ["XCOR-OK-001"] if alignment_status == "ALIGNED" else ["XCOR-WARN-001"],
                "highest_severity": "OK" if alignment_status == "ALIGNED" else "WARN",
                "hypothetical_action": "CONTINUE" if alignment_status == "ALIGNED" else "CAUTION",
            },
        }

    def _make_mock_consistency_result(
        self,
        consistent: bool = True,
        status: str = "OK",
    ) -> Dict[str, Any]:
        """Create mock consistency result for testing."""
        return {
            "consistent": consistent,
            "status": status,
            "verified_at": "2025-12-10T00:00:00Z",
        }

    def _make_mock_director_panel(
        self,
        topology_status: str = "HEALTHY",
        bundle_status: str = "HEALTHY",
        alignment: str = "ALIGNED",
    ) -> Dict[str, Any]:
        """Create mock director panel for testing."""
        return {
            "schema_version": "1.0.0",
            "panel_type": "topology_bundle_director",
            "timestamp": "2025-12-10T00:00:00Z",
            "mode": "SHADOW",
            "topology_tile": {
                "mode": "STABLE",
                "health_status": topology_status,
                "indicator_color": "green",
            },
            "bundle_tile": {
                "chain_status": "VALID",
                "health_status": bundle_status,
                "indicator_color": "green",
            },
            "correlation_tile": {
                "alignment": alignment,
                "health_status": "HEALTHY",
                "indicator_color": "green",
                "active_signals": [],
            },
            "overall_health": {
                "status": "NOMINAL",
                "indicator_color": "green",
                "summary": "All systems nominal",
            },
        }

    def test_01_console_tile_has_required_fields(self) -> None:
        """Console tile contains all required fields."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view()
        consistency_result = self._make_mock_consistency_result()

        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # Verify required fields
        assert "schema_version" in tile
        assert "status_light" in tile
        assert "topology_stability" in tile
        assert "bundle_stability" in tile
        assert "cross_system_consistency" in tile
        assert "joint_status" in tile
        assert "conflict_codes" in tile
        assert "headline" in tile

    def test_02_console_tile_extracts_values_correctly(self) -> None:
        """Console tile extracts correct values from inputs."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view(
            topology_mode="DRIFT",
            bundle_status="WARN",
            alignment_status="TENSION",
        )
        consistency_result = self._make_mock_consistency_result(consistent=False)

        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # Verify extracted values
        assert tile["topology_stability"] == "DRIFTING"
        assert tile["bundle_stability"] == "ATTENTION"
        assert tile["cross_system_consistency"] is False
        assert tile["joint_status"] == "TENSION"


class TestTopologyBundleStatusMapping:
    """Test status mapping correctness."""

    def _make_mock_joint_view(
        self,
        topology_mode: str = "STABLE",
        bundle_status: str = "VALID",
        alignment_status: str = "ALIGNED",
    ) -> Dict[str, Any]:
        """Create mock joint view for testing."""
        return {
            "topology_snapshot": {"topology_mode": topology_mode},
            "bundle_snapshot": {"bundle_status": bundle_status},
            "alignment_status": {"overall_status": alignment_status},
        }

    def _make_mock_consistency_result(self) -> Dict[str, Any]:
        """Create mock consistency result for testing."""
        return {"consistent": True, "status": "OK"}

    def test_01_stable_valid_aligned_maps_to_green(self) -> None:
        """STABLE + VALID + ALIGNED -> GREEN."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view(
            topology_mode="STABLE",
            bundle_status="VALID",
            alignment_status="ALIGNED",
        )
        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=self._make_mock_consistency_result(),
        )

        assert tile["status_light"] == "GREEN"

    def test_02_drift_valid_aligned_maps_to_yellow(self) -> None:
        """DRIFT + VALID + ALIGNED -> YELLOW."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view(
            topology_mode="DRIFT",
            bundle_status="VALID",
            alignment_status="ALIGNED",
        )
        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=self._make_mock_consistency_result(),
        )

        assert tile["status_light"] == "YELLOW"

    def test_03_critical_maps_to_red(self) -> None:
        """CRITICAL topology -> RED."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view(
            topology_mode="CRITICAL",
            bundle_status="VALID",
            alignment_status="ALIGNED",
        )
        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=self._make_mock_consistency_result(),
        )

        assert tile["status_light"] == "RED"

    def test_04_broken_bundle_maps_to_red(self) -> None:
        """BROKEN bundle -> RED."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view(
            topology_mode="STABLE",
            bundle_status="BROKEN",
            alignment_status="ALIGNED",
        )
        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=self._make_mock_consistency_result(),
        )

        assert tile["status_light"] == "RED"

    def test_05_divergent_alignment_maps_to_red(self) -> None:
        """DIVERGENT alignment -> RED."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view(
            topology_mode="STABLE",
            bundle_status="VALID",
            alignment_status="DIVERGENT",
        )
        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=self._make_mock_consistency_result(),
        )

        assert tile["status_light"] == "RED"


class TestTopologyBundleDeterminism:
    """Test determinism - same inputs produce same outputs."""

    def _make_mock_joint_view(self) -> Dict[str, Any]:
        """Create mock joint view for testing."""
        return {
            "topology_snapshot": {"topology_mode": "DRIFT", "betti_0": 1, "betti_1": 2},
            "bundle_snapshot": {"bundle_status": "WARN"},
            "alignment_status": {"overall_status": "TENSION"},
            "governance_signals": {
                "correlation_codes": ["XCOR-WARN-001"],
            },
        }

    def _make_mock_consistency_result(self) -> Dict[str, Any]:
        """Create mock consistency result for testing."""
        return {"consistent": True, "status": "OK"}

    def test_01_console_tile_is_deterministic(self) -> None:
        """Same inputs produce identical console tiles."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view()
        consistency_result = self._make_mock_consistency_result()

        tile1 = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )
        tile2 = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        assert tile1 == tile2

    def test_02_governance_signal_is_deterministic(self) -> None:
        """Same inputs produce identical governance signals."""
        from backend.health.topology_bundle_adapter import (
            topology_bundle_to_governance_signal,
        )

        joint_view = self._make_mock_joint_view()
        consistency_result = self._make_mock_consistency_result()

        signal1 = topology_bundle_to_governance_signal(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )
        signal2 = topology_bundle_to_governance_signal(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        assert signal1 == signal2

    def test_03_inputs_not_mutated(self) -> None:
        """Input dicts are not modified by tile builder."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view()
        consistency_result = self._make_mock_consistency_result()

        # Deep copy inputs
        joint_view_copy = copy.deepcopy(joint_view)
        consistency_result_copy = copy.deepcopy(consistency_result)

        # Build tile
        build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # Verify inputs unchanged
        assert joint_view == joint_view_copy
        assert consistency_result == consistency_result_copy


class TestTopologyBundleEvidenceSafeJSON:
    """Test evidence-safe JSON serialization."""

    def _make_mock_joint_view(self) -> Dict[str, Any]:
        """Create mock joint view for testing."""
        return {
            "topology_snapshot": {"topology_mode": "STABLE"},
            "bundle_snapshot": {"bundle_status": "VALID"},
            "alignment_status": {"overall_status": "ALIGNED"},
        }

    def _make_mock_consistency_result(self) -> Dict[str, Any]:
        """Create mock consistency result for testing."""
        return {"consistent": True, "status": "OK"}

    def test_01_console_tile_serializes_to_json(self) -> None:
        """Console tile serializes to JSON without error."""
        from backend.health.topology_bundle_adapter import (
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view()
        consistency_result = self._make_mock_consistency_result()

        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # Should not raise
        json_str = json.dumps(tile)
        assert len(json_str) > 0

        # Round-trip should produce same structure
        parsed = json.loads(json_str)
        assert parsed == tile

    def test_02_governance_signal_serializes_to_json(self) -> None:
        """Governance signal serializes to JSON without error."""
        from backend.health.topology_bundle_adapter import (
            topology_bundle_to_governance_signal,
        )

        joint_view = self._make_mock_joint_view()
        consistency_result = self._make_mock_consistency_result()

        signal = topology_bundle_to_governance_signal(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # Should not raise
        json_str = json.dumps(signal)
        assert len(json_str) > 0

        # Round-trip should produce same structure
        parsed = json.loads(json_str)
        assert parsed == signal

    def test_03_evidence_attachment_serializes(self) -> None:
        """Evidence with attached tile serializes to JSON."""
        from backend.health.topology_bundle_adapter import (
            attach_topology_bundle_to_evidence,
            build_topology_bundle_console_tile,
        )

        joint_view = self._make_mock_joint_view()
        consistency_result = self._make_mock_consistency_result()

        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        evidence = {"timestamp": "2025-12-10T00:00:00Z", "data": {"key": "value"}}
        enriched = attach_topology_bundle_to_evidence(evidence, tile)

        # Should not raise
        json_str = json.dumps(enriched)
        assert len(json_str) > 0

        # Should contain governance section
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "topology_bundle" in parsed["governance"]


class TestTopologyBundleShadowModeValidation:
    """Test SHADOW mode contract validation."""

    def _make_mock_joint_view(self) -> Dict[str, Any]:
        """Create mock joint view for testing."""
        return {
            "topology_snapshot": {"topology_mode": "CRITICAL"},
            "bundle_snapshot": {"bundle_status": "BROKEN"},
            "alignment_status": {"overall_status": "DIVERGENT"},
        }

    def _make_mock_consistency_result(self) -> Dict[str, Any]:
        """Create mock consistency result for testing."""
        return {"consistent": False, "status": "CRITICAL"}

    def test_01_governance_signal_always_safe_for_policy_update(self) -> None:
        """SHADOW MODE: safe_for_policy_update is always True."""
        from backend.health.topology_bundle_adapter import (
            topology_bundle_to_governance_signal,
        )

        # Even with CRITICAL status
        joint_view = self._make_mock_joint_view()
        consistency_result = self._make_mock_consistency_result()

        signal = topology_bundle_to_governance_signal(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # SHADOW MODE: always permissive
        assert signal["safe_for_policy_update"] is True

    def test_02_governance_signal_always_safe_for_promotion(self) -> None:
        """SHADOW MODE: safe_for_promotion is always True."""
        from backend.health.topology_bundle_adapter import (
            topology_bundle_to_governance_signal,
        )

        # Even with CRITICAL status
        joint_view = self._make_mock_joint_view()
        consistency_result = self._make_mock_consistency_result()

        signal = topology_bundle_to_governance_signal(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # SHADOW MODE: always permissive
        assert signal["safe_for_promotion"] is True

    def test_03_tile_has_schema_version(self) -> None:
        """All tiles include schema version for forward compatibility."""
        from backend.health.topology_bundle_adapter import (
            TOPOLOGY_BUNDLE_TILE_SCHEMA_VERSION,
            build_topology_bundle_console_tile,
            topology_bundle_to_governance_signal,
        )

        joint_view = {
            "topology_snapshot": {"topology_mode": "STABLE"},
            "bundle_snapshot": {"bundle_status": "VALID"},
            "alignment_status": {"overall_status": "ALIGNED"},
        }
        consistency_result = {"consistent": True, "status": "OK"}

        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )
        signal = topology_bundle_to_governance_signal(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        assert tile["schema_version"] == TOPOLOGY_BUNDLE_TILE_SCHEMA_VERSION
        assert signal["schema_version"] == TOPOLOGY_BUNDLE_TILE_SCHEMA_VERSION

    def test_04_reasons_have_correct_prefixes(self) -> None:
        """Governance signal reasons are prefixed with [Topology] or [Bundle]."""
        from backend.health.topology_bundle_adapter import (
            topology_bundle_to_governance_signal,
        )

        joint_view = {
            "topology_snapshot": {"topology_mode": "DRIFT"},
            "bundle_snapshot": {"bundle_status": "WARN"},
            "alignment_status": {"overall_status": "TENSION"},
        }
        consistency_result = {"consistent": True, "status": "OK"}

        signal = topology_bundle_to_governance_signal(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # All reasons must be prefixed
        for reason in signal["reasons"]:
            assert reason.startswith("[Topology]") or reason.startswith("[Bundle]"), (
                f"Reason not prefixed: {reason}"
            )

    def test_05_evidence_attachment_is_non_mutating(self) -> None:
        """Evidence attachment does not modify original evidence dict."""
        from backend.health.topology_bundle_adapter import (
            attach_topology_bundle_to_evidence,
            build_topology_bundle_console_tile,
        )

        joint_view = {
            "topology_snapshot": {"topology_mode": "STABLE"},
            "bundle_snapshot": {"bundle_status": "VALID"},
            "alignment_status": {"overall_status": "ALIGNED"},
        }
        consistency_result = {"consistent": True, "status": "OK"}

        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        evidence = {"timestamp": "2025-12-10T00:00:00Z", "data": {"key": "value"}}
        evidence_copy = copy.deepcopy(evidence)

        # Attach should return new dict
        enriched = attach_topology_bundle_to_evidence(evidence, tile)

        # Original should be unchanged
        assert evidence == evidence_copy

        # Enriched should have governance section
        assert "governance" in enriched
        assert "governance" not in evidence


class TestGlobalHealthSurfaceTopologyBundleIntegration:
    """Test topology bundle tile attaches to global health surface."""

    def test_01_attach_function_works_correctly(self) -> None:
        """attach_topology_bundle_tile attaches tile to payload."""
        from backend.health.global_surface import attach_topology_bundle_tile

        joint_view = {
            "topology_snapshot": {"topology_mode": "STABLE"},
            "bundle_snapshot": {"bundle_status": "VALID"},
            "alignment_status": {"overall_status": "ALIGNED"},
        }
        consistency_result = {"consistent": True, "status": "OK"}

        payload = {"existing_key": "existing_value"}
        updated = attach_topology_bundle_tile(
            payload=payload,
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # Should have topology_bundle tile
        assert "topology_bundle" in updated

        # Should preserve existing keys
        assert updated["existing_key"] == "existing_value"

        # Should not modify original payload
        assert "topology_bundle" not in payload

    def test_02_attach_function_handles_missing_optional_director_panel(self) -> None:
        """attach_topology_bundle_tile works without director_panel."""
        from backend.health.global_surface import attach_topology_bundle_tile

        joint_view = {
            "topology_snapshot": {"topology_mode": "STABLE"},
            "bundle_snapshot": {"bundle_status": "VALID"},
            "alignment_status": {"overall_status": "ALIGNED"},
        }
        consistency_result = {"consistent": True, "status": "OK"}

        payload = {}
        updated = attach_topology_bundle_tile(
            payload=payload,
            joint_view=joint_view,
            consistency_result=consistency_result,
            director_panel=None,  # Explicitly None
        )

        # Should still attach tile
        assert "topology_bundle" in updated
