"""
Phase X: CI Smoke Test for Consensus Governance Tile Serialization

This test verifies that the consensus governance tile can be produced and serialized
without error. It does NOT test governance logic, polygraph semantics, or
conflict detection behavior.

SHADOW MODE CONTRACT:
- This test only verifies serialization and structural stability
- No governance decisions are tested or modified
- No polygraph analysis or real consensus checks are run
- The test is purely for observability validation

Test requirements (per Phase X spec):
1. Create synthetic polygraph_result, predictive_result, and director_panel
2. Call build_consensus_governance_tile()
3. Assert: isinstance(tile, dict)
4. Assert: json.dumps(tile) does not raise
5. Validate JSON structure, determinism, and neutral language
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest


class TestConsensusGovernanceTileSerializes:
    """
    CI smoke tests for consensus governance tile serialization.

    SHADOW MODE: These tests verify serialization only.
    No governance logic is tested.
    """

    def test_consensus_governance_tile_serializes_without_error(self) -> None:
        """
        Verify consensus governance tile can be produced and serialized.

        This is the primary CI gate test per Phase X spec.
        """
        from backend.health.consensus_polygraph_adapter import (
            CONSENSUS_GOVERNANCE_TILE_SCHEMA_VERSION,
            build_consensus_governance_tile,
        )

        # 1. Create synthetic polygraph result
        polygraph_result: Dict[str, Any] = {
            "system_conflicts": [
                {
                    "slice_id": "slice_1",
                    "component": None,
                    "conflicting_systems": ["semantic", "metric"],
                    "statuses": {"semantic": "BLOCK", "metric": "OK"},
                    "severity": "HIGH",
                }
            ],
            "agreement_rate": 0.75,
            "consensus_band": "MEDIUM",
            "neutral_notes": ["Analyzed 4 slices across 2 systems"],
            "total_slices": 4,
            "agreeing_slices": 3,
        }

        # 2. Create synthetic predictive result
        predictive_result: Dict[str, Any] = {
            "predictive_conflicts": [
                {
                    "slice_id": "slice_1",
                    "risk": "HIGH",
                    "reason": "Multiple systems show BLOCK status",
                    "systems_involved": ["semantic", "metric"],
                }
            ],
            "total_predictions": 1,
            "high_risk_predictions": 1,
        }

        # 3. Create synthetic director panel
        director_panel: Dict[str, Any] = {
            "status_light": "YELLOW",
            "consensus_band": "MEDIUM",
            "agreement_rate": 0.75,
            "headline": "Moderate consensus with some divergence. 1 conflicts detected.",
            "total_slices": 4,
            "conflicts": 1,
            "predictive_risks": 1,
        }

        # 4. Call build_consensus_governance_tile()
        tile = build_consensus_governance_tile(
            polygraph_result=polygraph_result,
            predictive_result=predictive_result,
            director_panel=director_panel,
        )

        # 5. Assert: isinstance(tile, dict)
        assert tile is not None, "Tile should not be None"
        assert isinstance(tile, dict), f"Tile should be dict, got {type(tile)}"

        # 6. Assert: json.dumps(tile) does not raise
        json_str = json.dumps(tile)
        assert json_str is not None
        assert len(json_str) > 0

        # Verify round-trip
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_consensus_governance_tile_has_required_fields(self) -> None:
        """Verify tile contains required fields per schema."""
        from backend.health.consensus_polygraph_adapter import (
            CONSENSUS_GOVERNANCE_TILE_SCHEMA_VERSION,
            build_consensus_governance_tile,
        )

        polygraph_result: Dict[str, Any] = {
            "system_conflicts": [],
            "agreement_rate": 1.0,
            "consensus_band": "HIGH",
            "neutral_notes": [],
            "total_slices": 10,
            "agreeing_slices": 10,
        }

        director_panel: Dict[str, Any] = {
            "status_light": "GREEN",
            "headline": "High consensus across systems",
        }

        tile = build_consensus_governance_tile(
            polygraph_result=polygraph_result,
            predictive_result=None,
            director_panel=director_panel,
        )

        # Required fields per Phase X spec
        required_fields = [
            "schema_version",
            "status_light",
            "consensus_band",
            "agreement_rate",
            "conflict_count",
            "predictive_risk_band",
            "predictive_conflict_count",
            "headline",
        ]

        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

        # Verify schema version
        assert tile["schema_version"] == CONSENSUS_GOVERNANCE_TILE_SCHEMA_VERSION

    def test_consensus_governance_tile_deterministic(self) -> None:
        """Verify tile output is deterministic (same inputs → same output)."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_governance_tile,
        )

        polygraph_result: Dict[str, Any] = {
            "system_conflicts": [
                {
                    "slice_id": "slice_1",
                    "component": None,
                    "conflicting_systems": ["semantic", "metric"],
                    "statuses": {"semantic": "BLOCK", "metric": "OK"},
                    "severity": "HIGH",
                }
            ],
            "agreement_rate": 0.8,
            "consensus_band": "HIGH",
            "neutral_notes": [],
            "total_slices": 5,
            "agreeing_slices": 4,
        }

        predictive_result: Dict[str, Any] = {
            "predictive_conflicts": [],
            "total_predictions": 0,
            "high_risk_predictions": 0,
        }

        director_panel: Dict[str, Any] = {
            "status_light": "GREEN",
            "headline": "High consensus across systems",
        }

        # Call twice with same inputs
        tile1 = build_consensus_governance_tile(
            polygraph_result=polygraph_result,
            predictive_result=predictive_result,
            director_panel=director_panel,
        )

        tile2 = build_consensus_governance_tile(
            polygraph_result=polygraph_result,
            predictive_result=predictive_result,
            director_panel=director_panel,
        )

        # Should be identical
        assert tile1 == tile2, "Tile output should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)
        assert json1 == json2, "JSON serialization should be deterministic"

    def test_consensus_governance_tile_neutral_language(self) -> None:
        """Verify tile uses neutral language (no prescriptive terms)."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_governance_tile,
        )

        polygraph_result: Dict[str, Any] = {
            "system_conflicts": [],
            "agreement_rate": 0.5,
            "consensus_band": "MEDIUM",
            "neutral_notes": [],
            "total_slices": 10,
            "agreeing_slices": 5,
        }

        director_panel: Dict[str, Any] = {
            "status_light": "YELLOW",
            "headline": "Moderate consensus with some divergence",
        }

        tile = build_consensus_governance_tile(
            polygraph_result=polygraph_result,
            predictive_result=None,
            director_panel=director_panel,
        )

        # Check headline for prescriptive terms
        headline = tile.get("headline", "").lower()
        prescriptive_terms = ["must", "required", "should", "need to", "have to"]
        for term in prescriptive_terms:
            assert term not in headline, f"Headline should not contain prescriptive term: {term}"

    def test_consensus_governance_tile_without_predictive_result(self) -> None:
        """Verify tile works correctly when predictive_result is None."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_governance_tile,
        )

        polygraph_result: Dict[str, Any] = {
            "system_conflicts": [],
            "agreement_rate": 0.9,
            "consensus_band": "HIGH",
            "neutral_notes": [],
            "total_slices": 10,
            "agreeing_slices": 9,
        }

        director_panel: Dict[str, Any] = {
            "status_light": "GREEN",
            "headline": "High consensus across systems",
        }

        tile = build_consensus_governance_tile(
            polygraph_result=polygraph_result,
            predictive_result=None,
            director_panel=director_panel,
        )

        assert tile["predictive_risk_band"] == "UNKNOWN"
        assert tile["predictive_conflict_count"] == 0

    def test_consensus_governance_tile_predictive_risk_bands(self) -> None:
        """Verify predictive risk band calculation."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_governance_tile,
        )

        polygraph_result: Dict[str, Any] = {
            "system_conflicts": [],
            "agreement_rate": 0.8,
            "consensus_band": "HIGH",
            "neutral_notes": [],
            "total_slices": 10,
            "agreeing_slices": 8,
        }

        director_panel: Dict[str, Any] = {
            "status_light": "GREEN",
            "headline": "High consensus",
        }

        # HIGH risk
        predictive_high: Dict[str, Any] = {
            "predictive_conflicts": [
                {"slice_id": "slice_1", "risk": "HIGH", "reason": "test", "systems_involved": []}
            ],
            "total_predictions": 1,
            "high_risk_predictions": 1,
        }

        tile_high = build_consensus_governance_tile(
            polygraph_result=polygraph_result,
            predictive_result=predictive_high,
            director_panel=director_panel,
        )
        assert tile_high["predictive_risk_band"] == "HIGH"

        # MEDIUM risk
        predictive_medium: Dict[str, Any] = {
            "predictive_conflicts": [
                {"slice_id": "slice_1", "risk": "MEDIUM", "reason": "test", "systems_involved": []}
            ],
            "total_predictions": 1,
            "high_risk_predictions": 0,
        }

        tile_medium = build_consensus_governance_tile(
            polygraph_result=polygraph_result,
            predictive_result=predictive_medium,
            director_panel=director_panel,
        )
        assert tile_medium["predictive_risk_band"] == "MEDIUM"

        # LOW risk
        predictive_low: Dict[str, Any] = {
            "predictive_conflicts": [],
            "total_predictions": 0,
            "high_risk_predictions": 0,
        }

        tile_low = build_consensus_governance_tile(
            polygraph_result=polygraph_result,
            predictive_result=predictive_low,
            director_panel=director_panel,
        )
        assert tile_low["predictive_risk_band"] == "LOW"

    def test_extract_consensus_signal_for_evidence(self) -> None:
        """Verify extract_consensus_signal_for_evidence helper."""
        from backend.health.consensus_polygraph_adapter import (
            extract_consensus_signal_for_evidence,
        )

        polygraph_result: Dict[str, Any] = {
            "system_conflicts": [
                {
                    "slice_id": "slice_1",
                    "component": None,
                    "conflicting_systems": ["semantic", "metric"],
                    "statuses": {"semantic": "BLOCK", "metric": "OK"},
                    "severity": "HIGH",
                }
            ],
            "agreement_rate": 0.75,
            "consensus_band": "MEDIUM",
            "neutral_notes": [],
            "total_slices": 4,
            "agreeing_slices": 3,
        }

        signal = extract_consensus_signal_for_evidence(polygraph_result)

        assert isinstance(signal, dict)
        assert "consensus_band" in signal
        assert "agreement_rate" in signal
        assert "conflict_count" in signal

        assert signal["consensus_band"] == "MEDIUM"
        assert signal["agreement_rate"] == 0.75
        assert signal["conflict_count"] == 1

        # Verify serializable
        json_str = json.dumps(signal)
        assert len(json_str) > 0

    def test_extract_consensus_signal_deterministic(self) -> None:
        """Verify extract_consensus_signal_for_evidence is deterministic."""
        from backend.health.consensus_polygraph_adapter import (
            extract_consensus_signal_for_evidence,
        )

        polygraph_result: Dict[str, Any] = {
            "system_conflicts": [],
            "agreement_rate": 0.9,
            "consensus_band": "HIGH",
            "neutral_notes": [],
            "total_slices": 10,
            "agreeing_slices": 9,
        }

        signal1 = extract_consensus_signal_for_evidence(polygraph_result)
        signal2 = extract_consensus_signal_for_evidence(polygraph_result)

        assert signal1 == signal2, "Signal extraction should be deterministic"


class TestGlobalHealthSurfaceConsensusIntegration:
    """
    Tests for consensus governance tile integration with GlobalHealthSurface.

    SHADOW MODE: These tests verify the tile attachment mechanism only.
    """

    def test_build_global_health_surface_with_consensus_governance(self) -> None:
        """Verify consensus governance tile attached when data provided."""
        from backend.health.global_surface import build_global_health_surface

        polygraph_result: Dict[str, Any] = {
            "system_conflicts": [],
            "agreement_rate": 0.9,
            "consensus_band": "HIGH",
            "neutral_notes": [],
            "total_slices": 10,
            "agreeing_slices": 9,
        }

        director_panel: Dict[str, Any] = {
            "status_light": "GREEN",
            "headline": "High consensus across systems",
        }

        payload = build_global_health_surface(
            consensus_polygraph_result=polygraph_result,
            consensus_director_panel=director_panel,
        )

        assert isinstance(payload, dict)
        assert "consensus_governance" in payload, "Consensus governance tile should be present"

        consensus_tile = payload["consensus_governance"]
        assert isinstance(consensus_tile, dict)
        assert "headline" in consensus_tile

        # Verify serializable
        json_str = json.dumps(payload)
        assert len(json_str) > 0

    def test_build_global_health_surface_without_consensus_governance(self) -> None:
        """Verify build works without consensus governance data."""
        from backend.health.global_surface import build_global_health_surface

        payload = build_global_health_surface()

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "consensus_governance" not in payload  # Should not be present

    def test_consensus_governance_tile_does_not_affect_dynamics(self) -> None:
        """Verify consensus governance tile presence doesn't change dynamics tile."""
        from backend.health.global_surface import build_global_health_surface

        polygraph_result: Dict[str, Any] = {
            "system_conflicts": [],
            "agreement_rate": 0.9,
            "consensus_band": "HIGH",
            "neutral_notes": [],
            "total_slices": 10,
            "agreeing_slices": 9,
        }

        director_panel: Dict[str, Any] = {
            "status_light": "GREEN",
            "headline": "High consensus",
        }

        # Build without consensus governance
        payload_without = build_global_health_surface()

        # Build with consensus governance
        payload_with = build_global_health_surface(
            consensus_polygraph_result=polygraph_result,
            consensus_director_panel=director_panel,
        )

        # Dynamics should be identical
        assert payload_without["dynamics"] == payload_with["dynamics"]


class TestFirstLightConflictLedger:
    """
    Tests for First Light conflict ledger annex.

    SHADOW MODE: These tests verify the conflict ledger structure and determinism.

    NOTE: The conflict ledger is designed as a compact disagreement summary for First Light
    evidence packs, providing a cross-layer disagreement index. It is NOT intended as a direct
    gating rule for governance decisions. The ledger serves observational and diagnostic
    purposes, enabling reviewers and fusion layers to assess consensus health without
    enforcing hard blocks based solely on conflict counts.
    """

    def test_build_first_light_conflict_ledger_structure(self) -> None:
        """Verify conflict ledger has required structure."""
        from backend.health.consensus_polygraph_adapter import (
            build_first_light_conflict_ledger,
        )

        tile: Dict[str, Any] = {
            "consensus_band": "HIGH",
            "agreement_rate": 0.9,
            "conflict_count": 2,
            "predictive_risk_band": "LOW",
            "status_light": "GREEN",
        }

        ledger = build_first_light_conflict_ledger(tile)

        required_fields = [
            "schema_version",
            "consensus_band",
            "agreement_rate",
            "conflict_count",
            "predictive_risk_band",
        ]

        for field in required_fields:
            assert field in ledger, f"Missing required field: {field}"

        assert ledger["schema_version"] == "1.0.0"
        assert ledger["consensus_band"] == "HIGH"
        assert ledger["agreement_rate"] == 0.9
        assert ledger["conflict_count"] == 2
        assert ledger["predictive_risk_band"] == "LOW"

    def test_build_first_light_conflict_ledger_deterministic(self) -> None:
        """Verify conflict ledger output is deterministic."""
        from backend.health.consensus_polygraph_adapter import (
            build_first_light_conflict_ledger,
        )

        tile: Dict[str, Any] = {
            "consensus_band": "MEDIUM",
            "agreement_rate": 0.75,
            "conflict_count": 3,
            "predictive_risk_band": "MEDIUM",
        }

        ledger1 = build_first_light_conflict_ledger(tile)
        ledger2 = build_first_light_conflict_ledger(tile)

        assert ledger1 == ledger2, "Ledger output should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(ledger1, sort_keys=True)
        json2 = json.dumps(ledger2, sort_keys=True)
        assert json1 == json2, "JSON serialization should be deterministic"

    def test_build_first_light_conflict_ledger_json_safe(self) -> None:
        """Verify conflict ledger is JSON-safe."""
        from backend.health.consensus_polygraph_adapter import (
            build_first_light_conflict_ledger,
        )

        tile: Dict[str, Any] = {
            "consensus_band": "LOW",
            "agreement_rate": 0.4,
            "conflict_count": 10,
            "predictive_risk_band": "HIGH",
        }

        ledger = build_first_light_conflict_ledger(tile)

        # Should serialize without error
        json_str = json.dumps(ledger)
        assert len(json_str) > 0

        # Should round-trip
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["schema_version"] == "1.0.0"

    def test_attach_consensus_governance_includes_conflict_ledger(self) -> None:
        """Verify evidence attachment includes conflict ledger."""
        from backend.health.consensus_polygraph_adapter import (
            attach_consensus_governance_to_evidence,
        )

        tile: Dict[str, Any] = {
            "consensus_band": "HIGH",
            "agreement_rate": 0.9,
            "conflict_count": 1,
            "predictive_risk_band": "LOW",
            "status_light": "GREEN",
            "predictive_conflict_count": 0,
            "headline": "High consensus",
        }

        signal: Dict[str, Any] = {
            "consensus_band": "HIGH",
            "agreement_rate": 0.9,
            "conflict_count": 1,
        }

        evidence: Dict[str, Any] = {}
        attach_consensus_governance_to_evidence(evidence, tile, signal)

        assert "governance" in evidence
        assert "consensus" in evidence["governance"]
        assert "first_light_conflict_ledger" in evidence["governance"]["consensus"]

        ledger = evidence["governance"]["consensus"]["first_light_conflict_ledger"]
        assert ledger["schema_version"] == "1.0.0"
        assert ledger["consensus_band"] == "HIGH"
        assert ledger["agreement_rate"] == 0.9
        assert ledger["conflict_count"] == 1

    def test_attach_consensus_governance_non_mutating(self) -> None:
        """Verify evidence attachment does not mutate input tile/signal."""
        from backend.health.consensus_polygraph_adapter import (
            attach_consensus_governance_to_evidence,
        )

        tile: Dict[str, Any] = {
            "consensus_band": "MEDIUM",
            "agreement_rate": 0.7,
            "conflict_count": 5,
            "predictive_risk_band": "MEDIUM",
            "status_light": "YELLOW",
            "predictive_conflict_count": 2,
            "headline": "Moderate consensus",
        }

        signal: Dict[str, Any] = {
            "consensus_band": "MEDIUM",
            "agreement_rate": 0.7,
            "conflict_count": 5,
        }

        tile_copy = tile.copy()
        signal_copy = signal.copy()

        evidence: Dict[str, Any] = {}
        attach_consensus_governance_to_evidence(evidence, tile, signal)

        # Original tile and signal should be unchanged
        assert tile == tile_copy, "Tile should not be mutated"
        assert signal == signal_copy, "Signal should not be mutated"

    def test_council_summary_includes_conflict_count_and_risk_band(self) -> None:
        """Verify council summary includes conflict_count and predictive_risk_band."""
        from backend.health.consensus_polygraph_adapter import (
            summarize_consensus_for_uplift_council,
        )

        tile: Dict[str, Any] = {
            "conflict_count": 6,
            "predictive_risk_band": "HIGH",
            "status_light": "RED",
        }

        council = summarize_consensus_for_uplift_council(tile)

        assert "conflict_count" in council
        assert "predictive_risk_band" in council
        assert council["conflict_count"] == 6
        assert council["predictive_risk_band"] == "HIGH"


class TestConsensusConflictRegister:
    """
    Tests for consensus conflict register (CAL-EXP aggregation).

    SHADOW MODE: These tests verify the register structure, determinism, and evidence integration.
    """

    def test_emit_cal_exp_conflict_ledger_creates_file(self) -> None:
        """Verify emit_cal_exp_conflict_ledger creates JSON file."""
        import tempfile
        from pathlib import Path
        from backend.health.consensus_polygraph_adapter import (
            emit_cal_exp_conflict_ledger,
        )

        tile: Dict[str, Any] = {
            "consensus_band": "MEDIUM",
            "agreement_rate": 0.75,
            "conflict_count": 3,
            "predictive_risk_band": "MEDIUM",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = emit_cal_exp_conflict_ledger("CAL-EXP-1", tile, output_dir=tmpdir)

            assert file_path.exists()
            assert file_path.name == "conflict_ledger_CAL-EXP-1.json"

            # Verify file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            assert content["cal_id"] == "CAL-EXP-1"
            assert content["consensus_band"] == "MEDIUM"
            assert content["agreement_rate"] == 0.75
            assert content["conflict_count"] == 3

    def test_build_consensus_conflict_register_structure(self) -> None:
        """Verify consensus conflict register has required structure."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
        )

        ledgers: List[Dict[str, Any]] = [
            {
                "schema_version": "1.0.0",
                "cal_id": "CAL-EXP-1",
                "consensus_band": "HIGH",
                "agreement_rate": 0.9,
                "conflict_count": 1,
                "predictive_risk_band": "LOW",
            },
            {
                "schema_version": "1.0.0",
                "cal_id": "CAL-EXP-2",
                "consensus_band": "MEDIUM",
                "agreement_rate": 0.75,
                "conflict_count": 4,
                "predictive_risk_band": "MEDIUM",
            },
            {
                "schema_version": "1.0.0",
                "cal_id": "CAL-EXP-3",
                "consensus_band": "LOW",
                "agreement_rate": 0.5,
                "conflict_count": 8,
                "predictive_risk_band": "HIGH",
            },
        ]

        register = build_consensus_conflict_register(ledgers)

        required_fields = [
            "schema_version",
            "total_experiments",
            "conflict_count_distribution",
            "high_risk_experiments_count",
            "experiments_high_conflict",
            "consensus_band_distribution",
            "average_agreement_rate",
        ]

        for field in required_fields:
            assert field in register, f"Missing required field: {field}"

        assert register["schema_version"] == "1.0.0"
        assert register["total_experiments"] == 3

    def test_build_consensus_conflict_register_bucket_counts(self) -> None:
        """Verify conflict count distribution buckets are calculated correctly."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 1, "consensus_band": "HIGH", "agreement_rate": 0.9, "predictive_risk_band": "LOW"},
            {"cal_id": "CAL-EXP-2", "conflict_count": 2, "consensus_band": "HIGH", "agreement_rate": 0.85, "predictive_risk_band": "LOW"},
            {"cal_id": "CAL-EXP-3", "conflict_count": 3, "consensus_band": "MEDIUM", "agreement_rate": 0.75, "predictive_risk_band": "MEDIUM"},
            {"cal_id": "CAL-EXP-4", "conflict_count": 5, "consensus_band": "MEDIUM", "agreement_rate": 0.7, "predictive_risk_band": "MEDIUM"},
            {"cal_id": "CAL-EXP-5", "conflict_count": 6, "consensus_band": "LOW", "agreement_rate": 0.5, "predictive_risk_band": "HIGH"},
            {"cal_id": "CAL-EXP-6", "conflict_count": 10, "consensus_band": "LOW", "agreement_rate": 0.4, "predictive_risk_band": "HIGH"},
        ]

        register = build_consensus_conflict_register(ledgers, high_conflict_threshold=5)

        distribution = register["conflict_count_distribution"]
        assert distribution["0-2"] == 2  # CAL-EXP-1, CAL-EXP-2
        assert distribution["3-5"] == 2  # CAL-EXP-3, CAL-EXP-4
        assert distribution[">5"] == 2  # CAL-EXP-5, CAL-EXP-6

        assert register["high_risk_experiments_count"] == 2  # CAL-EXP-5, CAL-EXP-6
        assert len(register["experiments_high_conflict"]) == 2  # CAL-EXP-5, CAL-EXP-6
        assert "CAL-EXP-5" in register["experiments_high_conflict"]
        assert "CAL-EXP-6" in register["experiments_high_conflict"]

    def test_build_consensus_conflict_register_deterministic(self) -> None:
        """Verify consensus conflict register output is deterministic."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 3, "consensus_band": "MEDIUM", "agreement_rate": 0.75, "predictive_risk_band": "MEDIUM"},
            {"cal_id": "CAL-EXP-2", "conflict_count": 1, "consensus_band": "HIGH", "agreement_rate": 0.9, "predictive_risk_band": "LOW"},
        ]

        register1 = build_consensus_conflict_register(ledgers)
        register2 = build_consensus_conflict_register(ledgers)

        assert register1 == register2, "Register output should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(register1, sort_keys=True)
        json2 = json.dumps(register2, sort_keys=True)
        assert json1 == json2, "JSON serialization should be deterministic"

    def test_build_consensus_conflict_register_ordering(self) -> None:
        """Verify experiments_high_conflict is sorted deterministically."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-3", "conflict_count": 10, "consensus_band": "LOW", "agreement_rate": 0.4, "predictive_risk_band": "HIGH"},
            {"cal_id": "CAL-EXP-1", "conflict_count": 8, "consensus_band": "LOW", "agreement_rate": 0.5, "predictive_risk_band": "HIGH"},
            {"cal_id": "CAL-EXP-2", "conflict_count": 6, "consensus_band": "MEDIUM", "agreement_rate": 0.6, "predictive_risk_band": "MEDIUM"},
        ]

        register = build_consensus_conflict_register(ledgers, high_conflict_threshold=5)

        # Should be sorted
        high_conflict = register["experiments_high_conflict"]
        assert high_conflict == ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"]

    def test_build_consensus_conflict_register_empty(self) -> None:
        """Verify register handles empty ledger list."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
        )

        register = build_consensus_conflict_register([])

        assert register["total_experiments"] == 0
        assert register["conflict_count_distribution"]["0-2"] == 0
        assert register["high_risk_experiments_count"] == 0
        assert register["experiments_high_conflict"] == []
        assert register["average_agreement_rate"] == 0.0

    def test_attach_consensus_conflict_register_to_evidence(self) -> None:
        """Verify evidence attachment includes consensus conflict register."""
        from backend.health.consensus_polygraph_adapter import (
            attach_consensus_conflict_register_to_evidence,
            build_consensus_conflict_register,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 2, "consensus_band": "HIGH", "agreement_rate": 0.9, "predictive_risk_band": "LOW"},
        ]

        register = build_consensus_conflict_register(ledgers)
        evidence: Dict[str, Any] = {}

        attach_consensus_conflict_register_to_evidence(evidence, register)

        assert "governance" in evidence
        assert "consensus_conflict_register" in evidence["governance"]

        attached = evidence["governance"]["consensus_conflict_register"]
        assert attached["total_experiments"] == 1
        assert attached["schema_version"] == "1.0.0"

    def test_attach_consensus_conflict_register_non_mutating(self) -> None:
        """Verify evidence attachment does not mutate input register."""
        from backend.health.consensus_polygraph_adapter import (
            attach_consensus_conflict_register_to_evidence,
            build_consensus_conflict_register,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 3, "consensus_band": "MEDIUM", "agreement_rate": 0.75, "predictive_risk_band": "MEDIUM"},
        ]

        register = build_consensus_conflict_register(ledgers)
        register_copy = register.copy()

        evidence: Dict[str, Any] = {}
        attach_consensus_conflict_register_to_evidence(evidence, register)

        # Original register should be unchanged
        assert register == register_copy, "Register should not be mutated"

    def test_consensus_conflict_register_json_safe(self) -> None:
        """Verify consensus conflict register is JSON-safe."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 1, "consensus_band": "HIGH", "agreement_rate": 0.9, "predictive_risk_band": "LOW"},
            {"cal_id": "CAL-EXP-2", "conflict_count": 5, "consensus_band": "MEDIUM", "agreement_rate": 0.7, "predictive_risk_band": "MEDIUM"},
        ]

        register = build_consensus_conflict_register(ledgers)

        # Should serialize without error
        json_str = json.dumps(register)
        assert len(json_str) > 0

        # Should round-trip
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["schema_version"] == "1.0.0"


class TestConsensusVsFusionCrossCheck:
    """
    Tests for consensus vs fusion consistency cross-check.

    SHADOW MODE: These tests verify the cross-check logic, determinism, and integration.
    """

    def test_summarize_consensus_vs_fusion_conflict_case(self) -> None:
        """Verify CONFLICT status when GGFL is L0/ALLOW but register has high conflicts."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
            summarize_consensus_vs_fusion,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 8, "consensus_band": "LOW", "agreement_rate": 0.5, "predictive_risk_band": "HIGH"},
        ]

        register = build_consensus_conflict_register(ledgers, high_conflict_threshold=5)

        # GGFL reports L0/ALLOW
        ggfl_results: Dict[str, Any] = {
            "escalation": {"level_name": "L0_NOMINAL"},
            "fusion_result": {"decision": "ALLOW"},
        }

        crosscheck = summarize_consensus_vs_fusion(register, ggfl_results)

        assert crosscheck["consistency_status"] == "CONFLICT"
        assert len(crosscheck["examples"]) > 0
        assert "CAL-EXP-1" in str(crosscheck["examples"])
        assert len(crosscheck["advisory_notes"]) > 0

    def test_summarize_consensus_vs_fusion_tension_case(self) -> None:
        """Verify TENSION status when GGFL WARNING and register has high risk count."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
            summarize_consensus_vs_fusion,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 2, "consensus_band": "MEDIUM", "agreement_rate": 0.7, "predictive_risk_band": "HIGH"},
            {"cal_id": "CAL-EXP-2", "conflict_count": 3, "consensus_band": "LOW", "agreement_rate": 0.6, "predictive_risk_band": "HIGH"},
        ]

        register = build_consensus_conflict_register(ledgers, high_conflict_threshold=5)

        # GGFL reports WARNING
        ggfl_results: Dict[str, Any] = {
            "escalation": {"level_name": "L1_WARNING"},
            "fusion_result": {"decision": "ALLOW"},
        }

        crosscheck = summarize_consensus_vs_fusion(register, ggfl_results)

        assert crosscheck["consistency_status"] == "TENSION"
        assert len(crosscheck["advisory_notes"]) > 0
        assert "high-risk" in str(crosscheck["advisory_notes"]).lower() or "HIGH" in str(crosscheck["advisory_notes"])

    def test_summarize_consensus_vs_fusion_consistent_case(self) -> None:
        """Verify CONSISTENT status when register and GGFL align."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
            summarize_consensus_vs_fusion,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 1, "consensus_band": "HIGH", "agreement_rate": 0.9, "predictive_risk_band": "LOW"},
        ]

        register = build_consensus_conflict_register(ledgers, high_conflict_threshold=5)

        # GGFL reports L0/ALLOW (consistent with low conflicts)
        ggfl_results: Dict[str, Any] = {
            "escalation": {"level_name": "L0_NOMINAL"},
            "fusion_result": {"decision": "ALLOW"},
        }

        crosscheck = summarize_consensus_vs_fusion(register, ggfl_results)

        assert crosscheck["consistency_status"] == "CONSISTENT"
        assert len(crosscheck["advisory_notes"]) > 0

    def test_summarize_consensus_vs_fusion_deterministic(self) -> None:
        """Verify cross-check output is deterministic."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
            summarize_consensus_vs_fusion,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-2", "conflict_count": 6, "consensus_band": "LOW", "agreement_rate": 0.5, "predictive_risk_band": "HIGH"},
            {"cal_id": "CAL-EXP-1", "conflict_count": 7, "consensus_band": "LOW", "agreement_rate": 0.4, "predictive_risk_band": "HIGH"},
        ]

        register = build_consensus_conflict_register(ledgers, high_conflict_threshold=5)

        ggfl_results: Dict[str, Any] = {
            "escalation": {"level_name": "L0_NOMINAL"},
            "fusion_result": {"decision": "ALLOW"},
        }

        crosscheck1 = summarize_consensus_vs_fusion(register, ggfl_results)
        crosscheck2 = summarize_consensus_vs_fusion(register, ggfl_results)

        assert crosscheck1 == crosscheck2, "Cross-check output should be deterministic"

        # Verify examples are sorted deterministically
        if crosscheck1["examples"]:
            cal_ids = [ex["cal_id"] for ex in crosscheck1["examples"]]
            assert cal_ids == sorted(cal_ids), "Examples should be sorted deterministically"

    def test_summarize_consensus_vs_fusion_json_safe(self) -> None:
        """Verify cross-check output is JSON-safe."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
            summarize_consensus_vs_fusion,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 3, "consensus_band": "MEDIUM", "agreement_rate": 0.75, "predictive_risk_band": "MEDIUM"},
        ]

        register = build_consensus_conflict_register(ledgers)
        ggfl_results: Dict[str, Any] = {
            "escalation": {"level_name": "L1_WARNING"},
        }

        crosscheck = summarize_consensus_vs_fusion(register, ggfl_results)

        # Should serialize without error
        json_str = json.dumps(crosscheck)
        assert len(json_str) > 0

        # Should round-trip
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["schema_version"] == "1.0.0"
        assert parsed["consistency_status"] in ("CONSISTENT", "TENSION", "CONFLICT")

    def test_summarize_consensus_vs_fusion_empty_register(self) -> None:
        """Verify cross-check handles empty register."""
        from backend.health.consensus_polygraph_adapter import (
            summarize_consensus_vs_fusion,
        )

        crosscheck = summarize_consensus_vs_fusion({})

        assert crosscheck["consistency_status"] == "CONSISTENT"
        assert len(crosscheck["advisory_notes"]) > 0

    def test_summarize_consensus_vs_fusion_no_ggfl(self) -> None:
        """Verify cross-check handles missing GGFL results."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
            summarize_consensus_vs_fusion,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 1, "consensus_band": "HIGH", "agreement_rate": 0.9, "predictive_risk_band": "LOW"},
        ]

        register = build_consensus_conflict_register(ledgers)

        crosscheck = summarize_consensus_vs_fusion(register, None)

        assert crosscheck["consistency_status"] == "CONSISTENT"

    def test_attach_consensus_conflict_register_with_fusion_crosscheck(self) -> None:
        """Verify evidence attachment includes fusion_crosscheck."""
        from backend.health.consensus_polygraph_adapter import (
            attach_consensus_conflict_register_to_evidence,
            build_consensus_conflict_register,
            summarize_consensus_vs_fusion,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 2, "consensus_band": "HIGH", "agreement_rate": 0.9, "predictive_risk_band": "LOW"},
        ]

        register = build_consensus_conflict_register(ledgers)
        ggfl_results: Dict[str, Any] = {
            "escalation": {"level_name": "L0_NOMINAL"},
        }

        crosscheck = summarize_consensus_vs_fusion(register, ggfl_results)
        evidence: Dict[str, Any] = {}

        attach_consensus_conflict_register_to_evidence(evidence, register, crosscheck)

        assert "governance" in evidence
        assert "consensus_conflict_register" in evidence["governance"]
        assert "fusion_crosscheck" in evidence["governance"]["consensus_conflict_register"]

        attached_crosscheck = evidence["governance"]["consensus_conflict_register"]["fusion_crosscheck"]
        assert attached_crosscheck["consistency_status"] in ("CONSISTENT", "TENSION", "CONFLICT")

    def test_attach_consensus_conflicts_signal(self) -> None:
        """Verify status hook attaches consensus_conflicts signal."""
        from backend.health.consensus_polygraph_adapter import (
            attach_consensus_conflicts_signal,
            build_consensus_conflict_register,
            summarize_consensus_vs_fusion,
        )

        ledgers: List[Dict[str, Any]] = [
            {"cal_id": "CAL-EXP-1", "conflict_count": 6, "consensus_band": "LOW", "agreement_rate": 0.5, "predictive_risk_band": "HIGH"},
        ]

        register = build_consensus_conflict_register(ledgers, high_conflict_threshold=5)
        ggfl_results: Dict[str, Any] = {
            "escalation": {"level_name": "L0_NOMINAL"},
        }

        crosscheck = summarize_consensus_vs_fusion(register, ggfl_results)
        signals: Dict[str, Any] = {}

        attach_consensus_conflicts_signal(signals, crosscheck, register)

        assert "consensus_conflicts" in signals
        assert signals["consensus_conflicts"]["experiments_high_conflict_count"] == 1
        assert signals["consensus_conflicts"]["high_risk_band_count"] == 1
        assert signals["consensus_conflicts"]["fusion_consistency_status"] == "CONFLICT"

    def test_attach_consensus_conflicts_signal_no_crosscheck(self) -> None:
        """Verify status hook does nothing if crosscheck is missing."""
        from backend.health.consensus_polygraph_adapter import (
            attach_consensus_conflicts_signal,
            build_consensus_conflict_register,
        )

        register = build_consensus_conflict_register([])
        signals: Dict[str, Any] = {}

        attach_consensus_conflicts_signal(signals, None, register)

        assert "consensus_conflicts" not in signals

    def test_summarize_consensus_vs_fusion_examples_limit(self) -> None:
        """Verify examples are limited to 5 and sorted deterministically."""
        from backend.health.consensus_polygraph_adapter import (
            build_consensus_conflict_register,
            summarize_consensus_vs_fusion,
        )

        # Create 7 high-conflict experiments
        ledgers: List[Dict[str, Any]] = [
            {"cal_id": f"CAL-EXP-{i}", "conflict_count": 6, "consensus_band": "LOW", "agreement_rate": 0.5, "predictive_risk_band": "HIGH"}
            for i in range(1, 8)
        ]

        register = build_consensus_conflict_register(ledgers, high_conflict_threshold=5)

        ggfl_results: Dict[str, Any] = {
            "escalation": {"level_name": "L0_NOMINAL"},
            "fusion_result": {"decision": "ALLOW"},
        }

        crosscheck = summarize_consensus_vs_fusion(register, ggfl_results)

        assert len(crosscheck["examples"]) <= 5, "Examples should be limited to 5"
        if crosscheck["examples"]:
            cal_ids = [ex["cal_id"] for ex in crosscheck["examples"]]
            assert cal_ids == sorted(cal_ids), "Examples should be sorted deterministically"

