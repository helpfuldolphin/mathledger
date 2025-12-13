"""P5 Replay Safety Smoke-Test Harness.

This module provides smoke-test stubs for validating the P5 replay safety
extract/build/attach chain.

Reference: docs/system_law/Replay_Safety_P5_Engineering_Plan.md Section 4

SMOKE-TEST READINESS CHECKLIST:
    A. Module Availability
    B. Signal Flow
    C. P5 Field Presence
    D. P5 Band Logic
    E. GGFL Integration
    F. Evidence Attachment
    G. Auditor Script
    H. SHADOW Mode

SHADOW MODE CONTRACT:
- All tests are observational only
- No governance decisions are made
- Tests verify signal structure, not control flow
"""

import json
import pytest
from typing import Any, Dict, List


# =============================================================================
# SMOKE-TEST FIXTURES
# =============================================================================

@pytest.fixture
def smoke_replay_logs_nominal() -> List[Dict[str, Any]]:
    """Nominal replay logs for smoke testing."""
    return [
        {
            "cycle_id": f"cycle_{i:03d}",
            "trace_hash": f"hash_{i:03d}",
            "timestamp": f"2025-12-10T00:{i:02d}:00Z",
            "latency_ms": 50.0 + i,
        }
        for i in range(10)
    ]


@pytest.fixture
def smoke_expected_hashes_nominal() -> Dict[str, str]:
    """Nominal expected hashes (all match)."""
    return {f"cycle_{i:03d}": f"hash_{i:03d}" for i in range(10)}


@pytest.fixture
def smoke_radar_nominal() -> Dict[str, Any]:
    """Nominal radar for smoke testing."""
    return {
        "status": "ok",
        "governance_alignment": "aligned",
        "safe_for_policy_update": True,
        "safe_for_promotion": True,
        "conflict": False,
        "reasons": [],
        "metrics": {
            "determinism_score": 100,
            "hash_match_rate": 1.0,
            "violation_count": 0,
        },
    }


@pytest.fixture
def smoke_promotion_eval_nominal() -> Dict[str, Any]:
    """Nominal promotion eval for smoke testing."""
    return {
        "status": "ok",
        "safe": True,
        "safe_for_policy_update": True,
        "safe_for_promotion": True,
        "reasons": [],
    }


@pytest.fixture
def smoke_evidence_base() -> Dict[str, Any]:
    """Base evidence pack for smoke testing."""
    return {
        "run_id": "smoke_test_run",
        "timestamp": "2025-12-10T00:00:00Z",
        "governance": {},
    }


# =============================================================================
# A. MODULE AVAILABILITY
# =============================================================================

class TestModuleAvailability:
    """Smoke tests for module availability (Checklist A)."""

    def test_replay_governance_adapter_importable(self) -> None:
        """backend/health/replay_governance_adapter.py must be importable."""
        from backend.health import replay_governance_adapter

        assert replay_governance_adapter is not None

    def test_p5_functions_importable(self) -> None:
        """All P5 functions must be importable without error."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
            attach_p5_replay_governance_to_evidence,
            replay_for_alignment_view_p5,
        )

        assert callable(extract_p5_replay_safety_from_logs)
        assert callable(build_p5_replay_governance_tile)
        assert callable(attach_p5_replay_governance_to_evidence)
        assert callable(replay_for_alignment_view_p5)

    def test_p5_constants_available(self) -> None:
        """P5 constants must be available."""
        from backend.health.replay_governance_adapter import (
            P5_REPLAY_SCHEMA_VERSION,
            P5_DETERMINISM_GREEN_THRESHOLD,
            P5_DETERMINISM_YELLOW_THRESHOLD,
        )

        assert P5_REPLAY_SCHEMA_VERSION == "1.0.0"
        assert P5_DETERMINISM_GREEN_THRESHOLD == 0.85
        assert P5_DETERMINISM_YELLOW_THRESHOLD == 0.70


# =============================================================================
# B. SIGNAL FLOW
# =============================================================================

class TestSignalFlow:
    """Smoke tests for signal flow (Checklist B)."""

    def test_extract_returns_valid_structure(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
    ) -> None:
        """extract_p5_replay_safety_from_logs() must return valid structure."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        # Required fields
        assert "status" in signal
        assert "determinism_rate" in signal
        assert "determinism_band" in signal
        assert "p5_grade" in signal

    def test_build_produces_json_safe_output(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
    ) -> None:
        """build_p5_replay_governance_tile() must produce JSON-safe output."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        # Must serialize without error
        json_str = json.dumps(tile)
        assert json_str is not None
        assert len(json_str) > 0

    def test_attach_is_non_mutating(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
        smoke_evidence_base: Dict[str, Any],
    ) -> None:
        """attach_p5_replay_governance_to_evidence() must be non-mutating."""
        import copy
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
            attach_p5_replay_governance_to_evidence,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        original_evidence = copy.deepcopy(smoke_evidence_base)

        result = attach_p5_replay_governance_to_evidence(
            smoke_evidence_base, signal, tile
        )

        # Original must be unchanged
        assert smoke_evidence_base == original_evidence
        # Result must be different object
        assert result is not smoke_evidence_base


# =============================================================================
# C. P5 FIELD PRESENCE
# =============================================================================

class TestP5FieldPresence:
    """Smoke tests for P5 field presence (Checklist C)."""

    def test_tile_includes_telemetry_source_real(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
    ) -> None:
        """Tile must include telemetry_source='real'."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        assert tile.get("telemetry_source") == "real"

    def test_tile_includes_production_run_id(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
    ) -> None:
        """Tile must include production_run_id (non-empty)."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        assert tile.get("production_run_id") == "smoke_prod_run"
        assert len(tile.get("production_run_id", "")) > 0

    def test_tile_includes_phase_p5(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
    ) -> None:
        """Tile must include phase='P5'."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        assert tile.get("phase") == "P5"

    def test_tile_includes_determinism_band(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
    ) -> None:
        """Tile must include determinism_band (GREEN/YELLOW/RED)."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        assert tile.get("determinism_band") in ("GREEN", "YELLOW", "RED")


# =============================================================================
# D. P5 BAND LOGIC
# =============================================================================

class TestP5BandLogic:
    """Smoke tests for P5 band logic (Checklist D)."""

    def test_green_band_threshold(self) -> None:
        """determinism_rate >= 0.85 must produce GREEN band."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        # 9 matches, 1 mismatch = 90% > 85% -> GREEN
        logs = [{"cycle_id": f"c{i}", "trace_hash": f"h{i}"} for i in range(10)]
        expected = {f"c{i}": f"h{i}" for i in range(9)}
        expected["c9"] = "wrong"  # 1 mismatch

        signal = extract_p5_replay_safety_from_logs(
            logs,
            production_run_id="test",
            expected_hashes=expected,
            telemetry_source="real",
        )

        assert signal["determinism_rate"] == 0.9
        assert signal["determinism_band"] == "GREEN"

    def test_yellow_band_threshold(self) -> None:
        """0.70 <= determinism_rate < 0.85 must produce YELLOW band."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        # 8 matches, 2 mismatches = 80% -> YELLOW
        logs = [{"cycle_id": f"c{i}", "trace_hash": f"h{i}"} for i in range(10)]
        expected = {f"c{i}": f"h{i}" for i in range(8)}
        expected["c8"] = "wrong"
        expected["c9"] = "wrong"

        signal = extract_p5_replay_safety_from_logs(
            logs,
            production_run_id="test",
            expected_hashes=expected,
            telemetry_source="real",
        )

        assert signal["determinism_rate"] == 0.8
        assert signal["determinism_band"] == "YELLOW"

    def test_red_band_threshold(self) -> None:
        """determinism_rate < 0.70 must produce RED band."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        # 6 matches, 4 mismatches = 60% -> RED
        logs = [{"cycle_id": f"c{i}", "trace_hash": f"h{i}"} for i in range(10)]
        expected = {f"c{i}": f"h{i}" for i in range(6)}
        expected["c6"] = "wrong"
        expected["c7"] = "wrong"
        expected["c8"] = "wrong"
        expected["c9"] = "wrong"

        signal = extract_p5_replay_safety_from_logs(
            logs,
            production_run_id="test",
            expected_hashes=expected,
            telemetry_source="real",
        )

        assert signal["determinism_rate"] == 0.6
        assert signal["determinism_band"] == "RED"


# =============================================================================
# E. GGFL INTEGRATION
# =============================================================================

class TestGGFLIntegration:
    """Smoke tests for GGFL integration (Checklist E)."""

    def test_replay_for_alignment_view_p5_handles_p5_signals(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
    ) -> None:
        """replay_for_alignment_view_p5() must handle P5 signals."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            replay_for_alignment_view_p5,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        result = replay_for_alignment_view_p5(signal)

        assert "status" in result
        assert "alignment" in result
        assert "conflict" in result
        assert "top_reasons" in result

    def test_replay_prefix_stripping(self) -> None:
        """[Replay] prefix must be stripped for P5 reasons."""
        from backend.health.replay_governance_adapter import replay_for_alignment_view_p5

        signal = {
            "status": "warn",
            "determinism_band": "YELLOW",
            "p5_grade": True,
            "telemetry_source": "real",
            "reasons": ["[Replay] Test reason 1", "[Replay] Test reason 2"],
        }

        result = replay_for_alignment_view_p5(signal)

        for reason in result["top_reasons"]:
            assert not reason.startswith("[Replay]")

    def test_top_reasons_limited_to_five(self) -> None:
        """top_reasons must be limited to 5."""
        from backend.health.replay_governance_adapter import replay_for_alignment_view_p5

        signal = {
            "status": "block",
            "determinism_band": "RED",
            "p5_grade": True,
            "telemetry_source": "real",
            "reasons": [f"[Replay] Reason {i}" for i in range(20)],
        }

        result = replay_for_alignment_view_p5(signal)

        assert len(result["top_reasons"]) == 5


# =============================================================================
# F. EVIDENCE ATTACHMENT
# =============================================================================

class TestEvidenceAttachment:
    """Smoke tests for evidence attachment (Checklist F)."""

    def test_evidence_includes_governance_replay(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
        smoke_evidence_base: Dict[str, Any],
    ) -> None:
        """P5 evidence must include governance.replay."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
            attach_p5_replay_governance_to_evidence,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        result = attach_p5_replay_governance_to_evidence(
            smoke_evidence_base, signal, tile
        )

        assert "governance" in result
        assert "replay" in result["governance"]

    def test_evidence_includes_governance_replay_p5(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
        smoke_evidence_base: Dict[str, Any],
    ) -> None:
        """P5 evidence must include governance.replay_p5 (extension fields)."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
            attach_p5_replay_governance_to_evidence,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        result = attach_p5_replay_governance_to_evidence(
            smoke_evidence_base, signal, tile
        )

        assert "replay_p5" in result["governance"]

    def test_replay_p5_grade_true_when_valid(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
        smoke_evidence_base: Dict[str, Any],
    ) -> None:
        """replay_p5_grade must be true when valid P5 signal."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
            attach_p5_replay_governance_to_evidence,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        result = attach_p5_replay_governance_to_evidence(
            smoke_evidence_base, signal, tile
        )

        assert result["replay_p5_grade"] is True


# =============================================================================
# G. AUDITOR SCRIPT
# =============================================================================

class TestAuditorScript:
    """Smoke tests for auditor script (Checklist G)."""

    def test_audit_p5_replay_safety_ok(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
        smoke_evidence_base: Dict[str, Any],
    ) -> None:
        """audit_p5_replay_safety() must return P5_REPLAY_OK for nominal signal."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
            attach_p5_replay_governance_to_evidence,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        evidence = attach_p5_replay_governance_to_evidence(
            smoke_evidence_base, signal, tile
        )

        verdict = audit_p5_replay_safety(evidence)
        assert verdict == "P5_REPLAY_OK"

    def test_audit_p5_replay_safety_warn(self) -> None:
        """audit_p5_replay_safety() must return P5_REPLAY_WARN for status=warn."""
        evidence = {
            "governance": {
                "replay": {
                    "status": "warn",
                    "governance_alignment": "aligned",
                    "conflict": False,
                },
                "replay_p5": {"telemetry_source": "real"},
            },
            "replay_p5_grade": True,
        }

        verdict = audit_p5_replay_safety(evidence)
        assert verdict == "P5_REPLAY_WARN"

    def test_audit_p5_replay_safety_investigate_on_block(self) -> None:
        """audit_p5_replay_safety() must return P5_REPLAY_INVESTIGATE for status=block."""
        evidence = {
            "governance": {
                "replay": {
                    "status": "block",
                    "governance_alignment": "divergent",
                    "conflict": True,
                },
                "replay_p5": {"telemetry_source": "real"},
            },
            "replay_p5_grade": True,
        }

        verdict = audit_p5_replay_safety(evidence)
        assert verdict == "P5_REPLAY_INVESTIGATE"

    def test_audit_p5_replay_safety_investigate_on_conflict(self) -> None:
        """audit_p5_replay_safety() must return P5_REPLAY_INVESTIGATE for conflict=true."""
        evidence = {
            "governance": {
                "replay": {
                    "status": "ok",
                    "governance_alignment": "aligned",
                    "conflict": True,
                },
                "replay_p5": {"telemetry_source": "real"},
            },
            "replay_p5_grade": True,
        }

        verdict = audit_p5_replay_safety(evidence)
        assert verdict == "P5_REPLAY_INVESTIGATE"


# =============================================================================
# H. SHADOW MODE
# =============================================================================

class TestShadowMode:
    """Smoke tests for SHADOW mode (Checklist H)."""

    def test_no_function_modifies_control_flow(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
        smoke_evidence_base: Dict[str, Any],
    ) -> None:
        """No function modifies control flow (SHADOW mode)."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
            attach_p5_replay_governance_to_evidence,
        )

        # All functions are pure - they return values, no side effects
        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        result = attach_p5_replay_governance_to_evidence(
            smoke_evidence_base, signal, tile
        )

        # Verify SHADOW mode contract in tile
        assert tile.get("mode") == "SHADOW"
        assert tile.get("shadow_mode_contract", {}).get("observational_only") is True
        assert tile.get("shadow_mode_contract", {}).get("no_control_flow_influence") is True
        assert tile.get("shadow_mode_contract", {}).get("no_governance_modification") is True

    def test_all_tiles_include_shadow_mode_contract(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
    ) -> None:
        """All tiles must include shadow_mode_contract."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        assert "shadow_mode_contract" in tile

    def test_no_gating_logic_present(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
    ) -> None:
        """No gating logic present in tile."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
        )

        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        json_str = json.dumps(tile)

        # Should not contain gating keywords
        assert "gate" not in json_str.lower()
        assert "block_execution" not in json_str.lower()
        assert "halt" not in json_str.lower()


# =============================================================================
# AUDITOR SCRIPT (from Engineering Plan Section 3)
# =============================================================================

def audit_p5_replay_safety(evidence_pack: Dict[str, Any]) -> str:
    """
    5-Step Auditor Script for P5 Replay Safety Verification.

    Reference: Replay_Safety_P5_Engineering_Plan.md Section 3

    Returns: "P5_REPLAY_OK" | "P5_REPLAY_WARN" | "P5_REPLAY_INVESTIGATE"
    """
    # STEP 1: Locate replay_governance in evidence pack
    governance = evidence_pack.get("governance", {})
    replay_gov = governance.get("replay") or governance.get("replay_governance")

    if replay_gov is None:
        return "P5_REPLAY_INVESTIGATE"  # Missing replay governance

    # STEP 2: Extract status, alignment, conflict
    status = replay_gov.get("status", "").lower()
    alignment = replay_gov.get("governance_alignment", replay_gov.get("alignment", ""))
    conflict = replay_gov.get("conflict", False)

    # STEP 3: Check P5-grade indicators
    is_p5_grade = (
        replay_gov.get("telemetry_source") == "real"
        or evidence_pack.get("replay_p5_grade", False)
        or governance.get("replay_p5") is not None
    )

    if not is_p5_grade:
        # Not P5-grade evidence, investigate
        return "P5_REPLAY_INVESTIGATE"

    # STEP 4: Apply verdict logic
    if status == "block":
        return "P5_REPLAY_INVESTIGATE"

    if conflict:
        return "P5_REPLAY_INVESTIGATE"

    if alignment == "divergent":
        return "P5_REPLAY_INVESTIGATE"

    if status == "warn":
        return "P5_REPLAY_WARN"

    if status == "ok" and alignment in ("aligned", ""):
        return "P5_REPLAY_OK"

    # STEP 5: Default to investigate for unexpected states
    return "P5_REPLAY_INVESTIGATE"


# =============================================================================
# SMOKE-TEST CHAIN HARNESS
# =============================================================================

class TestP5SmokeChain:
    """End-to-end smoke test for extract/build/attach chain."""

    def test_full_p5_chain_nominal(
        self,
        smoke_replay_logs_nominal: List[Dict[str, Any]],
        smoke_expected_hashes_nominal: Dict[str, str],
        smoke_radar_nominal: Dict[str, Any],
        smoke_promotion_eval_nominal: Dict[str, Any],
        smoke_evidence_base: Dict[str, Any],
    ) -> None:
        """Full P5 chain: extract -> build -> attach."""
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            build_p5_replay_governance_tile,
            attach_p5_replay_governance_to_evidence,
            replay_for_alignment_view_p5,
        )

        # Step 1: Extract P5 signal from logs
        signal = extract_p5_replay_safety_from_logs(
            smoke_replay_logs_nominal,
            production_run_id="smoke_prod_run",
            expected_hashes=smoke_expected_hashes_nominal,
            telemetry_source="real",
        )

        assert signal["status"] == "ok"
        assert signal["p5_grade"] is True

        # Step 2: Build P5 tile
        tile = build_p5_replay_governance_tile(
            signal, smoke_radar_nominal, smoke_promotion_eval_nominal
        )

        assert tile["phase"] == "P5"
        assert tile["status"] == "ok"

        # Step 3: Attach to evidence
        evidence = attach_p5_replay_governance_to_evidence(
            smoke_evidence_base, signal, tile
        )

        assert evidence["replay_p5_grade"] is True
        assert evidence["replay_safety_ok"] is True

        # Step 4: GGFL integration
        ggfl_view = replay_for_alignment_view_p5(signal)

        assert ggfl_view["status"] == "ok"
        assert ggfl_view["alignment"] == "aligned"

        # Step 5: Auditor script
        verdict = audit_p5_replay_safety(evidence)

        assert verdict == "P5_REPLAY_OK"
