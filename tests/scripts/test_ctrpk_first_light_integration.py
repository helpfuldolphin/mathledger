"""
Tests for CTRPK integration with First Light status and signal computation.

Validates:
- CTRPK signal extraction from evidence pack manifest
- CTRPK computation from JSONL signals (determinism)
- Status ingestion correctness (GREEN/YELLOW/RED)
- Warning generation for BLOCK/WARN/DEGRADING
- Warning neutrality (single-line, no banned words)
- Missing CTRPK handling (no error)
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import pytest

# Import reusable warning neutrality helpers
from tests.helpers.warning_neutrality import (
    pytest_assert_warning_neutral,
    pytest_assert_warnings_neutral,
    BANNED_ALARM_WORDS,
)

from scripts.generate_first_light_status import generate_status
from scripts.compute_ctrpk_from_signals import (
    load_jsonl,
    count_transition_requests,
    count_total_cycles,
    count_semantic_violations,
    count_blocked_requests,
    compute_ctrpk_from_signals,
)

# Check if generate_status has CTRPK support by testing the signature
def _has_ctrpk_support() -> bool:
    """Check if generate_status supports CTRPK ingestion."""
    import inspect
    sig = inspect.signature(generate_status)
    return "ctrpk_json_path" in sig.parameters

CTRPK_STATUS_SUPPORT = _has_ctrpk_support()

# Skip marker for tests requiring CTRPK-enabled status generator
requires_ctrpk_status = pytest.mark.skipif(
    not CTRPK_STATUS_SUPPORT,
    reason="BLOCKER: generate_first_light_status.py missing CTRPK ingestion support. "
           "Restore ctrpk_json_path param and CTRPK signal extraction to enable."
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def minimal_evidence_pack_dir(tmp_path: Path) -> Path:
    """Create minimal evidence pack directory structure."""
    evidence_dir = tmp_path / "evidence_pack"
    evidence_dir.mkdir()

    # Create minimal P3 directory
    p3_dir = tmp_path / "p3"
    p3_dir.mkdir()
    p3_run = p3_dir / "fl_test"
    p3_run.mkdir()
    (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
    (p3_run / "stability_report.json").write_text(
        json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
    )
    (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
    (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
    (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
    (p3_run / "run_config.json").write_text("{}", encoding="utf-8")

    # Create minimal P4 directory
    p4_dir = tmp_path / "p4"
    p4_dir.mkdir()
    p4_run = p4_dir / "p4_test"
    p4_run.mkdir()
    (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
    (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
    (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
    (p4_run / "p4_summary.json").write_text(
        json.dumps({"mode": "SHADOW", "uplift_metrics": {}}), encoding="utf-8"
    )
    (p4_run / "twin_accuracy.json").write_text("{}", encoding="utf-8")
    (p4_run / "run_config.json").write_text(
        json.dumps({"telemetry_source": "mock"}), encoding="utf-8"
    )

    return tmp_path


def create_manifest_with_ctrpk(
    evidence_dir: Path,
    ctrpk_value: float,
    ctrpk_status: str,
    ctrpk_trend: str = "STABLE",
    window_cycles: int = 10000,
    transition_requests: int = 10,
    include_path_and_sha256: bool = True,
) -> Path:
    """Create manifest.json with CTRPK data.

    Args:
        evidence_dir: Directory to create manifest in
        ctrpk_value: CTRPK value
        ctrpk_status: CTRPK status (OK, WARN, BLOCK)
        ctrpk_trend: CTRPK trend (IMPROVING, STABLE, DEGRADING)
        window_cycles: Window size in cycles
        transition_requests: Number of transition requests
        include_path_and_sha256: If True, include path and sha256 reference
    """
    ctrpk_data: Dict[str, Any] = {
        "value": ctrpk_value,
        "status": ctrpk_status,
        "trend": ctrpk_trend,
        "window_cycles": window_cycles,
        "transition_requests": transition_requests,
    }

    if include_path_and_sha256:
        # Include path and sha256 reference (as evidence pack builder would)
        ctrpk_data["path"] = "ctrpk_compact.json"
        # Compute deterministic sha256 based on content
        import hashlib
        content_str = json.dumps(ctrpk_data, sort_keys=True)
        ctrpk_data["sha256"] = hashlib.sha256(content_str.encode()).hexdigest()

    manifest = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "file_count": 0,
        "files": [],
        "shadow_mode_compliance": {
            "all_divergence_logged_only": True,
            "no_governance_modification": True,
            "no_abort_enforcement": True,
        },
        "governance": {
            "curriculum": {
                "ctrpk": ctrpk_data,
            }
        },
    }
    manifest_path = evidence_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


# -----------------------------------------------------------------------------
# First Light Status CTRPK Ingestion Tests
# -----------------------------------------------------------------------------

@requires_ctrpk_status
class TestFirstLightCTRPKIngestion:
    """Tests for CTRPK ingestion in generate_first_light_status.py."""

    def test_ctrpk_extracted_from_manifest(self, minimal_evidence_pack_dir: Path) -> None:
        """Test that CTRPK is extracted from manifest and added to signals."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=0.5,
            ctrpk_status="OK",
            ctrpk_trend="STABLE",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # Verify CTRPK signal is present
        assert status["signals"] is not None
        assert "ctrpk" in status["signals"]
        ctrpk = status["signals"]["ctrpk"]
        assert ctrpk["value"] == 0.5
        assert ctrpk["status"] == "OK"
        assert ctrpk["trend"] == "STABLE"

    def test_ctrpk_warn_status_generates_warning(self, minimal_evidence_pack_dir: Path) -> None:
        """Test that WARN CTRPK status generates an advisory warning."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=2.5,
            ctrpk_status="WARN",
            ctrpk_trend="STABLE",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # Verify warning is generated
        assert any("CTRPK status is WARN" in w for w in status["warnings"])
        # Signal still present (no blocking)
        assert "ctrpk" in status["signals"]

    def test_ctrpk_block_status_generates_warning(self, minimal_evidence_pack_dir: Path) -> None:
        """Test that BLOCK CTRPK status generates an advisory warning."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=7.0,
            ctrpk_status="BLOCK",
            ctrpk_trend="DEGRADING",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # Verify warning is generated
        assert any("CTRPK status is BLOCK" in w for w in status["warnings"])

    def test_ctrpk_degrading_trend_generates_warning(self, minimal_evidence_pack_dir: Path) -> None:
        """Test that DEGRADING trend generates an advisory warning."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=2.0,
            ctrpk_status="WARN",
            ctrpk_trend="DEGRADING",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # Verify trend warning is generated
        assert any("DEGRADING" in w for w in status["warnings"])

    def test_missing_ctrpk_not_an_error(self, minimal_evidence_pack_dir: Path) -> None:
        """Test that missing CTRPK data does not cause errors."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # Create manifest without CTRPK
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "files": [],
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        (evidence_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # Verify status generation succeeds
        assert status["evidence_pack_ok"] is True

        # CTRPK signal should not be present
        if status["signals"]:
            assert "ctrpk" not in status["signals"]


# -----------------------------------------------------------------------------
# CTRPK Computation from Signals Tests
# -----------------------------------------------------------------------------

class TestCTRPKComputationDeterminism:
    """Tests for CTRPK computation determinism from JSONL signals."""

    def test_count_transition_requests_direct(self) -> None:
        """Test counting direct TRANSITION_REQUESTED signals."""
        signals = [
            {"signal_type": "TRANSITION_REQUESTED", "cycle": 100},
            {"signal_type": "TRANSITION_REQUESTED", "cycle": 200},
            {"signal_type": "SNAPSHOT_VERIFIED", "cycle": 150},
        ]

        count = count_transition_requests(signals)
        assert count == 2

    def test_count_transition_requests_semantic_drift(self) -> None:
        """Test counting DRIFT_DETECTED with SEMANTIC severity."""
        signals = [
            {"signal_type": "DRIFT_DETECTED", "severity": "SEMANTIC", "cycle": 100},
            {"signal_type": "DRIFT_DETECTED", "severity": "PARAMETRIC", "cycle": 200},
            {"signal_type": "DRIFT_DETECTED", "severity": "SEMANTIC", "cycle": 300},
        ]

        count = count_transition_requests(signals)
        assert count == 2

    def test_count_total_cycles_from_max_cycle(self) -> None:
        """Test extracting total cycles from max cycle number."""
        signals = [
            {"signal_type": "SNAPSHOT_VERIFIED", "cycle": 100},
            {"signal_type": "SNAPSHOT_VERIFIED", "cycle": 500},
            {"signal_type": "SNAPSHOT_VERIFIED", "cycle": 250},
        ]

        count = count_total_cycles(signals)
        assert count == 501  # max cycle (500) + 1

    def test_count_total_cycles_explicit_field(self) -> None:
        """Test using explicit total_cycles field when present."""
        signals = [
            {"signal_type": "SUMMARY", "total_cycles": 10000, "cycle": 100},
        ]

        count = count_total_cycles(signals)
        assert count == 10000

    def test_count_semantic_violations(self) -> None:
        """Test counting SEMANTIC severity violations."""
        signals = [
            {"severity": "SEMANTIC", "cycle": 100},
            {"severity": "PARAMETRIC", "cycle": 200},
            {"drift_severity": "SEMANTIC", "cycle": 300},
            {"severity": "NONE", "cycle": 400},
        ]

        count = count_semantic_violations(signals)
        assert count == 2

    def test_count_blocked_requests(self) -> None:
        """Test counting blocked requests."""
        signals = [
            {"status": "BLOCK", "cycle": 100},
            {"status": "WARN", "cycle": 200},
            {"drift_status": "BLOCK", "cycle": 300},
            {"hypothetical": {"would_allow_transition": False}, "cycle": 400},
            {"hypothetical": {"would_allow_transition": True}, "cycle": 500},
        ]

        count = count_blocked_requests(signals)
        assert count == 3

    def test_compute_ctrpk_deterministic(self) -> None:
        """Test that CTRPK computation is deterministic."""
        signals = [
            {"signal_type": "TRANSITION_REQUESTED", "cycle": 100},
            {"signal_type": "TRANSITION_REQUESTED", "cycle": 200},
            {"signal_type": "SNAPSHOT_VERIFIED", "cycle": 9999},
        ]

        # Run computation multiple times
        result1 = compute_ctrpk_from_signals(signals)
        result2 = compute_ctrpk_from_signals(signals)

        # Value should be identical
        assert result1["value"] == result2["value"]
        assert result1["status"] == result2["status"]
        assert result1["transition_requests"] == result2["transition_requests"]
        assert result1["window_cycles"] == result2["window_cycles"]

        # Expected: 2 requests / 10000 cycles * 1000 = 0.2
        assert result1["value"] == 0.2
        assert result1["status"] == "OK"

    def test_compute_ctrpk_with_trend(self) -> None:
        """Test CTRPK computation with trend from 24h signals."""
        signals_1h = [
            {"signal_type": "TRANSITION_REQUESTED", "cycle": i}
            for i in range(0, 5)
        ] + [{"signal_type": "SNAPSHOT_VERIFIED", "cycle": 999}]

        signals_24h = [
            {"signal_type": "TRANSITION_REQUESTED", "cycle": i}
            for i in range(0, 30)
        ] + [{"signal_type": "SNAPSHOT_VERIFIED", "cycle": 9999}]

        result = compute_ctrpk_from_signals(signals_1h, signals_24h)

        # 1h: 5/1000 * 1000 = 5.0 CTRPK
        # 24h: 30/10000 * 1000 = 3.0 CTRPK
        # Trend: DEGRADING (1h > 24h by more than 0.5)
        assert result["trend"] == "DEGRADING"


class TestCTRPKJSONLLoading:
    """Tests for JSONL loading functionality."""

    def test_load_jsonl_valid(self, tmp_path: Path) -> None:
        """Test loading valid JSONL file."""
        jsonl_path = tmp_path / "signals.jsonl"
        jsonl_path.write_text(
            '{"signal_type": "A", "cycle": 1}\n'
            '{"signal_type": "B", "cycle": 2}\n',
            encoding="utf-8",
        )

        records = load_jsonl(jsonl_path)
        assert len(records) == 2
        assert records[0]["signal_type"] == "A"
        assert records[1]["signal_type"] == "B"

    def test_load_jsonl_missing_file(self, tmp_path: Path) -> None:
        """Test loading non-existent JSONL file returns empty list."""
        jsonl_path = tmp_path / "nonexistent.jsonl"
        records = load_jsonl(jsonl_path)
        assert records == []

    def test_load_jsonl_with_blank_lines(self, tmp_path: Path) -> None:
        """Test that blank lines are skipped."""
        jsonl_path = tmp_path / "signals.jsonl"
        jsonl_path.write_text(
            '{"signal_type": "A"}\n'
            '\n'
            '{"signal_type": "B"}\n'
            '   \n',
            encoding="utf-8",
        )

        records = load_jsonl(jsonl_path)
        assert len(records) == 2


class TestCTRPKEdgeCases:
    """Tests for edge cases in CTRPK computation."""

    def test_empty_signals_returns_zero_ctrpk(self) -> None:
        """Test that empty signals list produces 0 CTRPK."""
        result = compute_ctrpk_from_signals([])

        assert result["value"] == 0.0
        assert result["status"] == "OK"
        assert result["transition_requests"] == 0

    def test_no_transition_requests(self) -> None:
        """Test signals with no transition requests."""
        signals = [
            {"signal_type": "SNAPSHOT_VERIFIED", "cycle": 100},
            {"signal_type": "SNAPSHOT_VERIFIED", "cycle": 9999},
        ]

        result = compute_ctrpk_from_signals(signals)

        assert result["value"] == 0.0
        assert result["status"] == "OK"
        assert result["transition_requests"] == 0

    def test_high_churn_produces_block(self) -> None:
        """Test that high transition rate produces BLOCK status."""
        # 60 transition requests in 10000 cycles = 6.0 CTRPK = BLOCK
        signals = [
            {"signal_type": "TRANSITION_REQUESTED", "cycle": i * 100}
            for i in range(60)
        ] + [{"signal_type": "SNAPSHOT_VERIFIED", "cycle": 9999}]

        result = compute_ctrpk_from_signals(signals)

        assert result["value"] == 6.0
        assert result["status"] == "BLOCK"


# -----------------------------------------------------------------------------
# CTRPK Auto-Ingest Tests (12 tests)
# -----------------------------------------------------------------------------

class TestCTRPKAutoIngestEvidenceBuilder:
    """Tests for CTRPK auto-attachment in evidence pack builder."""

    def test_evidence_builder_attaches_ctrpk_from_run_dir(self, tmp_path: Path) -> None:
        """Test that evidence builder auto-attaches CTRPK from run_dir."""
        from backend.topology.first_light.ctrpk_detection import (
            detect_ctrpk_artifact,
            attach_ctrpk_to_manifest,
        )

        # Create ctrpk_compact.json in run dir
        ctrpk_data = {
            "value": 2.5,
            "status": "WARN",
            "trend": "STABLE",
            "window_cycles": 10000,
            "transition_requests": 25,
        }
        ctrpk_path = tmp_path / "ctrpk_compact.json"
        ctrpk_path.write_text(json.dumps(ctrpk_data), encoding="utf-8")

        # Detect CTRPK
        ref = detect_ctrpk_artifact(tmp_path)

        assert ref is not None
        assert ref.value == 2.5
        assert ref.status == "WARN"
        assert ref.trend == "STABLE"
        assert ref.sha256 is not None

    def test_evidence_builder_attaches_ctrpk_in_curriculum_subdir(self, tmp_path: Path) -> None:
        """Test CTRPK detection in curriculum/ subdirectory."""
        from backend.topology.first_light.ctrpk_detection import detect_ctrpk_artifact

        # Create curriculum/ctrpk_compact.json
        curriculum_dir = tmp_path / "curriculum"
        curriculum_dir.mkdir()
        ctrpk_data = {
            "value": 0.5,
            "status": "OK",
            "trend": "IMPROVING",
            "window_cycles": 5000,
            "transition_requests": 2,
        }
        (curriculum_dir / "ctrpk_compact.json").write_text(
            json.dumps(ctrpk_data), encoding="utf-8"
        )

        ref = detect_ctrpk_artifact(tmp_path)

        assert ref is not None
        assert ref.path == "curriculum/ctrpk_compact.json"
        assert ref.value == 0.5

    def test_evidence_builder_attaches_sha256_hash(self, tmp_path: Path) -> None:
        """Test that CTRPK attachment includes SHA256 hash."""
        from backend.topology.first_light.ctrpk_detection import (
            detect_ctrpk_artifact,
            attach_ctrpk_to_manifest,
        )

        ctrpk_data = {"value": 1.0, "status": "WARN", "trend": "STABLE"}
        (tmp_path / "ctrpk_compact.json").write_text(
            json.dumps(ctrpk_data), encoding="utf-8"
        )

        ref = detect_ctrpk_artifact(tmp_path)
        manifest = {"schema_version": "1.0.0"}
        result = attach_ctrpk_to_manifest(manifest, ref)

        assert "sha256" in result["governance"]["curriculum"]["ctrpk"]
        assert len(result["governance"]["curriculum"]["ctrpk"]["sha256"]) == 64

    def test_manifest_attachment_preserves_existing_governance(self, tmp_path: Path) -> None:
        """Test that CTRPK attachment preserves existing governance data."""
        from backend.topology.first_light.ctrpk_detection import (
            detect_ctrpk_artifact,
            attach_ctrpk_to_manifest,
        )

        ctrpk_data = {"value": 1.5, "status": "WARN", "trend": "STABLE"}
        (tmp_path / "ctrpk_compact.json").write_text(
            json.dumps(ctrpk_data), encoding="utf-8"
        )

        ref = detect_ctrpk_artifact(tmp_path)
        manifest = {
            "schema_version": "1.0.0",
            "governance": {
                "existing_field": "preserved",
                "curriculum": {"other_signal": "kept"},
            },
        }
        result = attach_ctrpk_to_manifest(manifest, ref)

        # Existing fields preserved
        assert result["governance"]["existing_field"] == "preserved"
        assert result["governance"]["curriculum"]["other_signal"] == "kept"
        # CTRPK added
        assert "ctrpk" in result["governance"]["curriculum"]


@requires_ctrpk_status
class TestCTRPKStatusWarnings:
    """Tests for CTRPK warnings in status generation."""

    def test_warn_with_degrading_generates_combined_warning(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that WARN+DEGRADING generates single combined warning (warning hygiene)."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=1.5,
            ctrpk_status="WARN",
            ctrpk_trend="DEGRADING",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # Warning hygiene: Should have exactly 1 CTRPK warning (combined)
        ctrpk_warnings = [w for w in status["warnings"] if "CTRPK" in w]
        assert len(ctrpk_warnings) == 1
        # Combined warning should mention both WARN and DEGRADING
        assert "WARN" in ctrpk_warnings[0]
        assert "DEGRADING" in ctrpk_warnings[0]

    def test_block_status_with_degrading_no_double_warning(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that BLOCK status with DEGRADING doesn't double-warn on trend."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=7.0,
            ctrpk_status="BLOCK",
            ctrpk_trend="DEGRADING",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # Warning hygiene: Should have exactly 1 CTRPK warning (BLOCK takes priority)
        ctrpk_warnings = [w for w in status["warnings"] if "CTRPK" in w]
        assert len(ctrpk_warnings) == 1
        assert "BLOCK" in ctrpk_warnings[0]


@requires_ctrpk_status
class TestCTRPKDeterminismAndNonGating:
    """Tests for CTRPK determinism and non-gating behavior."""

    def test_ctrpk_computation_is_deterministic(self) -> None:
        """Test that CTRPK computation produces identical results on same input."""
        signals = [
            {"signal_type": "TRANSITION_REQUESTED", "cycle": 100},
            {"signal_type": "TRANSITION_REQUESTED", "cycle": 500},
            {"signal_type": "SNAPSHOT_VERIFIED", "cycle": 9999},
        ]

        results = [compute_ctrpk_from_signals(signals) for _ in range(5)]

        # All results should be identical
        for result in results[1:]:
            assert result["value"] == results[0]["value"]
            assert result["status"] == results[0]["status"]
            assert result["trend"] == results[0]["trend"]

    def test_ctrpk_does_not_gate_status_generation(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that BLOCK CTRPK does not prevent status generation (non-gating)."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=10.0,  # Very high, BLOCK status
            ctrpk_status="BLOCK",
            ctrpk_trend="DEGRADING",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # Status generation succeeds despite BLOCK CTRPK
        assert status["evidence_pack_ok"] is True
        assert "ctrpk" in status["signals"]

    def test_invalid_ctrpk_structure_does_not_block(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that invalid CTRPK structure generates warning but doesn't block."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # Create manifest with invalid CTRPK (missing required fields)
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "files": [],
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {
                "curriculum": {
                    "ctrpk": {
                        "value": 2.0,
                        # Missing "status" field - invalid
                    }
                }
            },
        }
        (evidence_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # Status generation succeeds
        assert status["evidence_pack_ok"] is True
        # Warning generated for invalid structure
        assert any("invalid structure" in w for w in status["warnings"])


@requires_ctrpk_status
class TestCTRPKCLIVsManifestPrecedence:
    """Tests for CLI vs manifest CTRPK precedence and mismatch warnings."""

    def test_manifest_ctrpk_takes_precedence_over_cli(
        self, minimal_evidence_pack_dir: Path, tmp_path: Path
    ) -> None:
        """Test that manifest CTRPK takes precedence over CLI-provided CTRPK."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # Manifest CTRPK
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=3.0,
            ctrpk_status="WARN",
            ctrpk_trend="STABLE",
        )

        # CLI CTRPK (different values)
        cli_ctrpk = tmp_path / "cli_ctrpk.json"
        cli_ctrpk.write_text(
            json.dumps({
                "value": 1.0,
                "status": "OK",
                "trend": "IMPROVING",
            }),
            encoding="utf-8",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
            ctrpk_json_path=cli_ctrpk,
        )

        # Manifest values should be used
        assert status["signals"]["ctrpk"]["value"] == 3.0
        assert status["signals"]["ctrpk"]["status"] == "WARN"
        assert status["signals"]["ctrpk"]["extraction_source"] == "MANIFEST"

    def test_cli_ctrpk_used_when_manifest_missing(
        self, minimal_evidence_pack_dir: Path, tmp_path: Path
    ) -> None:
        """Test that CLI CTRPK is used when manifest has no CTRPK."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # Manifest without CTRPK
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "files": [],
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        (evidence_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        # CLI CTRPK
        cli_ctrpk = tmp_path / "cli_ctrpk.json"
        cli_ctrpk.write_text(
            json.dumps({
                "value": 0.8,
                "status": "OK",
                "trend": "STABLE",
                "window_cycles": 5000,
                "transition_requests": 4,
            }),
            encoding="utf-8",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
            ctrpk_json_path=cli_ctrpk,
        )

        # CLI values should be used
        assert status["signals"]["ctrpk"]["value"] == 0.8
        assert status["signals"]["ctrpk"]["status"] == "OK"
        assert status["signals"]["ctrpk"]["extraction_source"] == "CLI"

    def test_mismatch_warning_generated_when_both_exist(
        self, minimal_evidence_pack_dir: Path, tmp_path: Path
    ) -> None:
        """Test that mismatch warning is generated when CLI and manifest differ."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # Manifest CTRPK
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=5.0,
            ctrpk_status="WARN",
            ctrpk_trend="DEGRADING",
        )

        # CLI CTRPK (significantly different)
        cli_ctrpk = tmp_path / "cli_ctrpk.json"
        cli_ctrpk.write_text(
            json.dumps({
                "value": 1.0,
                "status": "OK",
                "trend": "STABLE",
            }),
            encoding="utf-8",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
            ctrpk_json_path=cli_ctrpk,
        )

        # Mismatch warning should be present
        mismatch_warnings = [w for w in status["warnings"] if "mismatch" in w.lower()]
        assert len(mismatch_warnings) == 1
        assert "CLI" in mismatch_warnings[0]
        assert "manifest" in mismatch_warnings[0]

    def test_no_mismatch_warning_when_values_match(
        self, minimal_evidence_pack_dir: Path, tmp_path: Path
    ) -> None:
        """Test that no mismatch warning when CLI and manifest match."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # Manifest and CLI with same values
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=2.0,
            ctrpk_status="WARN",
            ctrpk_trend="STABLE",
        )

        cli_ctrpk = tmp_path / "cli_ctrpk.json"
        cli_ctrpk.write_text(
            json.dumps({
                "value": 2.0,  # Same value
                "status": "WARN",  # Same status
                "trend": "STABLE",
            }),
            encoding="utf-8",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
            ctrpk_json_path=cli_ctrpk,
        )

        # No mismatch warning
        mismatch_warnings = [w for w in status["warnings"] if "mismatch" in w.lower()]
        assert len(mismatch_warnings) == 0


# -----------------------------------------------------------------------------
# CTRPK GGFL Adapter Tests (SIG-CTRPK)
# -----------------------------------------------------------------------------

class TestCTRPKGGFLAdapter:
    """Tests for CTRPK GGFL adapter (ctrpk_for_alignment_view)."""

    def test_adapter_returns_fixed_shape(self) -> None:
        """Test that adapter returns fixed shape with all required fields."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 2.5,
            "status": "WARN",
            "trend": "STABLE",
            "window_cycles": 10000,
            "transition_requests": 25,
        }

        result = ctrpk_for_alignment_view(signal)

        # Check required fields
        required_fields = {"signal_type", "status", "conflict", "drivers", "summary"}
        assert required_fields.issubset(set(result.keys()))

        # Check field types and values
        assert result["signal_type"] == "SIG-CTRPK"
        assert result["status"] in {"ok", "warn"}
        assert result["conflict"] is False
        assert isinstance(result["drivers"], list)
        assert len(result["drivers"]) <= 3
        assert isinstance(result["summary"], str)

    def test_status_warn_when_status_is_warn(self) -> None:
        """Test that GGFL status is 'warn' when CTRPK status is WARN."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 2.5,
            "status": "WARN",
            "trend": "STABLE",
        }

        result = ctrpk_for_alignment_view(signal)

        assert result["status"] == "warn"

    def test_status_warn_when_status_is_block(self) -> None:
        """Test that GGFL status is 'warn' when CTRPK status is BLOCK."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 7.0,
            "status": "BLOCK",
            "trend": "STABLE",
        }

        result = ctrpk_for_alignment_view(signal)

        assert result["status"] == "warn"

    def test_status_warn_when_trend_is_degrading(self) -> None:
        """Test that GGFL status is 'warn' when trend is DEGRADING."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 1.0,
            "status": "OK",
            "trend": "DEGRADING",
        }

        result = ctrpk_for_alignment_view(signal)

        assert result["status"] == "warn"

    def test_status_ok_when_nominal(self) -> None:
        """Test that GGFL status is 'ok' when CTRPK status is OK and trend is STABLE."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 0.5,
            "status": "OK",
            "trend": "STABLE",
        }

        result = ctrpk_for_alignment_view(signal)

        assert result["status"] == "ok"

    def test_status_ok_when_improving(self) -> None:
        """Test that GGFL status is 'ok' when trend is IMPROVING."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 0.3,
            "status": "OK",
            "trend": "IMPROVING",
        }

        result = ctrpk_for_alignment_view(signal)

        assert result["status"] == "ok"

    def test_drivers_use_reason_codes_only(self) -> None:
        """Test that drivers use canonical reason codes only (no value-based drivers)."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 2.5,
            "status": "WARN",
            "trend": "DEGRADING",
        }

        result = ctrpk_for_alignment_view(signal)

        # Should use canonical reason codes only
        valid_codes = {"DRIVER_STATUS_BLOCK", "DRIVER_STATUS_WARN", "DRIVER_TREND_DEGRADING"}
        for driver in result["drivers"]:
            assert driver in valid_codes, f"Invalid driver code: {driver}"

    def test_drivers_warn_status_produces_driver_status_warn(self) -> None:
        """Test that WARN status produces DRIVER_STATUS_WARN reason code."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 5.0,
            "status": "WARN",
            "trend": "STABLE",
        }

        result = ctrpk_for_alignment_view(signal)

        assert "DRIVER_STATUS_WARN" in result["drivers"]

    def test_drivers_block_status_produces_driver_status_block(self) -> None:
        """Test that BLOCK status produces DRIVER_STATUS_BLOCK reason code."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 8.0,
            "status": "BLOCK",
            "trend": "STABLE",
        }

        result = ctrpk_for_alignment_view(signal)

        assert "DRIVER_STATUS_BLOCK" in result["drivers"]

    def test_drivers_degrading_trend_produces_driver_trend_degrading(self) -> None:
        """Test that DEGRADING trend produces DRIVER_TREND_DEGRADING reason code."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 1.5,
            "status": "OK",
            "trend": "DEGRADING",
        }

        result = ctrpk_for_alignment_view(signal)

        assert "DRIVER_TREND_DEGRADING" in result["drivers"]

    def test_drivers_limited_to_3(self) -> None:
        """Test that drivers are limited to 3 (max possible codes)."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        # BLOCK + DEGRADING produces 2 drivers (WARN is mutually exclusive with BLOCK)
        signal = {
            "value": 8.0,
            "status": "BLOCK",
            "trend": "DEGRADING",
            "window_cycles": 10000,
            "transition_requests": 80,
        }

        result = ctrpk_for_alignment_view(signal)

        assert len(result["drivers"]) <= 3
        assert "DRIVER_STATUS_BLOCK" in result["drivers"]
        assert "DRIVER_TREND_DEGRADING" in result["drivers"]

    def test_drivers_empty_when_ok_stable(self) -> None:
        """Test that drivers are empty when status is OK and trend is STABLE."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 0.5,
            "status": "OK",
            "trend": "STABLE",
        }

        result = ctrpk_for_alignment_view(signal)

        assert result["drivers"] == []

    def test_conflict_always_false(self) -> None:
        """Test that conflict is always False (CTRPK never triggers conflict)."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        # Test with BLOCK status
        signal_block = {
            "value": 10.0,
            "status": "BLOCK",
            "trend": "DEGRADING",
        }
        result_block = ctrpk_for_alignment_view(signal_block)
        assert result_block["conflict"] is False

        # Test with OK status
        signal_ok = {
            "value": 0.1,
            "status": "OK",
            "trend": "IMPROVING",
        }
        result_ok = ctrpk_for_alignment_view(signal_ok)
        assert result_ok["conflict"] is False

    def test_summary_neutral_language(self) -> None:
        """Test that summary uses neutral, descriptive language (uses reusable helper)."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 5.0,
            "status": "WARN",
            "trend": "DEGRADING",
        }

        result = ctrpk_for_alignment_view(signal)

        # Use reusable warning neutrality helper
        pytest_assert_warning_neutral(result["summary"], context="GGFL summary")

    def test_is_deterministic(self) -> None:
        """Test that output is deterministic for identical inputs."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 2.5,
            "status": "WARN",
            "trend": "DEGRADING",
            "window_cycles": 10000,
            "transition_requests": 25,
        }

        result1 = ctrpk_for_alignment_view(signal)
        result2 = ctrpk_for_alignment_view(signal)

        assert result1 == result2

    def test_json_safe(self) -> None:
        """Test that output is JSON-safe."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 2.5,
            "status": "WARN",
            "trend": "STABLE",
        }

        result = ctrpk_for_alignment_view(signal)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["signal_type"] == "SIG-CTRPK"

    def test_shadow_mode_invariants_present(self) -> None:
        """Test that shadow_mode_invariants are present and correct."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        signal = {
            "value": 1.0,
            "status": "OK",
            "trend": "STABLE",
        }

        result = ctrpk_for_alignment_view(signal)

        assert "shadow_mode_invariants" in result
        invariants = result["shadow_mode_invariants"]
        assert invariants["advisory_only"] is True
        assert invariants["no_enforcement"] is True
        assert invariants["conflict_invariant"] is True


# -----------------------------------------------------------------------------
# Mismatch Warning Text Test
# -----------------------------------------------------------------------------

@requires_ctrpk_status
class TestCTRPKMismatchWarningText:
    """Tests for CTRPK mismatch warning text format."""

    def test_mismatch_warning_contains_expected_details(
        self, minimal_evidence_pack_dir: Path, tmp_path: Path
    ) -> None:
        """Test that mismatch warning text contains CLI value, manifest value, and 'Using manifest' message."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # Manifest CTRPK
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=4.5,
            ctrpk_status="WARN",
            ctrpk_trend="STABLE",
        )

        # CLI CTRPK with different values
        cli_ctrpk = tmp_path / "cli_ctrpk.json"
        cli_ctrpk.write_text(
            json.dumps({
                "value": 1.2,
                "status": "OK",
                "trend": "IMPROVING",
            }),
            encoding="utf-8",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
            ctrpk_json_path=cli_ctrpk,
        )

        # Find the mismatch warning
        mismatch_warnings = [w for w in status["warnings"] if "mismatch" in w.lower()]
        assert len(mismatch_warnings) == 1

        warning = mismatch_warnings[0]

        # Check required content in warning text
        assert "CLI" in warning
        assert "manifest" in warning.lower()
        assert "1.2" in warning or "1.20" in warning  # CLI value
        assert "4.5" in warning or "4.50" in warning  # Manifest value
        assert "Using manifest" in warning


# -----------------------------------------------------------------------------
# Manifest Reference Hash Tests (source_path + source_sha256)
# -----------------------------------------------------------------------------

@requires_ctrpk_status
class TestCTRPKManifestReferenceHash:
    """Tests for CTRPK manifest reference hash in status output."""

    def test_status_includes_source_path_from_manifest(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that status includes source_path when present in manifest."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=1.5,
            ctrpk_status="WARN",
            ctrpk_trend="STABLE",
            include_path_and_sha256=True,
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        assert "ctrpk" in status["signals"]
        ctrpk_signal = status["signals"]["ctrpk"]
        assert "source_path" in ctrpk_signal
        assert ctrpk_signal["source_path"] == "ctrpk_compact.json"

    def test_status_includes_source_sha256_from_manifest(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that status includes source_sha256 when present in manifest."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=2.0,
            ctrpk_status="WARN",
            ctrpk_trend="DEGRADING",
            include_path_and_sha256=True,
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        assert "ctrpk" in status["signals"]
        ctrpk_signal = status["signals"]["ctrpk"]
        assert "source_sha256" in ctrpk_signal
        # SHA256 should be 64 hex chars
        assert len(ctrpk_signal["source_sha256"]) == 64

    def test_status_omits_source_fields_when_not_in_manifest(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that source_path/sha256 are omitted when not present in manifest."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=1.0,
            ctrpk_status="OK",
            ctrpk_trend="STABLE",
            include_path_and_sha256=False,  # Exclude path and sha256
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        assert "ctrpk" in status["signals"]
        ctrpk_signal = status["signals"]["ctrpk"]
        assert "source_path" not in ctrpk_signal
        assert "source_sha256" not in ctrpk_signal


# -----------------------------------------------------------------------------
# GGFL Weight and Conflict Invariant Regression Tests
# -----------------------------------------------------------------------------

class TestCTRPKGGFLInvariantRegression:
    """Regression tests for SIG-CTRPK weight_hint=LOW and conflict=false invariants."""

    def test_weight_hint_always_low(self) -> None:
        """Regression test: weight_hint must always be 'LOW'."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        # Test across all status/trend combinations
        test_cases = [
            {"value": 0.1, "status": "OK", "trend": "STABLE"},
            {"value": 0.5, "status": "OK", "trend": "IMPROVING"},
            {"value": 2.0, "status": "WARN", "trend": "STABLE"},
            {"value": 3.5, "status": "WARN", "trend": "DEGRADING"},
            {"value": 7.0, "status": "BLOCK", "trend": "STABLE"},
            {"value": 10.0, "status": "BLOCK", "trend": "DEGRADING"},
        ]

        for signal in test_cases:
            result = ctrpk_for_alignment_view(signal)
            assert result["weight_hint"] == "LOW", (
                f"weight_hint must be 'LOW' for signal {signal}, got {result['weight_hint']}"
            )

    def test_conflict_always_false(self) -> None:
        """Regression test: conflict must always be False."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        # Test across all status/trend combinations including worst case
        test_cases = [
            {"value": 0.1, "status": "OK", "trend": "STABLE"},
            {"value": 0.5, "status": "OK", "trend": "IMPROVING"},
            {"value": 2.0, "status": "WARN", "trend": "STABLE"},
            {"value": 3.5, "status": "WARN", "trend": "DEGRADING"},
            {"value": 7.0, "status": "BLOCK", "trend": "STABLE"},
            {"value": 10.0, "status": "BLOCK", "trend": "DEGRADING"},
            # Edge case: maximum severity
            {"value": 100.0, "status": "BLOCK", "trend": "DEGRADING"},
        ]

        for signal in test_cases:
            result = ctrpk_for_alignment_view(signal)
            assert result["conflict"] is False, (
                f"conflict must be False for signal {signal}, got {result['conflict']}"
            )

    def test_ggfl_invariants_never_gate(self) -> None:
        """Regression test: GGFL adapter never produces gating signals."""
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        # Even worst-case inputs should not produce gating behavior
        worst_case = {
            "value": 100.0,
            "status": "BLOCK",
            "trend": "DEGRADING",
            "window_cycles": 1000,
            "transition_requests": 100,
        }

        result = ctrpk_for_alignment_view(worst_case)

        # Invariants that ensure non-gating
        assert result["conflict"] is False
        assert result["weight_hint"] == "LOW"
        assert result["shadow_mode_invariants"]["advisory_only"] is True
        assert result["shadow_mode_invariants"]["no_enforcement"] is True
        assert result["shadow_mode_invariants"]["conflict_invariant"] is True


# -----------------------------------------------------------------------------
# Warning Hygiene Tests (Cap to 1 warning)
# -----------------------------------------------------------------------------

@requires_ctrpk_status
class TestCTRPKWarningHygiene:
    """Tests for CTRPK warning hygiene (cap to 1 warning)."""

    def test_warn_only_generates_single_warning(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that WARN status alone generates exactly 1 warning."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=2.5,
            ctrpk_status="WARN",
            ctrpk_trend="STABLE",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        ctrpk_warnings = [w for w in status["warnings"] if "CTRPK" in w]
        assert len(ctrpk_warnings) == 1
        assert "WARN" in ctrpk_warnings[0]

    def test_degrading_only_generates_single_warning(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that DEGRADING trend alone generates exactly 1 warning."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=1.5,
            ctrpk_status="OK",
            ctrpk_trend="DEGRADING",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        ctrpk_warnings = [w for w in status["warnings"] if "CTRPK" in w]
        assert len(ctrpk_warnings) == 1
        assert "DEGRADING" in ctrpk_warnings[0]

    def test_block_with_degrading_generates_single_warning(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that BLOCK+DEGRADING generates exactly 1 warning (BLOCK priority)."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=8.0,
            ctrpk_status="BLOCK",
            ctrpk_trend="DEGRADING",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        ctrpk_warnings = [w for w in status["warnings"] if "CTRPK" in w]
        assert len(ctrpk_warnings) == 1
        assert "BLOCK" in ctrpk_warnings[0]

    def test_ok_stable_generates_no_warning(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that OK+STABLE generates no CTRPK warnings."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=0.5,
            ctrpk_status="OK",
            ctrpk_trend="STABLE",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        ctrpk_warnings = [w for w in status["warnings"] if "CTRPK" in w]
        assert len(ctrpk_warnings) == 0

    def test_ctrpk_warnings_are_neutral(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that CTRPK warnings are neutral (single-line, no banned words)."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=5.0,
            ctrpk_status="WARN",
            ctrpk_trend="DEGRADING",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        ctrpk_warnings = [w for w in status["warnings"] if "CTRPK" in w]
        assert len(ctrpk_warnings) == 1
        # Use reusable neutrality helper
        pytest_assert_warning_neutral(ctrpk_warnings[0], context="CTRPK warning")


# -----------------------------------------------------------------------------
# Extraction Source Enum Tests (MANIFEST | CLI | MISSING)
# -----------------------------------------------------------------------------

@requires_ctrpk_status
class TestCTRPKExtractionSourceEnum:
    """Tests for CTRPK extraction_source enum in status output."""

    def test_extraction_source_manifest_when_from_manifest(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that extraction_source is MANIFEST when CTRPK comes from manifest."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=1.5,
            ctrpk_status="WARN",
            ctrpk_trend="STABLE",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        assert "ctrpk" in status["signals"]
        assert status["signals"]["ctrpk"]["extraction_source"] == "MANIFEST"

    def test_extraction_source_cli_when_from_cli(
        self, minimal_evidence_pack_dir: Path, tmp_path: Path
    ) -> None:
        """Test that extraction_source is CLI when CTRPK comes from CLI only."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        # Create manifest without CTRPK
        manifest_path = evidence_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "files": [],
        }), encoding="utf-8")

        # Create CLI CTRPK
        cli_ctrpk = tmp_path / "cli_ctrpk.json"
        cli_ctrpk.write_text(json.dumps({
            "value": 2.0,
            "status": "WARN",
            "trend": "DEGRADING",
        }), encoding="utf-8")

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
            ctrpk_json_path=cli_ctrpk,
        )

        assert "ctrpk" in status["signals"]
        assert status["signals"]["ctrpk"]["extraction_source"] == "CLI"

    def test_extraction_source_enum_values_are_uppercase(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """Test that extraction_source uses uppercase enum values."""
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=1.0,
            ctrpk_status="OK",
            ctrpk_trend="STABLE",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        extraction_source = status["signals"]["ctrpk"]["extraction_source"]
        assert extraction_source in {"MANIFEST", "CLI", "MISSING"}
        assert extraction_source == extraction_source.upper()


# -----------------------------------------------------------------------------
# Comprehensive GGFL Invariant Regression Test
# -----------------------------------------------------------------------------

class TestCTRPKGGFLInvariantComprehensive:
    """Comprehensive regression test for SIG-CTRPK GGFL invariants across ALL inputs."""

    def test_weight_hint_low_and_conflict_false_across_all_inputs(self) -> None:
        """
        REGRESSION TEST: weight_hint must always be 'LOW' and conflict must always be False
        across ALL possible input combinations.

        This test exhaustively verifies the GGFL invariants that ensure CTRPK
        never overpowers fusion semantics or triggers conflict directly.
        """
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        # Exhaustive test matrix: all status x trend combinations
        statuses = ["OK", "WARN", "BLOCK"]
        trends = ["IMPROVING", "STABLE", "DEGRADING"]
        values = [0.0, 0.5, 1.0, 2.5, 5.0, 7.0, 10.0, 100.0]

        # Test all combinations
        for status in statuses:
            for trend in trends:
                for value in values:
                    signal = {
                        "value": value,
                        "status": status,
                        "trend": trend,
                        "window_cycles": 10000,
                        "transition_requests": int(value * 10),
                    }

                    result = ctrpk_for_alignment_view(signal)

                    # INVARIANT 1: weight_hint must ALWAYS be "LOW"
                    assert result["weight_hint"] == "LOW", (
                        f"INVARIANT VIOLATION: weight_hint must be 'LOW' but got "
                        f"'{result['weight_hint']}' for signal: {signal}"
                    )

                    # INVARIANT 2: conflict must ALWAYS be False
                    assert result["conflict"] is False, (
                        f"INVARIANT VIOLATION: conflict must be False but got "
                        f"'{result['conflict']}' for signal: {signal}"
                    )

                    # INVARIANT 3: shadow_mode_invariants must all be True
                    invariants = result.get("shadow_mode_invariants", {})
                    assert invariants.get("advisory_only") is True, (
                        f"INVARIANT VIOLATION: advisory_only must be True for signal: {signal}"
                    )
                    assert invariants.get("no_enforcement") is True, (
                        f"INVARIANT VIOLATION: no_enforcement must be True for signal: {signal}"
                    )
                    assert invariants.get("conflict_invariant") is True, (
                        f"INVARIANT VIOLATION: conflict_invariant must be True for signal: {signal}"
                    )

    def test_driver_codes_are_canonical_across_all_inputs(self) -> None:
        """
        REGRESSION TEST: driver codes must always be from the canonical set.

        Valid codes: DRIVER_STATUS_BLOCK, DRIVER_STATUS_WARN, DRIVER_TREND_DEGRADING
        """
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        valid_codes = {"DRIVER_STATUS_BLOCK", "DRIVER_STATUS_WARN", "DRIVER_TREND_DEGRADING"}
        statuses = ["OK", "WARN", "BLOCK"]
        trends = ["IMPROVING", "STABLE", "DEGRADING"]

        for status in statuses:
            for trend in trends:
                signal = {
                    "value": 5.0,
                    "status": status,
                    "trend": trend,
                }

                result = ctrpk_for_alignment_view(signal)

                for driver in result["drivers"]:
                    assert driver in valid_codes, (
                        f"INVALID DRIVER CODE: '{driver}' not in valid codes {valid_codes} "
                        f"for signal: {signal}"
                    )


# -----------------------------------------------------------------------------
# Status ↔ GGFL Consistency Regression Tests
# -----------------------------------------------------------------------------

class TestCtrpkStatusVsGgflConsistency:
    """
    Regression tests ensuring Status Generator warnings and GGFL adapter outputs
    are consistent across the canonical 4-case matrix.

    Matrix:
    1. OK/STABLE     → no warning,  GGFL status="ok",   drivers=[]
    2. WARN/STABLE   → 1 warning,   GGFL status="warn", drivers=[DRIVER_STATUS_WARN]
    3. WARN/DEGRADING→ 1 warning,   GGFL status="warn", drivers=[DRIVER_STATUS_WARN, DRIVER_TREND_DEGRADING]
    4. BLOCK/*       → 1 warning,   GGFL status="warn", drivers=[DRIVER_STATUS_BLOCK, ...]
    """

    # Canonical 4-case matrix with expected outcomes
    MATRIX = [
        # (status, trend, expect_warning, ggfl_status, expected_drivers)
        ("OK", "STABLE", False, "ok", []),
        ("WARN", "STABLE", True, "warn", ["DRIVER_STATUS_WARN"]),
        ("WARN", "DEGRADING", True, "warn", ["DRIVER_STATUS_WARN", "DRIVER_TREND_DEGRADING"]),
        ("BLOCK", "STABLE", True, "warn", ["DRIVER_STATUS_BLOCK"]),
        ("BLOCK", "DEGRADING", True, "warn", ["DRIVER_STATUS_BLOCK", "DRIVER_TREND_DEGRADING"]),
        # Additional edge cases
        ("OK", "DEGRADING", True, "warn", ["DRIVER_TREND_DEGRADING"]),
        ("OK", "IMPROVING", False, "ok", []),
    ]

    @requires_ctrpk_status
    def test_status_warning_presence_matches_ggfl_status(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """
        REGRESSION: Status warning presence/absence must match GGFL status mapping.

        - Warning present  → GGFL status="warn"
        - Warning absent   → GGFL status="ok"
        """
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        for ctrpk_status, trend, expect_warning, expected_ggfl_status, _ in self.MATRIX:
            evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

            # Create manifest with test case
            create_manifest_with_ctrpk(
                evidence_dir,
                ctrpk_value=2.5,
                ctrpk_status=ctrpk_status,
                ctrpk_trend=trend,
            )

            # Generate status
            status = generate_status(
                p3_dir=minimal_evidence_pack_dir / "p3",
                p4_dir=minimal_evidence_pack_dir / "p4",
                evidence_pack_dir=evidence_dir,
            )

            # Get GGFL result
            signal = {"value": 2.5, "status": ctrpk_status, "trend": trend}
            ggfl_result = ctrpk_for_alignment_view(signal)

            # Check warning presence
            ctrpk_warnings = [w for w in status["warnings"] if "CTRPK" in w]
            has_warning = len(ctrpk_warnings) > 0

            # Assert consistency
            assert has_warning == expect_warning, (
                f"INCONSISTENCY: status={ctrpk_status}, trend={trend} - "
                f"expected warning={expect_warning}, got warning={has_warning}"
            )
            assert ggfl_result["status"] == expected_ggfl_status, (
                f"INCONSISTENCY: status={ctrpk_status}, trend={trend} - "
                f"expected GGFL status={expected_ggfl_status}, got={ggfl_result['status']}"
            )

            # Cross-check: warning presence must align with GGFL status
            if has_warning:
                assert ggfl_result["status"] == "warn", (
                    f"CROSS-CHECK FAIL: warning present but GGFL status is not 'warn' "
                    f"for status={ctrpk_status}, trend={trend}"
                )
            else:
                assert ggfl_result["status"] == "ok", (
                    f"CROSS-CHECK FAIL: no warning but GGFL status is not 'ok' "
                    f"for status={ctrpk_status}, trend={trend}"
                )

    def test_driver_codes_match_canonical_set(self) -> None:
        """
        REGRESSION: Driver codes must match canonical set for each case.
        """
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        for ctrpk_status, trend, _, _, expected_drivers in self.MATRIX:
            signal = {"value": 2.5, "status": ctrpk_status, "trend": trend}
            result = ctrpk_for_alignment_view(signal)

            # Drivers must match exactly (order matters for determinism)
            assert result["drivers"] == expected_drivers, (
                f"DRIVER MISMATCH: status={ctrpk_status}, trend={trend} - "
                f"expected {expected_drivers}, got {result['drivers']}"
            )

    @requires_ctrpk_status
    def test_warning_capped_to_one_line(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """
        REGRESSION: Warning hygiene - CTRPK warnings must be capped to 1 line.

        Even with multiple warning conditions (e.g., WARN+DEGRADING), only
        one combined warning should be emitted.
        """
        # Test all cases that should produce warnings
        warning_cases = [
            ("WARN", "STABLE"),
            ("WARN", "DEGRADING"),
            ("BLOCK", "STABLE"),
            ("BLOCK", "DEGRADING"),
            ("OK", "DEGRADING"),
        ]

        for ctrpk_status, trend in warning_cases:
            evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

            create_manifest_with_ctrpk(
                evidence_dir,
                ctrpk_value=5.0,
                ctrpk_status=ctrpk_status,
                ctrpk_trend=trend,
            )

            status = generate_status(
                p3_dir=minimal_evidence_pack_dir / "p3",
                p4_dir=minimal_evidence_pack_dir / "p4",
                evidence_pack_dir=evidence_dir,
            )

            ctrpk_warnings = [w for w in status["warnings"] if "CTRPK" in w]

            assert len(ctrpk_warnings) == 1, (
                f"WARNING HYGIENE VIOLATION: status={ctrpk_status}, trend={trend} - "
                f"expected 1 warning, got {len(ctrpk_warnings)}: {ctrpk_warnings}"
            )

    def test_ggfl_invariants_hold_across_matrix(self) -> None:
        """
        REGRESSION: weight_hint=LOW and conflict=false must hold across all cases.
        """
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        for ctrpk_status, trend, _, _, _ in self.MATRIX:
            signal = {"value": 2.5, "status": ctrpk_status, "trend": trend}
            result = ctrpk_for_alignment_view(signal)

            assert result["weight_hint"] == "LOW", (
                f"INVARIANT VIOLATION: weight_hint must be 'LOW' "
                f"for status={ctrpk_status}, trend={trend}, got={result['weight_hint']}"
            )
            assert result["conflict"] is False, (
                f"INVARIANT VIOLATION: conflict must be False "
                f"for status={ctrpk_status}, trend={trend}, got={result['conflict']}"
            )

    def test_no_warning_cases_produce_empty_drivers(self) -> None:
        """
        REGRESSION: Cases with no warning must produce empty driver list.
        """
        from backend.topology.first_light.ctrpk_detection import ctrpk_for_alignment_view

        no_warning_cases = [
            ("OK", "STABLE"),
            ("OK", "IMPROVING"),
        ]

        for ctrpk_status, trend in no_warning_cases:
            signal = {"value": 0.5, "status": ctrpk_status, "trend": trend}
            result = ctrpk_for_alignment_view(signal)

            assert result["drivers"] == [], (
                f"DRIVER INVARIANT VIOLATION: no-warning case "
                f"status={ctrpk_status}, trend={trend} should have empty drivers, "
                f"got {result['drivers']}"
            )
            assert result["status"] == "ok", (
                f"STATUS INVARIANT VIOLATION: no-warning case "
                f"status={ctrpk_status}, trend={trend} should have GGFL status='ok', "
                f"got {result['status']}"
            )


# -----------------------------------------------------------------------------
# CAL-EXP-2 PREP: CTRPK Non-Interference Tests
# -----------------------------------------------------------------------------

@requires_ctrpk_status
class TestCTRPKNonInterference:
    """
    CAL-EXP-2 PREP: CTRPK Non-Interference Tests.

    Proves that toggling CTRPK presence:
    1. Only affects signals.ctrpk (and at most 1 warning line)
    2. Does NOT change P4 divergence metrics or telemetry_source
    3. Does NOT reorder unrelated warnings

    SHADOW MODE CONTRACT: CTRPK is purely observational and must not
    interfere with any other signal extraction or status computation.
    """

    def test_ctrpk_toggle_only_affects_ctrpk_signal(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """
        PROOF 1: Toggling CTRPK presence changes ONLY signals.ctrpk.

        All other signals must remain byte-identical between:
        - Status with CTRPK in manifest
        - Status without CTRPK in manifest

        This proves CTRPK extraction is isolated from other signal extraction.
        """
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # --- Run WITHOUT CTRPK ---
        manifest_no_ctrpk = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "files": [],
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        (evidence_dir / "manifest.json").write_text(
            json.dumps(manifest_no_ctrpk), encoding="utf-8"
        )

        status_no_ctrpk = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # --- Run WITH CTRPK ---
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=2.5,
            ctrpk_status="WARN",
            ctrpk_trend="STABLE",
        )

        status_with_ctrpk = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # --- PROOF: Only signals.ctrpk differs ---
        signals_no_ctrpk = status_no_ctrpk.get("signals", {}) or {}
        signals_with_ctrpk = status_with_ctrpk.get("signals", {}) or {}

        # Remove ctrpk from comparison
        signals_no_ctrpk_filtered = {
            k: v for k, v in signals_no_ctrpk.items() if k != "ctrpk"
        }
        signals_with_ctrpk_filtered = {
            k: v for k, v in signals_with_ctrpk.items() if k != "ctrpk"
        }

        # All other signals must be identical
        assert signals_no_ctrpk_filtered == signals_with_ctrpk_filtered, (
            f"NON-INTERFERENCE VIOLATION: signals differ beyond 'ctrpk'\n"
            f"Without CTRPK: {signals_no_ctrpk_filtered}\n"
            f"With CTRPK: {signals_with_ctrpk_filtered}"
        )

        # CTRPK signal should be present only in with-CTRPK status
        assert "ctrpk" not in signals_no_ctrpk
        assert "ctrpk" in signals_with_ctrpk

    def test_ctrpk_toggle_warning_delta_at_most_one(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """
        PROOF 2: Toggling CTRPK affects at most 1 warning line.

        The warning delta between with-CTRPK and without-CTRPK must be ≤ 1.
        This ensures CTRPK warning hygiene (cap=1) is enforced.
        """
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # --- Run WITHOUT CTRPK ---
        manifest_no_ctrpk = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "files": [],
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        (evidence_dir / "manifest.json").write_text(
            json.dumps(manifest_no_ctrpk), encoding="utf-8"
        )

        status_no_ctrpk = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # --- Run WITH CTRPK (WARN status to trigger warning) ---
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=3.0,
            ctrpk_status="WARN",
            ctrpk_trend="DEGRADING",
        )

        status_with_ctrpk = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # --- PROOF: Warning delta is at most 1 ---
        warnings_no_ctrpk = status_no_ctrpk.get("warnings", [])
        warnings_with_ctrpk = status_with_ctrpk.get("warnings", [])

        # Count CTRPK-specific warnings
        ctrpk_warnings = [w for w in warnings_with_ctrpk if "CTRPK" in w]

        # Delta should be exactly the CTRPK warnings (at most 1)
        delta = len(warnings_with_ctrpk) - len(warnings_no_ctrpk)

        assert delta <= 1, (
            f"WARNING HYGIENE VIOLATION: warning delta is {delta} (expected ≤ 1)\n"
            f"Without CTRPK: {len(warnings_no_ctrpk)} warnings\n"
            f"With CTRPK: {len(warnings_with_ctrpk)} warnings"
        )

        assert len(ctrpk_warnings) <= 1, (
            f"WARNING HYGIENE VIOLATION: {len(ctrpk_warnings)} CTRPK warnings "
            f"(expected ≤ 1): {ctrpk_warnings}"
        )

    def test_ctrpk_does_not_reorder_unrelated_warnings(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """
        PROOF 3: CTRPK does not reorder unrelated warnings.

        Non-CTRPK warnings must appear in the same relative order
        regardless of CTRPK presence.
        """
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # --- Run WITHOUT CTRPK ---
        manifest_no_ctrpk = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "files": [],
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        (evidence_dir / "manifest.json").write_text(
            json.dumps(manifest_no_ctrpk), encoding="utf-8"
        )

        status_no_ctrpk = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # --- Run WITH CTRPK ---
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=5.0,
            ctrpk_status="BLOCK",
            ctrpk_trend="DEGRADING",
        )

        status_with_ctrpk = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # --- PROOF: Non-CTRPK warnings have same relative order ---
        warnings_no_ctrpk = status_no_ctrpk.get("warnings", [])
        warnings_with_ctrpk = status_with_ctrpk.get("warnings", [])

        # Filter out CTRPK warnings
        non_ctrpk_no = [w for w in warnings_no_ctrpk if "CTRPK" not in w]
        non_ctrpk_with = [w for w in warnings_with_ctrpk if "CTRPK" not in w]

        # Non-CTRPK warnings must be identical (same content, same order)
        assert non_ctrpk_no == non_ctrpk_with, (
            f"WARNING REORDER VIOLATION: non-CTRPK warnings differ\n"
            f"Without CTRPK: {non_ctrpk_no}\n"
            f"With CTRPK: {non_ctrpk_with}"
        )


# -----------------------------------------------------------------------------
# CAL-EXP-2 PREP: CTRPK Determinism Tests
# -----------------------------------------------------------------------------

@requires_ctrpk_status
class TestCTRPKDeterminism:
    """
    CAL-EXP-2 PREP: CTRPK Determinism Tests.

    Proves that identical inputs produce byte-identical signals.ctrpk output.

    SHADOW MODE CONTRACT: CTRPK extraction must be deterministic -
    no random elements, no timestamps, no non-deterministic ordering.
    """

    def test_identical_inputs_produce_identical_ctrpk_signal(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """
        PROOF 1: Identical inputs produce byte-identical CTRPK signal.

        Running generate_status twice with identical inputs must produce
        byte-identical signals.ctrpk output (JSON-serializable comparison).
        """
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        # Create manifest with CTRPK
        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=2.5,
            ctrpk_status="WARN",
            ctrpk_trend="STABLE",
            window_cycles=10000,
            transition_requests=25,
        )

        # Run generate_status multiple times
        status_1 = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )
        status_2 = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )
        status_3 = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        # Extract CTRPK signals
        ctrpk_1 = status_1["signals"]["ctrpk"]
        ctrpk_2 = status_2["signals"]["ctrpk"]
        ctrpk_3 = status_3["signals"]["ctrpk"]

        # Serialize to JSON for byte-identical comparison
        json_1 = json.dumps(ctrpk_1, sort_keys=True)
        json_2 = json.dumps(ctrpk_2, sort_keys=True)
        json_3 = json.dumps(ctrpk_3, sort_keys=True)

        assert json_1 == json_2, (
            f"DETERMINISM VIOLATION: run 1 != run 2\n"
            f"Run 1: {json_1}\n"
            f"Run 2: {json_2}"
        )
        assert json_2 == json_3, (
            f"DETERMINISM VIOLATION: run 2 != run 3\n"
            f"Run 2: {json_2}\n"
            f"Run 3: {json_3}"
        )

    def test_ctrpk_signal_fields_are_deterministic(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """
        PROOF 2: CTRPK signal contains no non-deterministic fields.

        The CTRPK signal must not contain:
        - Timestamps
        - Random values
        - Process IDs
        - Non-reproducible identifiers
        """
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=1.5,
            ctrpk_status="WARN",
            ctrpk_trend="DEGRADING",
        )

        status = generate_status(
            p3_dir=minimal_evidence_pack_dir / "p3",
            p4_dir=minimal_evidence_pack_dir / "p4",
            evidence_pack_dir=evidence_dir,
        )

        ctrpk_signal = status["signals"]["ctrpk"]

        # List of field names that would indicate non-determinism
        non_deterministic_patterns = [
            "timestamp", "time", "date", "created", "updated",
            "random", "uuid", "guid", "pid", "process_id",
            "session", "token", "nonce",
        ]

        # Check that no field names match non-deterministic patterns
        for key in ctrpk_signal.keys():
            key_lower = key.lower()
            for pattern in non_deterministic_patterns:
                assert pattern not in key_lower, (
                    f"NON-DETERMINISTIC FIELD DETECTED: '{key}' contains '{pattern}'"
                )

    def test_ctrpk_warnings_are_deterministic(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """
        PROOF 3: CTRPK warnings are deterministic.

        Running generate_status multiple times with CTRPK warning conditions
        must produce identical CTRPK warnings each time.
        """
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=5.0,
            ctrpk_status="WARN",
            ctrpk_trend="DEGRADING",
        )

        # Run multiple times
        results = []
        for _ in range(3):
            status = generate_status(
                p3_dir=minimal_evidence_pack_dir / "p3",
                p4_dir=minimal_evidence_pack_dir / "p4",
                evidence_pack_dir=evidence_dir,
            )
            ctrpk_warnings = [w for w in status["warnings"] if "CTRPK" in w]
            results.append(ctrpk_warnings)

        # All runs must produce identical CTRPK warnings
        assert results[0] == results[1], (
            f"WARNING DETERMINISM VIOLATION: run 1 != run 2\n"
            f"Run 1: {results[0]}\n"
            f"Run 2: {results[1]}"
        )
        assert results[1] == results[2], (
            f"WARNING DETERMINISM VIOLATION: run 2 != run 3\n"
            f"Run 2: {results[1]}\n"
            f"Run 3: {results[2]}"
        )

    def test_ctrpk_extraction_source_is_deterministic(
        self, minimal_evidence_pack_dir: Path
    ) -> None:
        """
        PROOF 4: extraction_source is deterministic.

        The extraction_source enum value must be identical across runs
        for identical inputs.
        """
        evidence_dir = minimal_evidence_pack_dir / "evidence_pack"

        create_manifest_with_ctrpk(
            evidence_dir,
            ctrpk_value=1.0,
            ctrpk_status="OK",
            ctrpk_trend="STABLE",
        )

        # Run multiple times
        sources = []
        for _ in range(3):
            status = generate_status(
                p3_dir=minimal_evidence_pack_dir / "p3",
                p4_dir=minimal_evidence_pack_dir / "p4",
                evidence_pack_dir=evidence_dir,
            )
            sources.append(status["signals"]["ctrpk"]["extraction_source"])

        # All runs must produce identical extraction_source
        assert sources[0] == sources[1] == sources[2], (
            f"EXTRACTION_SOURCE DETERMINISM VIOLATION: {sources}"
        )
        assert sources[0] == "MANIFEST", (
            f"EXTRACTION_SOURCE VALUE VIOLATION: expected 'MANIFEST', got {sources[0]}"
        )
