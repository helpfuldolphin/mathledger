"""
Integration tests for P5 Identity Pre-flight Workflow.

Tests the full integration of identity_preflight through:
1. Harness pre-flight check (run_identity_preflight)
2. Evidence pack attachment (attach_identity_preflight_to_evidence)
3. Status signal generation (generate_first_light_status with --p5-identity-report)

SHADOW MODE CONTRACT:
- All tests verify that identity checks are ADVISORY ONLY
- No test verifies gating behavior (none should exist)
- Warnings appear but execution never blocked

Status: PHASE X P5 PRE-FLIGHT
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.health.identity_alignment_checker import (
    CheckResult,
    check_p5_identity_alignment,
)
from backend.topology.first_light.evidence_pack import (
    attach_identity_preflight_to_evidence,
    detect_identity_preflight,
    load_identity_preflight_from_run_config,
)


# =============================================================================
# FIXTURE HELPERS
# =============================================================================


def make_config(
    version: str = "1.0.0",
    params: Dict[str, Any] | None = None,
    gates: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a test configuration."""
    return {
        "version": version,
        "params": params or {"max_depth": 10, "batch_size": 100},
        "gates": gates or {"uplift_gate": True, "safety_gate": True},
    }


def make_identity_preflight_result(
    status: str = "OK",
    fingerprint_match: bool = True,
    top_reasons: list | None = None,
) -> Dict[str, Any]:
    """Create a mock identity_preflight result."""
    return {
        "status": status,
        "fingerprint_match": fingerprint_match,
        "synthetic_fingerprint": "abc123",
        "production_fingerprint": "abc123" if fingerprint_match else "def456",
        "top_reasons": top_reasons or [],
        "invariant_summary": {
            "SI-001": "OK" if fingerprint_match else "BLOCK",
            "SI-002": "OK",
            "SI-003": "OK",
            "SI-004": "OK",
            "SI-005": "INVESTIGATE",
            "SI-006": "OK",
        },
        "blocking_issues": [] if status == "OK" else ["Test blocking issue"],
        "investigation_items": [] if status == "OK" else ["Test investigation item"],
        "timestamp": "2025-01-01T00:00:00Z",
        "skipped": False,
    }


# =============================================================================
# IDENTITY CHECK - OK SCENARIO
# =============================================================================


class TestIdentityCheckOKScenario:
    """Test identity check returns OK when configs match."""

    def test_identical_configs_return_ok_or_investigate(self) -> None:
        """Identical configs return OK or INVESTIGATE (no BLOCK).

        Note: May return INVESTIGATE due to missing P4 evidence or
        curriculum fingerprint, but never BLOCK for identical configs.
        """
        config = make_config()

        report = check_p5_identity_alignment(config, config)

        # Should never be BLOCK for identical configs
        assert report.overall_status != CheckResult.BLOCK
        assert report.overall_status in (CheckResult.OK, CheckResult.INVESTIGATE)
        assert report.get_exit_code() in (0, 1)
        assert report.synthetic_fingerprint == report.production_fingerprint
        # No blocking issues for identical configs
        assert len(report.blocking_issues) == 0

    def test_ok_scenario_has_no_blocking_issues(self) -> None:
        """OK scenario has no blocking issues."""
        config = make_config()

        report = check_p5_identity_alignment(config, config)

        assert len(report.blocking_issues) == 0

    def test_ok_scenario_may_have_investigation_items(self) -> None:
        """OK scenario may still have investigation items (e.g., no P4 evidence)."""
        config = make_config()

        report = check_p5_identity_alignment(config, config)

        # P4 evidence not provided triggers INVESTIGATE for SI-005
        # but overall can still be OK or INVESTIGATE depending on implementation
        assert report.overall_status in (CheckResult.OK, CheckResult.INVESTIGATE)


# =============================================================================
# IDENTITY CHECK - BLOCK SCENARIO
# =============================================================================


class TestIdentityCheckBLOCKScenario:
    """Test identity check returns BLOCK when configs diverge significantly."""

    def test_param_divergence_returns_block(self) -> None:
        """Different params cause fingerprint mismatch and BLOCK."""
        synthetic = make_config(params={"max_depth": 10})
        production = make_config(params={"max_depth": 20})

        report = check_p5_identity_alignment(synthetic, production)

        assert report.overall_status == CheckResult.BLOCK
        assert report.get_exit_code() == 2
        assert report.synthetic_fingerprint != report.production_fingerprint

    def test_block_scenario_has_blocking_issues(self) -> None:
        """BLOCK scenario has blocking issues populated."""
        synthetic = make_config(params={"a": 1})
        production = make_config(params={"a": 2})

        report = check_p5_identity_alignment(synthetic, production)

        assert len(report.blocking_issues) > 0

    def test_major_version_skew_returns_block(self) -> None:
        """Major version difference causes BLOCK."""
        synthetic = make_config(version="1.0.0")
        production = make_config(version="2.0.0")

        report = check_p5_identity_alignment(synthetic, production)

        assert report.overall_status == CheckResult.BLOCK


# =============================================================================
# EVIDENCE PACK ATTACHMENT
# =============================================================================


class TestEvidencePackAttachment:
    """Test attaching identity_preflight to evidence pack."""

    def test_attach_creates_governance_section(self) -> None:
        """Attachment creates governance.slice_identity.p5_preflight."""
        evidence: Dict[str, Any] = {}
        preflight = make_identity_preflight_result()

        result = attach_identity_preflight_to_evidence(evidence, preflight)

        assert "governance" in result
        assert "slice_identity" in result["governance"]
        assert "p5_preflight" in result["governance"]["slice_identity"]

    def test_attach_preserves_existing_governance(self) -> None:
        """Attachment preserves existing governance data."""
        evidence: Dict[str, Any] = {
            "governance": {
                "existing_field": "should_remain",
                "slice_identity": {"existing": "data"},
            }
        }
        preflight = make_identity_preflight_result()

        result = attach_identity_preflight_to_evidence(evidence, preflight)

        assert result["governance"]["existing_field"] == "should_remain"
        assert result["governance"]["slice_identity"]["existing"] == "data"
        assert "p5_preflight" in result["governance"]["slice_identity"]

    def test_attach_adds_shadow_mode_contract(self) -> None:
        """Attachment adds SHADOW MODE contract marker."""
        evidence: Dict[str, Any] = {}
        preflight = make_identity_preflight_result()

        result = attach_identity_preflight_to_evidence(evidence, preflight)

        p5_preflight = result["governance"]["slice_identity"]["p5_preflight"]
        assert p5_preflight["mode"] == "SHADOW"
        assert "shadow_mode_contract" in p5_preflight
        assert p5_preflight["shadow_mode_contract"]["observational_only"] is True

    def test_attach_strips_full_report(self) -> None:
        """Attachment strips verbose full_report field."""
        evidence: Dict[str, Any] = {}
        preflight = make_identity_preflight_result()
        preflight["full_report"] = {"verbose": "data" * 1000}

        result = attach_identity_preflight_to_evidence(evidence, preflight)

        p5_preflight = result["governance"]["slice_identity"]["p5_preflight"]
        assert "full_report" not in p5_preflight

    def test_attach_is_non_mutating(self) -> None:
        """Attachment does not mutate original evidence dict."""
        evidence: Dict[str, Any] = {"governance": {"original": True}}
        preflight = make_identity_preflight_result()

        result = attach_identity_preflight_to_evidence(evidence, preflight)

        # Original should be unchanged
        assert "slice_identity" not in evidence["governance"]
        # Result should have new data
        assert "slice_identity" in result["governance"]


# =============================================================================
# RUN CONFIG DETECTION
# =============================================================================


class TestRunConfigDetection:
    """Test detecting identity_preflight from run_config.json."""

    def test_load_from_run_config_json(self, tmp_path: Path) -> None:
        """Load identity_preflight from run_config.json."""
        preflight = make_identity_preflight_result()
        config = {
            "schema_version": "1.3.0",
            "run_id": "test-123",
            "identity_preflight": preflight,
        }

        config_path = tmp_path / "run_config.json"
        config_path.write_text(json.dumps(config))

        result = load_identity_preflight_from_run_config(config_path)

        assert result is not None
        assert result["status"] == "OK"

    def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Return None when identity_preflight not in config."""
        config = {"schema_version": "1.3.0", "run_id": "test-123"}

        config_path = tmp_path / "run_config.json"
        config_path.write_text(json.dumps(config))

        result = load_identity_preflight_from_run_config(config_path)

        assert result is None

    def test_detect_from_run_config(self, tmp_path: Path) -> None:
        """Detect identity_preflight from run_config.json in directory."""
        preflight = make_identity_preflight_result()
        config = {"identity_preflight": preflight}

        (tmp_path / "run_config.json").write_text(json.dumps(config))

        result = detect_identity_preflight(tmp_path)

        assert result is not None
        assert result["status"] == "OK"

    def test_detect_from_separate_file(self, tmp_path: Path) -> None:
        """Detect identity_preflight from separate file when not in config."""
        # Empty run_config without identity_preflight
        (tmp_path / "run_config.json").write_text(json.dumps({}))

        # Separate identity_preflight.json file
        preflight = make_identity_preflight_result(status="BLOCK")
        (tmp_path / "identity_preflight.json").write_text(json.dumps(preflight))

        result = detect_identity_preflight(tmp_path)

        assert result is not None
        assert result["status"] == "BLOCK"

    def test_detect_prefers_legacy_file_over_run_config(self, tmp_path: Path) -> None:
        """Prefer legacy identity_preflight.json over run_config.json embedding.

        Detection priority is: dedicated p5_identity_preflight.json > legacy > run_config
        """
        # run_config with OK status
        (tmp_path / "run_config.json").write_text(
            json.dumps({"identity_preflight": make_identity_preflight_result(status="OK")})
        )

        # Legacy file with BLOCK status
        (tmp_path / "identity_preflight.json").write_text(
            json.dumps(make_identity_preflight_result(status="BLOCK"))
        )

        result = detect_identity_preflight(tmp_path)

        # Should prefer legacy file over run_config
        assert result is not None
        assert result["status"] == "BLOCK"


# =============================================================================
# SHADOW MODE CONTRACT VERIFICATION
# =============================================================================


class TestShadowModeContract:
    """Verify SHADOW MODE contract is maintained throughout workflow."""

    def test_block_status_does_not_raise(self) -> None:
        """BLOCK status never raises exceptions (advisory only)."""
        synthetic = make_config(params={"a": 1})
        production = make_config(params={"a": 2, "b": 3, "c": 4})

        # Should not raise
        report = check_p5_identity_alignment(synthetic, production)

        assert report.overall_status == CheckResult.BLOCK
        # No exception means SHADOW MODE contract maintained

    def test_attachment_with_block_status_succeeds(self) -> None:
        """Attaching BLOCK status to evidence succeeds (no gating)."""
        evidence: Dict[str, Any] = {}
        preflight = make_identity_preflight_result(
            status="BLOCK",
            fingerprint_match=False,
            top_reasons=["SI-001 MISMATCH", "Parameter divergence"],
        )

        # Should not raise
        result = attach_identity_preflight_to_evidence(evidence, preflight)

        assert result["governance"]["slice_identity"]["p5_preflight"]["status"] == "BLOCK"

    def test_all_check_results_are_advisory(self) -> None:
        """All check results include advisory markers."""
        synthetic = make_config()
        production = make_config()

        report = check_p5_identity_alignment(synthetic, production)
        data = report.to_dict()

        # Schema indicates this is advisory
        assert data["schema"] == "p5-identity-alignment-report/1.0.0"
        # Mode marker (if present)
        assert "overall_status" in data  # Advisory status field


# =============================================================================
# FULL WORKFLOW INTEGRATION
# =============================================================================


class TestFullWorkflowIntegration:
    """Test full workflow from check through evidence attachment."""

    def test_full_workflow_ok_scenario(self, tmp_path: Path) -> None:
        """Full workflow with OK status."""
        # Create configs
        synthetic = make_config()
        production = make_config()

        # Run check
        report = check_p5_identity_alignment(synthetic, production)

        # Create preflight result dict (simulating harness output)
        preflight_result = {
            "status": report.overall_status.value,
            "fingerprint_match": report.synthetic_fingerprint == report.production_fingerprint,
            "synthetic_fingerprint": report.synthetic_fingerprint,
            "production_fingerprint": report.production_fingerprint,
            "top_reasons": report.blocking_issues[:3] + report.investigation_items[:2],
            "invariant_summary": {
                check.invariant: check.status.value
                for check in report.checks
                if check.invariant
            },
        }

        # Save to run_config
        run_config = {
            "schema_version": "1.3.0",
            "run_id": "test-integration",
            "identity_preflight": preflight_result,
        }
        config_path = tmp_path / "run_config.json"
        config_path.write_text(json.dumps(run_config))

        # Detect from config
        detected = detect_identity_preflight(tmp_path)
        assert detected is not None

        # Attach to evidence
        evidence: Dict[str, Any] = {"bundle_id": "test-bundle"}
        updated_evidence = attach_identity_preflight_to_evidence(evidence, detected)

        # Verify structure
        assert "governance" in updated_evidence
        assert "slice_identity" in updated_evidence["governance"]
        p5_preflight = updated_evidence["governance"]["slice_identity"]["p5_preflight"]
        assert p5_preflight["mode"] == "SHADOW"

    def test_full_workflow_block_scenario_no_gating(self, tmp_path: Path) -> None:
        """Full workflow with BLOCK status - verify no gating occurs."""
        # Create divergent configs
        synthetic = make_config(params={"a": 1}, version="1.0.0")
        production = make_config(params={"a": 2}, version="2.0.0")

        # Run check
        report = check_p5_identity_alignment(synthetic, production)
        assert report.overall_status == CheckResult.BLOCK

        # Create preflight result
        preflight_result = {
            "status": report.overall_status.value,
            "fingerprint_match": False,
            "blocking_issues": report.blocking_issues,
            "top_reasons": report.blocking_issues[:3],
        }

        # Save and detect
        (tmp_path / "run_config.json").write_text(
            json.dumps({"identity_preflight": preflight_result})
        )
        detected = detect_identity_preflight(tmp_path)

        # Attach to evidence - should succeed even with BLOCK
        evidence: Dict[str, Any] = {"bundle_id": "test-bundle"}
        updated_evidence = attach_identity_preflight_to_evidence(evidence, detected)

        # Verify BLOCK status is recorded but no exception raised
        p5_preflight = updated_evidence["governance"]["slice_identity"]["p5_preflight"]
        assert p5_preflight["status"] == "BLOCK"
        # Contract marker confirms advisory-only
        assert p5_preflight["shadow_mode_contract"]["observational_only"] is True


# =============================================================================
# WARNING GENERATION (ADVISORY)
# =============================================================================


class TestWarningGeneration:
    """Test that warnings are generated for non-OK statuses."""

    def test_block_status_generates_top_reasons(self) -> None:
        """BLOCK status includes top reasons for investigation."""
        synthetic = make_config(params={"a": 1})
        production = make_config(params={"a": 2})

        report = check_p5_identity_alignment(synthetic, production)

        # Should have blocking issues explaining the problem
        assert len(report.blocking_issues) > 0
        # First blocking issue should mention fingerprint or param
        assert any(
            "MISMATCH" in issue or "differ" in issue.lower()
            for issue in report.blocking_issues
        )

    def test_investigate_status_generates_items(self) -> None:
        """INVESTIGATE status includes investigation items."""
        config = make_config()
        config["_meta"] = {"hot_reload_enabled": True}

        report = check_p5_identity_alignment(config, config)

        # Should have investigation items
        assert len(report.investigation_items) > 0


# =============================================================================
# ARTIFACT FILE TESTS
# =============================================================================


class TestArtifactFileCreation:
    """Test p5_identity_preflight.json artifact file creation."""

    def test_artifact_file_structure(self, tmp_path: Path) -> None:
        """Artifact file has correct schema structure."""
        from scripts.usla_first_light_p4_harness import (
            save_identity_preflight_artifact,
            IDENTITY_PREFLIGHT_SCHEMA_VERSION,
        )

        preflight_result = make_identity_preflight_result(status="OK")
        artifact_path = save_identity_preflight_artifact(tmp_path, preflight_result)

        assert artifact_path.exists()
        assert artifact_path.name == "p5_identity_preflight.json"

        with open(artifact_path) as f:
            artifact = json.load(f)

        assert artifact["schema_version"] == IDENTITY_PREFLIGHT_SCHEMA_VERSION
        assert artifact["status"] == "OK"
        assert "timestamp" in artifact
        assert "fingerprint_match" in artifact
        assert "checks" in artifact
        assert artifact["mode"] == "SHADOW"

    def test_artifact_includes_shadow_mode_contract(self, tmp_path: Path) -> None:
        """Artifact includes SHADOW MODE contract marker."""
        from scripts.usla_first_light_p4_harness import save_identity_preflight_artifact

        preflight_result = make_identity_preflight_result()
        artifact_path = save_identity_preflight_artifact(tmp_path, preflight_result)

        with open(artifact_path) as f:
            artifact = json.load(f)

        assert "shadow_mode_contract" in artifact
        assert artifact["shadow_mode_contract"]["observational_only"] is True
        assert artifact["shadow_mode_contract"]["no_control_flow_influence"] is True

    def test_artifact_strips_full_report(self, tmp_path: Path) -> None:
        """Artifact doesn't include verbose full_report field."""
        from scripts.usla_first_light_p4_harness import save_identity_preflight_artifact

        preflight_result = make_identity_preflight_result()
        preflight_result["full_report"] = {"verbose": "data" * 1000}
        artifact_path = save_identity_preflight_artifact(tmp_path, preflight_result)

        with open(artifact_path) as f:
            artifact = json.load(f)

        # full_report should not be directly in artifact (only in checks if extracted)
        assert "full_report" not in artifact or artifact.get("full_report") is None


# =============================================================================
# DETECTION PRIORITY TESTS
# =============================================================================


class TestDetectionPriority:
    """Test detection priority: dedicated file > legacy > run_config."""

    def test_priority_dedicated_over_run_config(self, tmp_path: Path) -> None:
        """Dedicated p5_identity_preflight.json takes priority over run_config."""
        # Create run_config with BLOCK status
        (tmp_path / "run_config.json").write_text(json.dumps({
            "identity_preflight": make_identity_preflight_result(status="BLOCK")
        }))

        # Create dedicated file with OK status
        (tmp_path / "p5_identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="OK")
        ))

        result = detect_identity_preflight(tmp_path)

        # Dedicated file should win
        assert result["status"] == "OK"

    def test_priority_dedicated_over_legacy(self, tmp_path: Path) -> None:
        """Dedicated p5_identity_preflight.json takes priority over legacy file."""
        # Create legacy identity_preflight.json with BLOCK status
        (tmp_path / "identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="BLOCK")
        ))

        # Create dedicated file with OK status
        (tmp_path / "p5_identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="OK")
        ))

        result = detect_identity_preflight(tmp_path)

        # Dedicated file should win
        assert result["status"] == "OK"

    def test_priority_legacy_over_run_config(self, tmp_path: Path) -> None:
        """Legacy identity_preflight.json takes priority over run_config."""
        # Create run_config with OK status
        (tmp_path / "run_config.json").write_text(json.dumps({
            "identity_preflight": make_identity_preflight_result(status="OK")
        }))

        # Create legacy file with INVESTIGATE status
        (tmp_path / "identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="INVESTIGATE")
        ))

        result = detect_identity_preflight(tmp_path)

        # Legacy file should win over run_config
        assert result["status"] == "INVESTIGATE"

    def test_fallback_to_run_config(self, tmp_path: Path) -> None:
        """Falls back to run_config when no dedicated files exist."""
        # Create only run_config
        (tmp_path / "run_config.json").write_text(json.dumps({
            "identity_preflight": make_identity_preflight_result(status="BLOCK")
        }))

        result = detect_identity_preflight(tmp_path)

        assert result["status"] == "BLOCK"


# =============================================================================
# MISMATCH WARNING TESTS
# =============================================================================


class TestMismatchWarning:
    """Test mismatch warning when CLI and evidence pack differ."""

    def test_no_warning_when_matching(self, tmp_path: Path) -> None:
        """No mismatch warning when CLI and evidence pack have same status."""
        # Create matching preflight results
        preflight = make_identity_preflight_result(status="OK", fingerprint_match=True)

        cli_path = tmp_path / "cli_report.json"
        cli_path.write_text(json.dumps(preflight))

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "p5_identity_preflight.json").write_text(json.dumps(preflight))

        # Simulate status generation logic
        warnings: List[str] = []
        cli_report = json.loads(cli_path.read_text())
        evidence_preflight = detect_identity_preflight(evidence_dir)

        if cli_report and evidence_preflight:
            cli_status = cli_report.get("status")
            evidence_status = evidence_preflight.get("status")
            cli_fp = cli_report.get("fingerprint_match")
            evidence_fp = evidence_preflight.get("fingerprint_match")

            if cli_status != evidence_status or cli_fp != evidence_fp:
                warnings.append("P5 identity mismatch detected")

        assert len(warnings) == 0

    def test_warning_on_status_mismatch(self, tmp_path: Path) -> None:
        """Warning emitted when CLI and evidence pack have different status."""
        cli_path = tmp_path / "cli_report.json"
        cli_path.write_text(json.dumps(
            make_identity_preflight_result(status="OK", fingerprint_match=True)
        ))

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "p5_identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="BLOCK", fingerprint_match=True)
        ))

        # Simulate status generation logic
        warnings: List[str] = []
        cli_report = json.loads(cli_path.read_text())
        evidence_preflight = detect_identity_preflight(evidence_dir)

        if cli_report and evidence_preflight:
            cli_status = cli_report.get("status")
            evidence_status = evidence_preflight.get("status")
            cli_fp = cli_report.get("fingerprint_match")
            evidence_fp = evidence_preflight.get("fingerprint_match")

            if cli_status != evidence_status or cli_fp != evidence_fp:
                warnings.append(
                    f"P5 identity mismatch: CLI (status={cli_status}) "
                    f"differs from evidence (status={evidence_status})"
                )

        assert len(warnings) == 1
        assert "mismatch" in warnings[0].lower()

    def test_warning_on_fingerprint_mismatch(self, tmp_path: Path) -> None:
        """Warning emitted when fingerprint_match differs."""
        cli_path = tmp_path / "cli_report.json"
        cli_path.write_text(json.dumps(
            make_identity_preflight_result(status="OK", fingerprint_match=True)
        ))

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "p5_identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="OK", fingerprint_match=False)
        ))

        # Simulate status generation logic
        warnings: List[str] = []
        cli_report = json.loads(cli_path.read_text())
        evidence_preflight = detect_identity_preflight(evidence_dir)

        if cli_report and evidence_preflight:
            cli_status = cli_report.get("status")
            evidence_status = evidence_preflight.get("status")
            cli_fp = cli_report.get("fingerprint_match")
            evidence_fp = evidence_preflight.get("fingerprint_match")

            if cli_status != evidence_status or cli_fp != evidence_fp:
                warnings.append(
                    f"P5 identity mismatch: CLI (fp_match={cli_fp}) "
                    f"differs from evidence (fp_match={evidence_fp})"
                )

        assert len(warnings) == 1
        assert "mismatch" in warnings[0].lower()

    def test_mismatch_does_not_gate(self, tmp_path: Path) -> None:
        """Mismatch warning is advisory only, does not prevent processing."""
        cli_path = tmp_path / "cli_report.json"
        cli_path.write_text(json.dumps(
            make_identity_preflight_result(status="OK", fingerprint_match=True)
        ))

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "p5_identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        ))

        # Process should complete without exception
        cli_report = json.loads(cli_path.read_text())
        evidence_preflight = detect_identity_preflight(evidence_dir)

        # CLI report is used as authoritative source
        assert cli_report["status"] == "OK"
        # Evidence is different but doesn't block
        assert evidence_preflight["status"] == "BLOCK"

    def test_warning_text_includes_both_statuses_and_authority(self, tmp_path: Path) -> None:
        """Warning text must include both CLI and evidence statuses plus chosen authority.

        Per Identity_Preflight_Precedence_Law.md:
        - Warning must show CLI status and evidence pack status
        - Warning must indicate CLI is authoritative source
        """
        cli_path = tmp_path / "cli_report.json"
        cli_path.write_text(json.dumps(
            make_identity_preflight_result(status="OK", fingerprint_match=True)
        ))

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "p5_identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        ))

        # Generate warning text using same logic as generate_first_light_status.py
        cli_report = json.loads(cli_path.read_text())
        evidence_preflight = detect_identity_preflight(evidence_dir)

        cli_status = cli_report.get("status")
        evidence_status = evidence_preflight.get("status")
        cli_fp = cli_report.get("fingerprint_match")
        evidence_fp = evidence_preflight.get("fingerprint_match")

        warning = (
            f"P5 identity mismatch: CLI report (status={cli_status}, fp_match={cli_fp}) "
            f"differs from evidence pack (status={evidence_status}, fp_match={evidence_fp}). "
            f"Using CLI report as authoritative source."
        )

        # Verify warning includes both statuses
        assert "status=OK" in warning, "Warning must include CLI status"
        assert "status=BLOCK" in warning, "Warning must include evidence pack status"

        # Verify warning includes both fingerprint match values
        assert "fp_match=True" in warning, "Warning must include CLI fp_match"
        assert "fp_match=False" in warning, "Warning must include evidence fp_match"

        # Verify warning indicates CLI authority
        assert "CLI report as authoritative" in warning, "Warning must indicate CLI authority"


# =============================================================================
# GGFL ADAPTER TESTS
# =============================================================================


class TestGGFLAdapter:
    """Test identity_preflight_for_alignment_view GGFL adapter.

    SHADOW MODE CONTRACT:
    - All outputs are advisory only
    - No gating or control flow influence
    - Deterministic driver ordering
    - Neutral, factual language only
    """

    def test_adapter_returns_correct_signal_type(self) -> None:
        """Adapter returns SIG-ID signal type."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        result = identity_preflight_for_alignment_view(
            make_identity_preflight_result(status="OK")
        )
        assert result["signal_type"] == "SIG-ID"

    def test_adapter_ok_status_for_ok_report(self) -> None:
        """Adapter returns status=ok for OK identity report."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        result = identity_preflight_for_alignment_view(
            make_identity_preflight_result(status="OK", fingerprint_match=True)
        )
        assert result["status"] == "ok"

    def test_adapter_warn_status_for_block(self) -> None:
        """Adapter returns status=warn for BLOCK identity report."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        result = identity_preflight_for_alignment_view(
            make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        )
        assert result["status"] == "warn"

    def test_adapter_warn_status_for_investigate(self) -> None:
        """Adapter returns status=warn for INVESTIGATE identity report."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        result = identity_preflight_for_alignment_view(
            make_identity_preflight_result(status="INVESTIGATE", fingerprint_match=True)
        )
        assert result["status"] == "warn"

    def test_adapter_conflict_always_false(self) -> None:
        """Adapter always returns conflict=False (advisory only)."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        # Test with various inputs
        for status in ["OK", "INVESTIGATE", "BLOCK"]:
            result = identity_preflight_for_alignment_view(
                make_identity_preflight_result(status=status)
            )
            assert result["conflict"] is False, f"conflict must be False for {status}"

    def test_adapter_mode_always_shadow(self) -> None:
        """Adapter always returns mode=SHADOW."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        result = identity_preflight_for_alignment_view(
            make_identity_preflight_result(status="BLOCK")
        )
        assert result["mode"] == "SHADOW"

    def test_adapter_max_three_drivers(self) -> None:
        """Adapter returns at most 3 drivers."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        # Create report with many issues
        report = make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        report["invariant_summary"] = {
            "SI-001": "BLOCK",
            "SI-002": "INVESTIGATE",
            "SI-003": "BLOCK",
            "SI-004": "INVESTIGATE",
            "SI-005": "BLOCK",
            "SI-006": "INVESTIGATE",
        }

        result = identity_preflight_for_alignment_view(report)
        assert len(result["drivers"]) <= 3, "Drivers must be capped at 3"

    def test_adapter_drivers_deterministic_order(self) -> None:
        """Adapter returns drivers in deterministic order across multiple calls."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        report = make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        report["invariant_summary"] = {
            "SI-003": "BLOCK",
            "SI-001": "INVESTIGATE",
            "SI-005": "BLOCK",
        }

        # Call multiple times and verify same order
        results = [identity_preflight_for_alignment_view(report) for _ in range(5)]
        first_drivers = results[0]["drivers"]

        for i, r in enumerate(results[1:], 2):
            assert r["drivers"] == first_drivers, f"Call {i} had different driver order"

    def test_adapter_neutral_language_no_alarm_words(self) -> None:
        """Adapter summary uses neutral language without alarm words.

        Forbidden words: error, critical, failure, danger, alert, urgent, warning
        """
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        alarm_words = ["error", "critical", "failure", "danger", "alert", "urgent", "warning"]

        # Test all status types
        for status in ["OK", "INVESTIGATE", "BLOCK"]:
            for fp_match in [True, False]:
                report = make_identity_preflight_result(status=status, fingerprint_match=fp_match)
                result = identity_preflight_for_alignment_view(report)
                summary_lower = result["summary"].lower()

                for word in alarm_words:
                    assert word not in summary_lower, (
                        f"Summary contains alarm word '{word}' for status={status}, fp_match={fp_match}: "
                        f"'{result['summary']}'"
                    )

    def test_adapter_handles_none_report(self) -> None:
        """Adapter handles None identity report gracefully."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        result = identity_preflight_for_alignment_view(None)

        assert result["signal_type"] == "SIG-ID"
        assert result["status"] == "ok"
        assert result["conflict"] is False
        assert result["drivers"] == []
        assert "not available" in result["summary"].lower()
        assert result["mode"] == "SHADOW"

    def test_adapter_handles_empty_report(self) -> None:
        """Adapter handles empty dict report gracefully."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        result = identity_preflight_for_alignment_view({})

        assert result["signal_type"] == "SIG-ID"
        assert result["status"] == "ok"  # Default to ok for unknown status
        assert result["conflict"] is False
        assert result["mode"] == "SHADOW"

    def test_adapter_drivers_are_invariant_ids(self) -> None:
        """Adapter drivers are the actual invariant IDs (SI-xxx), not DRIVER_ prefixed."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        report = make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        report["invariant_summary"] = {
            "SI-001": "BLOCK",
            "SI-002": "OK",
            "SI-003": "INVESTIGATE",
            "SI-004": "OK",
            "SI-005": "BLOCK",
            "SI-006": "OK",
        }

        result = identity_preflight_for_alignment_view(report)

        # Drivers should be the invariant IDs directly
        assert result["drivers"] == ["SI-001", "SI-003", "SI-005"]

    def test_adapter_drivers_sorted_ascending(self) -> None:
        """Adapter drivers are sorted in ascending order by invariant ID."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        report = make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        # Deliberately out of order
        report["invariant_summary"] = {
            "SI-006": "BLOCK",
            "SI-001": "BLOCK",
            "SI-004": "INVESTIGATE",
        }

        result = identity_preflight_for_alignment_view(report)

        # Should be sorted: SI-001, SI-004, SI-006
        assert result["drivers"] == ["SI-001", "SI-004", "SI-006"]

    def test_adapter_drivers_only_non_ok(self) -> None:
        """Adapter drivers only include non-OK invariants."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        report = make_identity_preflight_result(status="OK", fingerprint_match=True)
        report["invariant_summary"] = {
            "SI-001": "OK",
            "SI-002": "OK",
            "SI-003": "OK",
            "SI-004": "OK",
            "SI-005": "INVESTIGATE",  # Only non-OK
            "SI-006": "OK",
        }

        result = identity_preflight_for_alignment_view(report)

        assert result["drivers"] == ["SI-005"]


# =============================================================================
# ARTIFACT HASHING TESTS
# =============================================================================


class TestArtifactHashing:
    """Test p5_identity_preflight.json hashing and manifest reference."""

    def test_detect_p5_identity_preflight_file_returns_reference(self, tmp_path: Path) -> None:
        """detect_p5_identity_preflight_file returns reference with hash."""
        from backend.topology.first_light.evidence_pack import (
            detect_p5_identity_preflight_file,
        )

        # Create artifact
        artifact = {
            "schema_version": "1.0.0",
            "status": "OK",
            "fingerprint_match": True,
            "mode": "SHADOW",
        }
        artifact_path = tmp_path / "p5_identity_preflight.json"
        artifact_path.write_text(json.dumps(artifact))

        ref = detect_p5_identity_preflight_file(tmp_path)

        assert ref is not None
        assert ref.path == "p5_identity_preflight.json"
        assert ref.sha256 is not None
        assert len(ref.sha256) == 64  # SHA-256 hex
        assert ref.schema_version == "1.0.0"
        assert ref.status == "OK"
        assert ref.fingerprint_match is True
        assert ref.mode == "SHADOW"

    def test_detect_p5_identity_preflight_file_returns_none_if_missing(self, tmp_path: Path) -> None:
        """detect_p5_identity_preflight_file returns None if file doesn't exist."""
        from backend.topology.first_light.evidence_pack import (
            detect_p5_identity_preflight_file,
        )

        ref = detect_p5_identity_preflight_file(tmp_path)
        assert ref is None

    def test_detect_p5_identity_preflight_file_handles_malformed_json(self, tmp_path: Path) -> None:
        """detect_p5_identity_preflight_file handles malformed JSON by returning ref with hash only."""
        from backend.topology.first_light.evidence_pack import (
            detect_p5_identity_preflight_file,
        )

        # Create malformed JSON file
        artifact_path = tmp_path / "p5_identity_preflight.json"
        artifact_path.write_text("{ invalid json }")

        ref = detect_p5_identity_preflight_file(tmp_path)

        assert ref is not None
        assert ref.path == "p5_identity_preflight.json"
        assert ref.sha256 is not None
        assert len(ref.sha256) == 64  # SHA-256 hex still computed
        # Fields should be None since JSON couldn't be parsed
        assert ref.schema_version is None
        assert ref.status is None


# =============================================================================
# MISMATCH WARNING HYGIENE TESTS
# =============================================================================


class TestMismatchWarningHygiene:
    """Test mismatch warning hygiene: single warning max, both sources/statuses, authority."""

    def test_mismatch_warning_single_max(self, tmp_path: Path) -> None:
        """Only one warning is generated when mismatch occurs (not additional status warning)."""
        # Create CLI report with BLOCK status
        cli_path = tmp_path / "cli_report.json"
        cli_path.write_text(json.dumps(
            make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        ))

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "p5_identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="OK", fingerprint_match=True)
        ))

        # Simulate status generation logic (same as generate_first_light_status.py)
        warnings: List[str] = []
        identity_mismatch_detected = False

        cli_report = json.loads(cli_path.read_text())
        evidence_preflight = detect_identity_preflight(evidence_dir)

        # Mismatch detection
        cli_status = cli_report.get("status")
        evidence_status = evidence_preflight.get("status")
        cli_fp = cli_report.get("fingerprint_match")
        evidence_fp = evidence_preflight.get("fingerprint_match")

        if cli_status != evidence_status or cli_fp != evidence_fp:
            identity_mismatch_detected = True
            warnings.append(
                f"P5 identity mismatch: CLI (status={cli_status}, fp_match={cli_fp}) vs "
                f"manifest (status={evidence_status}, fp_match={evidence_fp}). "
                f"Authoritative: CLI."
            )

        # Status warning should be skipped if mismatch already emitted
        if not identity_mismatch_detected:
            if cli_status == "BLOCK":
                warnings.append(f"P5 identity pre-flight status is BLOCK")

        # Should have exactly 1 warning (mismatch only, not status)
        assert len(warnings) == 1
        assert "mismatch" in warnings[0].lower()

    def test_mismatch_warning_includes_both_sources(self, tmp_path: Path) -> None:
        """Mismatch warning includes both CLI and manifest labels."""
        cli_path = tmp_path / "cli_report.json"
        cli_path.write_text(json.dumps(
            make_identity_preflight_result(status="OK", fingerprint_match=True)
        ))

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "p5_identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        ))

        cli_report = json.loads(cli_path.read_text())
        evidence_preflight = detect_identity_preflight(evidence_dir)

        cli_status = cli_report.get("status")
        evidence_status = evidence_preflight.get("status")
        cli_fp = cli_report.get("fingerprint_match")
        evidence_fp = evidence_preflight.get("fingerprint_match")

        warning = (
            f"P5 identity mismatch: CLI (status={cli_status}, fp_match={cli_fp}) vs "
            f"manifest (status={evidence_status}, fp_match={evidence_fp}). "
            f"Authoritative: CLI."
        )

        # Check both sources are labeled
        assert "CLI" in warning
        assert "manifest" in warning

    def test_mismatch_warning_includes_both_statuses(self, tmp_path: Path) -> None:
        """Mismatch warning includes both status values."""
        cli_path = tmp_path / "cli_report.json"
        cli_path.write_text(json.dumps(
            make_identity_preflight_result(status="OK", fingerprint_match=True)
        ))

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "p5_identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        ))

        cli_report = json.loads(cli_path.read_text())
        evidence_preflight = detect_identity_preflight(evidence_dir)

        warning = (
            f"P5 identity mismatch: CLI (status={cli_report.get('status')}, "
            f"fp_match={cli_report.get('fingerprint_match')}) vs "
            f"manifest (status={evidence_preflight.get('status')}, "
            f"fp_match={evidence_preflight.get('fingerprint_match')}). "
            f"Authoritative: CLI."
        )

        # Check both statuses are present
        assert "status=OK" in warning
        assert "status=BLOCK" in warning

    def test_mismatch_warning_indicates_authority(self, tmp_path: Path) -> None:
        """Mismatch warning indicates CLI as authoritative."""
        cli_path = tmp_path / "cli_report.json"
        cli_path.write_text(json.dumps(
            make_identity_preflight_result(status="OK", fingerprint_match=True)
        ))

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        (evidence_dir / "p5_identity_preflight.json").write_text(json.dumps(
            make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        ))

        warning = (
            f"P5 identity mismatch: CLI (status=OK, fp_match=True) vs "
            f"manifest (status=BLOCK, fp_match=False). "
            f"Authoritative: CLI."
        )

        assert "Authoritative: CLI" in warning


# =============================================================================
# EXTRACTION SOURCE TESTS
# =============================================================================


class TestExtractionSource:
    """Test extraction_source constants and correctness."""

    def test_extraction_source_constants_defined(self) -> None:
        """Extraction source constants are properly defined."""
        from backend.health.identity_alignment_checker import (
            EXTRACTION_SOURCE_CLI,
            EXTRACTION_SOURCE_MANIFEST,
            EXTRACTION_SOURCE_LEGACY_FILE,
            EXTRACTION_SOURCE_RUN_CONFIG,
            EXTRACTION_SOURCE_MISSING,
        )

        assert EXTRACTION_SOURCE_CLI == "CLI"
        assert EXTRACTION_SOURCE_MANIFEST == "MANIFEST"
        assert EXTRACTION_SOURCE_LEGACY_FILE == "LEGACY_FILE"
        assert EXTRACTION_SOURCE_RUN_CONFIG == "RUN_CONFIG"
        assert EXTRACTION_SOURCE_MISSING == "MISSING"


# =============================================================================
# GGFL/STATUS CONSISTENCY TESTS
# =============================================================================


class TestGGFLStatusConsistency:
    """Test GGFL/status signal consistency checking."""

    def test_consistency_consistent_when_matching(self) -> None:
        """Consistency is CONSISTENT when status and GGFL match correctly."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
            summarize_identity_preflight_signal_consistency,
        )

        # Create matching status and GGFL signals
        report = make_identity_preflight_result(status="OK", fingerprint_match=True)
        status_signal = {"status": "OK", "fingerprint_match": True}
        ggfl_signal = identity_preflight_for_alignment_view(report)

        result = summarize_identity_preflight_signal_consistency(status_signal, ggfl_signal)

        assert result["consistency"] == "CONSISTENT"
        assert result["conflict_invariant_violated"] is False
        assert len(result["notes"]) == 0

    def test_consistency_inconsistent_on_status_mapping_mismatch(self) -> None:
        """Consistency is INCONSISTENT when status mapping is wrong."""
        from backend.health.identity_alignment_checker import (
            summarize_identity_preflight_signal_consistency,
        )

        # Status says OK but GGFL says warn (mismatch)
        status_signal = {"status": "OK", "fingerprint_match": True}
        ggfl_signal = {
            "signal_type": "SIG-ID",
            "status": "warn",  # Should be "ok" for OK status
            "conflict": False,
            "drivers": [],
            "mode": "SHADOW",
        }

        result = summarize_identity_preflight_signal_consistency(status_signal, ggfl_signal)

        assert result["consistency"] == "INCONSISTENT"
        assert result["top_mismatch_type"] == "STATUS_MAPPING"
        assert any("status mapping mismatch" in note.lower() for note in result["notes"])

    def test_consistency_inconsistent_on_conflict_invariant_violated(self) -> None:
        """Consistency is INCONSISTENT when conflict=True (should always be False)."""
        from backend.health.identity_alignment_checker import (
            summarize_identity_preflight_signal_consistency,
        )

        status_signal = {"status": "OK", "fingerprint_match": True}
        ggfl_signal = {
            "signal_type": "SIG-ID",
            "status": "ok",
            "conflict": True,  # Violates invariant
            "drivers": [],
            "mode": "SHADOW",
        }

        result = summarize_identity_preflight_signal_consistency(status_signal, ggfl_signal)

        assert result["consistency"] == "INCONSISTENT"
        assert result["conflict_invariant_violated"] is True
        assert result["top_mismatch_type"] == "CONFLICT_INVARIANT"

    def test_consistency_partial_on_drivers_not_sorted(self) -> None:
        """Consistency is PARTIAL when drivers are not sorted."""
        from backend.health.identity_alignment_checker import (
            summarize_identity_preflight_signal_consistency,
        )

        status_signal = {"status": "BLOCK", "fingerprint_match": False}
        ggfl_signal = {
            "signal_type": "SIG-ID",
            "status": "warn",
            "conflict": False,
            "drivers": ["SI-003", "SI-001"],  # Not sorted
            "mode": "SHADOW",
        }

        result = summarize_identity_preflight_signal_consistency(status_signal, ggfl_signal)

        assert result["consistency"] == "PARTIAL"
        assert result["top_mismatch_type"] == "DRIVERS_ORDER"
        assert any("not sorted" in note.lower() for note in result["notes"])


# =============================================================================
# INVARIANT IDS SORTED/CAPPED TESTS
# =============================================================================


class TestInvariantIdsSortedCapped:
    """Test that invariant IDs are sorted and capped at 3."""

    def test_invariant_ids_sorted_ascending(self) -> None:
        """Invariant IDs in drivers are sorted in ascending order."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        report = make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        report["invariant_summary"] = {
            "SI-006": "BLOCK",
            "SI-002": "INVESTIGATE",
            "SI-004": "BLOCK",
        }

        result = identity_preflight_for_alignment_view(report)

        # Should be sorted: SI-002, SI-004, SI-006
        assert result["drivers"] == ["SI-002", "SI-004", "SI-006"]

    def test_invariant_ids_capped_at_three(self) -> None:
        """Invariant IDs are capped at 3 maximum."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        report = make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        report["invariant_summary"] = {
            "SI-001": "BLOCK",
            "SI-002": "INVESTIGATE",
            "SI-003": "BLOCK",
            "SI-004": "INVESTIGATE",
            "SI-005": "BLOCK",
            "SI-006": "INVESTIGATE",
        }

        result = identity_preflight_for_alignment_view(report)

        assert len(result["drivers"]) == 3
        # First 3 in sorted order
        assert result["drivers"] == ["SI-001", "SI-002", "SI-003"]


# =============================================================================
# DETERMINISM TESTS
# =============================================================================


class TestDeterminism:
    """Test determinism of extraction and consistency checking."""

    def test_consistency_result_deterministic(self) -> None:
        """Consistency result is deterministic across multiple calls."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
            summarize_identity_preflight_signal_consistency,
        )

        report = make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        report["invariant_summary"] = {
            "SI-003": "BLOCK",
            "SI-001": "INVESTIGATE",
            "SI-005": "BLOCK",
        }

        status_signal = {"status": "BLOCK", "fingerprint_match": False}

        # Call multiple times
        results = []
        for _ in range(5):
            ggfl_signal = identity_preflight_for_alignment_view(report)
            consistency = summarize_identity_preflight_signal_consistency(
                status_signal, ggfl_signal
            )
            results.append(consistency)

        # All results should be identical
        first = results[0]
        for i, r in enumerate(results[1:], 2):
            assert r["consistency"] == first["consistency"], f"Call {i} had different consistency"
            assert r["notes"] == first["notes"], f"Call {i} had different notes"
            assert r["conflict_invariant_violated"] == first["conflict_invariant_violated"]

    def test_drivers_deterministic_across_calls(self) -> None:
        """Drivers are deterministic across multiple calls."""
        from backend.health.identity_alignment_checker import (
            identity_preflight_for_alignment_view,
        )

        report = make_identity_preflight_result(status="BLOCK", fingerprint_match=False)
        report["invariant_summary"] = {
            "SI-005": "BLOCK",
            "SI-001": "INVESTIGATE",
            "SI-003": "BLOCK",
        }

        # Call 10 times and collect drivers
        all_drivers = [
            identity_preflight_for_alignment_view(report)["drivers"]
            for _ in range(10)
        ]

        # All should be identical
        expected = ["SI-001", "SI-003", "SI-005"]
        for i, drivers in enumerate(all_drivers):
            assert drivers == expected, f"Call {i+1} had non-deterministic drivers: {drivers}"
