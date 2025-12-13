"""
Tests for P5 Identity Alignment Checker.

Covers failure modes FM-001 through FM-004 as documented in:
  docs/system_law/P5_Identity_Flight_Check_Runbook.md

Status: PHASE X P5 PRE-FLIGHT
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.health.identity_alignment_checker import (
    CheckReport,
    CheckResult,
    check_p5_identity_alignment,
    diagnose_config_divergence,
)


# =============================================================================
# FIXTURE HELPERS
# =============================================================================


def make_baseline_config(
    version: str = "1.0.0",
    params: Dict[str, Any] | None = None,
    gates: Dict[str, Any] | None = None,
    meta: Dict[str, Any] | None = None,
    curriculum_fp: str | None = None,
) -> Dict[str, Any]:
    """Create baseline synthetic config."""
    config: Dict[str, Any] = {
        "version": version,
        "params": params or {
            "max_depth": 10,
            "batch_size": 100,
            "timeout_ms": 5000,
        },
        "gates": gates or {
            "uplift_gate": True,
            "safety_gate": True,
            "curriculum": {
                "enabled": True,
                "level": 3,
            },
        },
    }
    if meta:
        config["_meta"] = meta
    if curriculum_fp:
        config["_curriculum_fingerprint"] = curriculum_fp
    return config


def make_p4_evidence_pack(
    baseline_fingerprint: str | None = None,
    computed_fingerprint: str | None = None,
) -> Dict[str, Any]:
    """Create P4 evidence pack for testing."""
    slice_identity: Dict[str, Any] = {}
    if baseline_fingerprint:
        slice_identity["baseline_fingerprint"] = baseline_fingerprint
    if computed_fingerprint:
        slice_identity["computed_fingerprint"] = computed_fingerprint
    return {
        "governance": {
            "slice_identity": slice_identity,
        },
    }


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================


class TestCheckReport:
    """Tests for CheckReport class."""

    def test_empty_report_is_ok(self) -> None:
        """Empty report defaults to OK."""
        report = CheckReport()
        assert report.overall_status == CheckResult.OK
        assert report.get_exit_code() == 0

    def test_add_ok_check_keeps_ok(self) -> None:
        """Adding OK check keeps overall OK."""
        report = CheckReport()
        report.add_check("Test", CheckResult.OK, "All good", "SI-001")
        assert report.overall_status == CheckResult.OK
        assert report.get_exit_code() == 0

    def test_add_investigate_escalates(self) -> None:
        """Adding INVESTIGATE check escalates from OK."""
        report = CheckReport()
        report.add_check("Test", CheckResult.INVESTIGATE, "Check this", "SI-002")
        assert report.overall_status == CheckResult.INVESTIGATE
        assert report.get_exit_code() == 1

    def test_add_block_escalates_to_highest(self) -> None:
        """Adding BLOCK check escalates to highest severity."""
        report = CheckReport()
        report.add_check("T1", CheckResult.INVESTIGATE, "Check this", "SI-002")
        report.add_check("T2", CheckResult.BLOCK, "Stop here", "SI-001")
        assert report.overall_status == CheckResult.BLOCK
        assert report.get_exit_code() == 2

    def test_block_not_downgraded_by_investigate(self) -> None:
        """BLOCK status is not downgraded by subsequent INVESTIGATE."""
        report = CheckReport()
        report.add_check("T1", CheckResult.BLOCK, "Stop here", "SI-001")
        report.add_check("T2", CheckResult.INVESTIGATE, "Check this", "SI-002")
        assert report.overall_status == CheckResult.BLOCK
        assert report.get_exit_code() == 2

    def test_to_dict_includes_invariant_summary(self) -> None:
        """to_dict includes SI-001 through SI-006 status."""
        report = CheckReport()
        report.synthetic_fingerprint = "abc123"
        report.production_fingerprint = "abc123"
        report.add_check("FP", CheckResult.OK, "Match", "SI-001")

        data = report.to_dict()
        assert "invariant_summary" in data
        assert data["invariant_summary"]["SI-001"] == "OK"
        assert data["invariant_summary"]["SI-002"] == "UNCHECKED"

    def test_to_text_report_formatted(self) -> None:
        """to_text_report produces formatted output."""
        report = CheckReport()
        report.synthetic_fingerprint = "a" * 64
        report.production_fingerprint = "a" * 64
        report.add_check("Test", CheckResult.OK, "All good", "SI-001")

        text = report.to_text_report()
        assert "P5 IDENTITY FLIGHT CHECK REPORT" in text
        assert "[OK]" in text
        assert "[SI-001]" in text


# =============================================================================
# FM-001: CONFIG SOURCE DIVERGENCE
# =============================================================================


class TestFM001ConfigSourceDivergence:
    """
    FM-001: Config Source Divergence

    Scenario: Synthetic config loaded from YAML differs from production
              config loaded from Kubernetes ConfigMap due to path aliasing.

    Expected: BLOCK (SI-001 fingerprint mismatch)
    """

    def test_fm001_fingerprint_mismatch_blocks(self) -> None:
        """FM-001: Different param values cause SI-001 BLOCK."""
        synthetic = make_baseline_config(params={
            "max_depth": 10,
            "batch_size": 100,
            "timeout_ms": 5000,
        })
        production = make_baseline_config(params={
            "max_depth": 15,  # Changed!
            "batch_size": 100,
            "timeout_ms": 5000,
        })

        report = check_p5_identity_alignment(synthetic, production)

        assert report.overall_status == CheckResult.BLOCK
        assert report.get_exit_code() == 2

        # SI-001 should be BLOCK
        si001_check = next(c for c in report.checks if c.invariant == "SI-001")
        assert si001_check.status == CheckResult.BLOCK
        assert "MISMATCH" in si001_check.details

    def test_fm001_yaml_vs_configmap_divergence(self) -> None:
        """FM-001: Simulate YAML vs ConfigMap divergence."""
        # Synthetic: values from local YAML
        synthetic = {
            "version": "1.0.0",
            "params": {
                "threshold": 0.85,
                "retry_count": 3,
            },
            "gates": {"feature_x": True},
        }

        # Production: ConfigMap with slightly different defaults
        production = {
            "version": "1.0.0",
            "params": {
                "threshold": 0.80,  # ConfigMap had different default
                "retry_count": 3,
            },
            "gates": {"feature_x": True},
        }

        report = check_p5_identity_alignment(synthetic, production)

        # Should block due to fingerprint mismatch
        assert report.overall_status == CheckResult.BLOCK
        assert report.synthetic_fingerprint != report.production_fingerprint

        # Verify diagnosis shows the differing param
        diagnosis = diagnose_config_divergence(synthetic, production)
        assert not diagnosis["match"]
        assert any(p["param"] == "threshold" for p in diagnosis["differing_params"])

    def test_fm001_identical_configs_pass(self) -> None:
        """FM-001: Identical configs produce OK."""
        config = make_baseline_config()

        report = check_p5_identity_alignment(config, config)

        si001_check = next(c for c in report.checks if c.invariant == "SI-001")
        assert si001_check.status == CheckResult.OK
        assert report.synthetic_fingerprint == report.production_fingerprint


# =============================================================================
# FM-002: RUNTIME PARAMETER INJECTION
# =============================================================================


class TestFM002RuntimeParameterInjection:
    """
    FM-002: Runtime Parameter Injection

    Scenario: Production environment has hot-reload or auto-scaling enabled
              which can modify config mid-run.

    Expected: INVESTIGATE (SI-002 immutability concern)
    """

    def test_fm002_hot_reload_triggers_investigate(self) -> None:
        """FM-002: Hot-reload enabled triggers SI-002 INVESTIGATE."""
        synthetic = make_baseline_config()
        production = make_baseline_config(meta={
            "hot_reload_enabled": True,
        })

        report = check_p5_identity_alignment(synthetic, production)

        si002_check = next(c for c in report.checks if c.invariant == "SI-002")
        assert si002_check.status == CheckResult.INVESTIGATE
        assert "Hot-reload" in si002_check.details

    def test_fm002_auto_scaling_triggers_investigate(self) -> None:
        """FM-002: Auto-scaling enabled triggers SI-002 INVESTIGATE."""
        synthetic = make_baseline_config()
        production = make_baseline_config(meta={
            "auto_scaling_enabled": True,
        })

        report = check_p5_identity_alignment(synthetic, production)

        si002_check = next(c for c in report.checks if c.invariant == "SI-002")
        assert si002_check.status == CheckResult.INVESTIGATE
        assert "Auto-scaling" in si002_check.details

    def test_fm002_no_runtime_injection_is_ok(self) -> None:
        """FM-002: Disabled hot-reload/auto-scaling is OK."""
        synthetic = make_baseline_config()
        production = make_baseline_config(meta={
            "hot_reload_enabled": False,
            "auto_scaling_enabled": False,
        })

        report = check_p5_identity_alignment(synthetic, production)

        si002_check = next(c for c in report.checks if c.invariant == "SI-002")
        assert si002_check.status == CheckResult.OK

    def test_fm002_combined_with_fm001(self) -> None:
        """FM-002: Hot-reload + config divergence causes BLOCK (not just INVESTIGATE)."""
        synthetic = make_baseline_config(params={"value": 1})
        production = make_baseline_config(
            params={"value": 2},  # Diverged
            meta={"hot_reload_enabled": True},
        )

        report = check_p5_identity_alignment(synthetic, production)

        # Should be BLOCK due to fingerprint mismatch (SI-001 > SI-002)
        assert report.overall_status == CheckResult.BLOCK

        # Both checks should be present
        si001_check = next(c for c in report.checks if c.invariant == "SI-001")
        si002_check = next(c for c in report.checks if c.invariant == "SI-002")
        assert si001_check.status == CheckResult.BLOCK
        assert si002_check.status == CheckResult.INVESTIGATE


# =============================================================================
# FM-003: ENVIRONMENT-SPECIFIC GATES
# =============================================================================


class TestFM003EnvironmentSpecificGates:
    """
    FM-003: Environment-Specific Gates

    Scenario: Production has different feature gates than synthetic
              (e.g., production-only safety gates or disabled experimental features).

    Expected: INVESTIGATE or BLOCK depending on gate criticality
    """

    def test_fm003_gate_difference_triggers_investigate(self) -> None:
        """FM-003: Different gate values trigger gate alignment check."""
        synthetic = make_baseline_config(gates={
            "uplift_gate": True,
            "safety_gate": True,
            "experimental_feature": True,
        })
        production = make_baseline_config(gates={
            "uplift_gate": True,
            "safety_gate": True,
            "experimental_feature": False,  # Disabled in prod
        })

        report = check_p5_identity_alignment(synthetic, production)

        # Gate alignment check should flag this
        gate_check = next((c for c in report.checks if c.name == "Gate Alignment"), None)
        assert gate_check is not None
        assert gate_check.status == CheckResult.INVESTIGATE
        assert "experimental_feature" in gate_check.details

    def test_fm003_nested_gate_difference(self) -> None:
        """FM-003: Nested gate differences are detected."""
        synthetic = make_baseline_config(gates={
            "curriculum": {
                "enabled": True,
                "level": 3,
            },
        })
        production = make_baseline_config(gates={
            "curriculum": {
                "enabled": True,
                "level": 5,  # Different level
            },
        })

        report = check_p5_identity_alignment(synthetic, production)

        gate_check = next((c for c in report.checks if c.name == "Gate Alignment"), None)
        assert gate_check is not None
        assert gate_check.status == CheckResult.INVESTIGATE
        assert "curriculum.level" in gate_check.details

    def test_fm003_missing_gate_in_production(self) -> None:
        """FM-003: Gate present in synthetic but missing in production."""
        synthetic = make_baseline_config(gates={
            "uplift_gate": True,
            "safety_gate": True,
            "debug_mode": True,
        })
        production = make_baseline_config(gates={
            "uplift_gate": True,
            "safety_gate": True,
            # debug_mode missing
        })

        report = check_p5_identity_alignment(synthetic, production)

        gate_check = next((c for c in report.checks if c.name == "Gate Alignment"), None)
        assert gate_check is not None
        assert gate_check.status == CheckResult.INVESTIGATE
        # Should show syn=True prod=None
        assert "debug_mode" in gate_check.details

    def test_fm003_identical_gates_pass(self) -> None:
        """FM-003: Identical gates produce OK."""
        gates = {
            "uplift_gate": True,
            "safety_gate": True,
            "curriculum": {"enabled": True, "level": 3},
        }
        synthetic = make_baseline_config(gates=gates)
        production = make_baseline_config(gates=gates)

        report = check_p5_identity_alignment(synthetic, production)

        gate_check = next((c for c in report.checks if c.name == "Gate Alignment"), None)
        assert gate_check is not None
        assert gate_check.status == CheckResult.OK


# =============================================================================
# FM-004: CURRICULUM VERSION SKEW
# =============================================================================


class TestFM004CurriculumVersionSkew:
    """
    FM-004: Curriculum Version Skew

    Scenario: Production is running a different version of the curriculum
              or slice definition than was validated in P3/P4.

    Expected: INVESTIGATE (minor) or BLOCK (major version difference)
    """

    def test_fm004_major_version_skew_blocks(self) -> None:
        """FM-004: Major version difference triggers SI-006 BLOCK."""
        synthetic = make_baseline_config(version="1.5.0")
        production = make_baseline_config(version="2.0.0")

        report = check_p5_identity_alignment(synthetic, production)

        si006_check = next(c for c in report.checks if c.invariant == "SI-006")
        assert si006_check.status == CheckResult.BLOCK
        assert "MAJOR" in si006_check.details

    def test_fm004_minor_version_skew_investigates(self) -> None:
        """FM-004: Minor version difference triggers SI-006 INVESTIGATE."""
        synthetic = make_baseline_config(version="1.5.0")
        production = make_baseline_config(version="1.6.0")

        report = check_p5_identity_alignment(synthetic, production)

        si006_check = next(c for c in report.checks if c.invariant == "SI-006")
        assert si006_check.status == CheckResult.INVESTIGATE
        assert "Minor" in si006_check.details or "version diff" in si006_check.details.lower()

    def test_fm004_patch_version_skew_investigates(self) -> None:
        """FM-004: Patch version difference triggers SI-006 INVESTIGATE."""
        synthetic = make_baseline_config(version="1.5.0")
        production = make_baseline_config(version="1.5.1")

        report = check_p5_identity_alignment(synthetic, production)

        si006_check = next(c for c in report.checks if c.invariant == "SI-006")
        assert si006_check.status == CheckResult.INVESTIGATE

    def test_fm004_curriculum_fingerprint_mismatch(self) -> None:
        """FM-004: Curriculum fingerprint mismatch triggers SI-004 INVESTIGATE."""
        synthetic = make_baseline_config(curriculum_fp="curriculum_v3_abc123")
        production = make_baseline_config(curriculum_fp="curriculum_v4_def456")

        report = check_p5_identity_alignment(synthetic, production)

        si004_check = next(c for c in report.checks if c.invariant == "SI-004")
        assert si004_check.status == CheckResult.INVESTIGATE
        assert "diverge" in si004_check.details.lower()

    def test_fm004_curriculum_fingerprint_missing_one_side(self) -> None:
        """FM-004: Curriculum fingerprint missing on one side."""
        synthetic = make_baseline_config(curriculum_fp="curriculum_v3_abc123")
        production = make_baseline_config()  # No curriculum_fp

        report = check_p5_identity_alignment(synthetic, production)

        si004_check = next(c for c in report.checks if c.invariant == "SI-004")
        assert si004_check.status == CheckResult.INVESTIGATE
        assert "missing" in si004_check.details.lower()

    def test_fm004_identical_versions_pass(self) -> None:
        """FM-004: Identical versions produce OK."""
        synthetic = make_baseline_config(version="1.5.0", curriculum_fp="fp123")
        production = make_baseline_config(version="1.5.0", curriculum_fp="fp123")

        report = check_p5_identity_alignment(synthetic, production)

        si004_check = next(c for c in report.checks if c.invariant == "SI-004")
        si006_check = next(c for c in report.checks if c.invariant == "SI-006")
        assert si004_check.status == CheckResult.OK
        assert si006_check.status == CheckResult.OK


# =============================================================================
# P4 EVIDENCE PACK INTEGRATION (SI-005)
# =============================================================================


class TestP4EvidencePackIntegration:
    """Tests for P4 evidence pack validation."""

    def test_p4_evidence_fingerprint_match(self) -> None:
        """Production matching P4 evidence baseline is OK."""
        config = make_baseline_config()

        # Run once to get the fingerprint
        initial_report = check_p5_identity_alignment(config, config)
        fp = initial_report.production_fingerprint

        # Create P4 evidence with same fingerprint
        evidence = make_p4_evidence_pack(baseline_fingerprint=fp)

        report = check_p5_identity_alignment(config, config, evidence)

        si005_check = next(c for c in report.checks if c.invariant == "SI-005")
        assert si005_check.status == CheckResult.OK

    def test_p4_evidence_fingerprint_mismatch_blocks(self) -> None:
        """Production differing from P4 evidence baseline is BLOCK."""
        synthetic = make_baseline_config()
        production = make_baseline_config()

        evidence = make_p4_evidence_pack(baseline_fingerprint="different_fingerprint_abc123")

        report = check_p5_identity_alignment(synthetic, production, evidence)

        si005_check = next(c for c in report.checks if c.invariant == "SI-005")
        assert si005_check.status == CheckResult.BLOCK

    def test_p4_evidence_missing_fingerprint_investigates(self) -> None:
        """P4 evidence without baseline fingerprint triggers INVESTIGATE."""
        config = make_baseline_config()
        evidence = make_p4_evidence_pack()  # No fingerprint

        report = check_p5_identity_alignment(config, config, evidence)

        si005_check = next(c for c in report.checks if c.invariant == "SI-005")
        assert si005_check.status == CheckResult.INVESTIGATE

    def test_no_p4_evidence_investigates(self) -> None:
        """No P4 evidence pack triggers INVESTIGATE (not BLOCK)."""
        config = make_baseline_config()

        report = check_p5_identity_alignment(config, config)

        si005_check = next(c for c in report.checks if c.invariant == "SI-005")
        assert si005_check.status == CheckResult.INVESTIGATE


# =============================================================================
# DRIFT GUARD (SI-003)
# =============================================================================


class TestDriftGuard:
    """Tests for drift guard validation (SI-003)."""

    def test_drift_guard_disabled_blocks(self) -> None:
        """Drift guard disabled triggers BLOCK."""
        synthetic = make_baseline_config()
        production = make_baseline_config(meta={
            "drift_guard_enabled": False,
        })

        report = check_p5_identity_alignment(synthetic, production)

        si003_check = next(c for c in report.checks if c.invariant == "SI-003")
        assert si003_check.status == CheckResult.BLOCK
        assert "DISABLED" in si003_check.details

    def test_drift_guard_enabled_ok(self) -> None:
        """Drift guard enabled (default) is OK."""
        config = make_baseline_config()

        report = check_p5_identity_alignment(config, config)

        si003_check = next(c for c in report.checks if c.invariant == "SI-003")
        assert si003_check.status == CheckResult.OK


# =============================================================================
# DIAGNOSE CONFIG DIVERGENCE
# =============================================================================


class TestDiagnoseConfigDivergence:
    """Tests for diagnose_config_divergence function."""

    def test_diagnose_matching_configs(self) -> None:
        """Matching configs return match=True."""
        config = make_baseline_config()
        diagnosis = diagnose_config_divergence(config, config)

        assert diagnosis["match"] is True
        assert diagnosis["synthetic_fingerprint"] == diagnosis["production_fingerprint"]

    def test_diagnose_differing_params(self) -> None:
        """Differing params are identified."""
        synthetic = make_baseline_config(params={"a": 1, "b": 2})
        production = make_baseline_config(params={"a": 1, "b": 3})

        diagnosis = diagnose_config_divergence(synthetic, production)

        assert diagnosis["match"] is False
        assert len(diagnosis["differing_params"]) == 1
        assert diagnosis["differing_params"][0]["param"] == "b"
        assert diagnosis["differing_params"][0]["synthetic"] == 2
        assert diagnosis["differing_params"][0]["production"] == 3

    def test_diagnose_differing_gates(self) -> None:
        """Differing gates are identified."""
        synthetic = make_baseline_config(gates={"x": True, "y": False})
        production = make_baseline_config(gates={"x": True, "y": True})

        diagnosis = diagnose_config_divergence(synthetic, production)

        assert diagnosis["match"] is False
        assert len(diagnosis["differing_gates"]) == 1
        assert diagnosis["differing_gates"][0]["gate"] == "y"

    def test_diagnose_recommends_action(self) -> None:
        """Divergence diagnosis includes recommended action."""
        synthetic = make_baseline_config(params={"a": 1})
        production = make_baseline_config(params={"a": 2})

        diagnosis = diagnose_config_divergence(synthetic, production)

        assert "recommended_action" in diagnosis
        assert "Align" in diagnosis["recommended_action"]


# =============================================================================
# JSON OUTPUT FORMAT
# =============================================================================


class TestJsonOutputFormat:
    """Tests for JSON serialization."""

    def test_to_dict_is_json_serializable(self) -> None:
        """Report to_dict() is JSON serializable."""
        config = make_baseline_config()
        report = check_p5_identity_alignment(config, config)

        # Should not raise
        json_str = json.dumps(report.to_dict())
        data = json.loads(json_str)

        assert "schema" in data
        assert data["schema"] == "p5-identity-alignment-report/1.0.0"
        assert "overall_status" in data
        assert "invariant_summary" in data
        assert "checks" in data

    def test_json_includes_all_invariants(self) -> None:
        """JSON output includes SI-001 through SI-006."""
        config = make_baseline_config()
        report = check_p5_identity_alignment(config, config)
        data = report.to_dict()

        for inv in ["SI-001", "SI-002", "SI-003", "SI-004", "SI-005", "SI-006"]:
            assert inv in data["invariant_summary"]


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_configs(self) -> None:
        """Empty configs don't crash."""
        report = check_p5_identity_alignment({}, {})
        assert report.overall_status in (CheckResult.OK, CheckResult.INVESTIGATE, CheckResult.BLOCK)

    def test_malformed_version_handled(self) -> None:
        """Malformed version strings don't crash."""
        synthetic = make_baseline_config(version="not_a_version")
        production = make_baseline_config(version="1.0.0")

        # Should not raise
        report = check_p5_identity_alignment(synthetic, production)
        assert report.overall_status is not None

    def test_many_differing_params_blocks(self) -> None:
        """More than 2 differing params triggers BLOCK on parameter check."""
        synthetic = make_baseline_config(params={
            "a": 1, "b": 2, "c": 3, "d": 4,
        })
        production = make_baseline_config(params={
            "a": 10, "b": 20, "c": 30, "d": 40,
        })

        report = check_p5_identity_alignment(synthetic, production)

        # Should BLOCK from multiple sources
        assert report.overall_status == CheckResult.BLOCK

        param_check = next((c for c in report.checks if c.name == "Parameter Alignment"), None)
        assert param_check is not None
        assert param_check.status == CheckResult.BLOCK
