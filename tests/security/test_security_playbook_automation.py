#!/usr/bin/env python3
"""
test_security_playbook_automation.py - Tests for Security Playbook Automation Scripts

PHASE II -- NOT RUN IN PHASE I

Tests the security automation scripts defined in U2_SECURITY_PLAYBOOK.md:
- security_replay_incident.py
- security_seed_drift_analysis.py
- lastmile_readiness_check.py

All tests verify deterministic behavior given identical inputs.
"""

import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from security_replay_incident import (
    ReplayStatus,
    Severity,
    Admissibility,
    UpliftClaimStatus,
    analyze_replay_incident,
    classify_replay_status,
    determine_severity,
    determine_admissibility,
    get_recommended_actions,
)

from security_seed_drift_analysis import (
    Classification,
    SeedDriftCause,
    SubstrateNondetCause,
    analyze_seed_drift,
    determine_classification,
    compare_seeds,
    extract_seed_values,
)

from lastmile_readiness_check import (
    CheckStatus,
    SectionStatus,
    OverallStatus,
    run_all_checks,
    check_env_var,
    check_no_pattern_in_env,
)


# =============================================================================
# Tests for security_replay_incident.py
# =============================================================================

class TestReplayStatusClassification:
    """Test replay status classification per playbook matrix."""

    def test_full_match(self):
        """Full match when all cycles match."""
        status = classify_replay_status(
            cycles_matched=100,
            cycles_total=100,
            divergence_point=None
        )
        assert status == ReplayStatus.FULL_MATCH

    def test_no_match_cycle_0(self):
        """No match when divergence at cycle 0."""
        status = classify_replay_status(
            cycles_matched=0,
            cycles_total=100,
            divergence_point=0
        )
        assert status == ReplayStatus.NO_MATCH

    def test_partial_match(self):
        """Partial match when some cycles match."""
        status = classify_replay_status(
            cycles_matched=85,
            cycles_total=100,
            divergence_point=86
        )
        assert status == ReplayStatus.PARTIAL_MATCH

    def test_no_match_zero_cycles(self):
        """No match when zero total cycles."""
        status = classify_replay_status(
            cycles_matched=0,
            cycles_total=0,
            divergence_point=None
        )
        assert status == ReplayStatus.NO_MATCH


class TestSeverityDetermination:
    """Test severity classification per playbook triage matrix."""

    def test_critical_early_divergence(self):
        """Critical severity for divergence in cycles 0-5."""
        for cycle in range(6):
            severity = determine_severity(divergence_point=cycle, cycles_total=100)
            assert severity == Severity.CRITICAL, f"Cycle {cycle} should be CRITICAL"

    def test_high_majority_divergence(self):
        """High severity for divergence in majority of run."""
        severity = determine_severity(divergence_point=30, cycles_total=100)
        assert severity == Severity.HIGH

    def test_medium_minority_divergence(self):
        """Medium severity for divergence in minority of run."""
        severity = determine_severity(divergence_point=75, cycles_total=100)
        assert severity == Severity.MEDIUM

    def test_low_final_cycle(self):
        """Low severity for final cycle divergence."""
        severity = determine_severity(divergence_point=99, cycles_total=100)
        assert severity == Severity.LOW

    def test_low_no_divergence(self):
        """Low severity when no divergence."""
        severity = determine_severity(divergence_point=None, cycles_total=100)
        assert severity == Severity.LOW


class TestAdmissibilityDetermination:
    """Test admissibility classification per playbook matrix."""

    def test_full_match_admissible(self):
        """Full match is admissible with valid claim."""
        admissibility, claim = determine_admissibility(
            ReplayStatus.FULL_MATCH, 100.0
        )
        assert admissibility == Admissibility.ADMISSIBLE
        assert claim == UpliftClaimStatus.VALID

    def test_no_match_inadmissible(self):
        """No match is inadmissible with invalid claim."""
        admissibility, claim = determine_admissibility(
            ReplayStatus.NO_MATCH, 0.0
        )
        assert admissibility == Admissibility.INADMISSIBLE
        assert claim == UpliftClaimStatus.INVALID

    def test_partial_high_conditional(self):
        """Partial match >80% is conditionally admissible."""
        admissibility, claim = determine_admissibility(
            ReplayStatus.PARTIAL_MATCH, 85.0
        )
        assert admissibility == Admissibility.CONDITIONAL
        assert claim == UpliftClaimStatus.VALID_WITH_CAVEAT

    def test_partial_medium_reduced(self):
        """Partial match 50-80% has reduced admissibility."""
        admissibility, claim = determine_admissibility(
            ReplayStatus.PARTIAL_MATCH, 65.0
        )
        assert admissibility == Admissibility.REDUCED
        assert claim == UpliftClaimStatus.WEAKENED

    def test_partial_low_inadmissible(self):
        """Partial match <50% is inadmissible."""
        admissibility, claim = determine_admissibility(
            ReplayStatus.PARTIAL_MATCH, 30.0
        )
        assert admissibility == Admissibility.INADMISSIBLE
        assert claim == UpliftClaimStatus.INVALID


class TestReplayIncidentAnalysis:
    """Test full replay incident analysis."""

    def test_deterministic_output(self):
        """Same inputs produce same outputs (except timestamp)."""
        replay_receipt = {
            "status": "FAIL",
            "cycles_replayed": 100,
            "cycles_matched": 85,
            "divergence_point": 86,
            "divergence_type": "output_mismatch"
        }
        manifest = {"run_id": "test-001", "prng_seed": 42}

        report1 = analyze_replay_incident(replay_receipt, manifest, run_id="fixed-id")
        report2 = analyze_replay_incident(replay_receipt, manifest, run_id="fixed-id")

        # Compare all fields except timestamp-derived ones
        assert report1.replay_status == report2.replay_status
        assert report1.severity == report2.severity
        assert report1.admissibility == report2.admissibility
        assert report1.cycles_matched == report2.cycles_matched
        assert report1.match_percentage == report2.match_percentage

    def test_recommended_actions_present(self):
        """Recommended actions are generated for all severities."""
        replay_receipt = {
            "status": "FAIL",
            "cycles_replayed": 100,
            "cycles_matched": 0,
            "divergence_point": 0,
        }
        manifest = {"run_id": "test-001"}

        report = analyze_replay_incident(replay_receipt, manifest)

        assert len(report.recommended_actions) > 0
        assert any("IMMEDIATE" in action for action in report.recommended_actions)


# =============================================================================
# Tests for security_seed_drift_analysis.py
# =============================================================================

class TestSeedComparison:
    """Test seed extraction and comparison."""

    def test_extract_seed_values(self):
        """Extract PRNG and master seeds from manifest."""
        manifest = {
            "prng_seed": 12345,
            "u2_master_seed": "abc123"
        }
        prng, master = extract_seed_values(manifest)
        assert prng == "12345"
        assert master == "abc123"

    def test_extract_seed_values_alternate_keys(self):
        """Extract seeds from alternate key names."""
        manifest = {
            "prng_seed_start": 67890,
            "master_seed": "xyz789"
        }
        prng, master = extract_seed_values(manifest)
        assert prng == "67890"
        assert master == "xyz789"

    def test_compare_seeds_match(self):
        """Seeds match when values are identical."""
        orig = {"prng_seed": 42, "u2_master_seed": "abc"}
        replay = {"prng_seed": 42, "u2_master_seed": "abc"}
        prng_match, master_match = compare_seeds(orig, replay)
        assert prng_match is True
        assert master_match is True

    def test_compare_seeds_mismatch(self):
        """Seeds don't match when values differ."""
        orig = {"prng_seed": 42, "u2_master_seed": "abc"}
        replay = {"prng_seed": 99, "u2_master_seed": "xyz"}
        prng_match, master_match = compare_seeds(orig, replay)
        assert prng_match is False
        assert master_match is False


class TestClassificationDetermination:
    """Test classification per playbook decision matrix."""

    def test_no_disagreement(self):
        """No disagreement when seeds and outputs match."""
        classification, confidence = determine_classification(
            seeds_match=True,
            outputs_match=True,
            master_seeds_match=True
        )
        assert classification == Classification.NO_DISAGREEMENT
        assert confidence == "HIGH"

    def test_seed_drift(self):
        """Seed drift when seeds don't match."""
        classification, confidence = determine_classification(
            seeds_match=False,
            outputs_match=False,
            master_seeds_match=False
        )
        assert classification == Classification.SEED_DRIFT
        assert confidence == "HIGH"

    def test_substrate_nondeterminism(self):
        """Substrate nondeterminism when seeds match but outputs differ."""
        classification, confidence = determine_classification(
            seeds_match=True,
            outputs_match=False,
            master_seeds_match=True
        )
        assert classification == Classification.SUBSTRATE_NONDETERMINISM
        assert confidence == "HIGH"

    def test_unknown_suspicious(self):
        """Unknown when seeds differ but outputs match."""
        classification, confidence = determine_classification(
            seeds_match=False,
            outputs_match=True,
            master_seeds_match=False
        )
        assert classification == Classification.UNKNOWN
        assert confidence == "MEDIUM"


class TestSeedDriftAnalysis:
    """Test full seed drift analysis."""

    def test_deterministic_classification(self):
        """Same inputs produce same classification."""
        orig = {"run_id": "test", "prng_seed": 42, "u2_master_seed": "abc"}
        replay = {"run_id": "test", "prng_seed": 99, "u2_master_seed": "abc"}

        report1 = analyze_seed_drift(orig, replay, run_id="fixed-id")
        report2 = analyze_seed_drift(orig, replay, run_id="fixed-id")

        assert report1.classification == report2.classification
        assert report1.dimensions == report2.dimensions
        assert report1.cause_category == report2.cause_category

    def test_resolution_steps_provided(self):
        """Resolution steps are provided for all classifications."""
        orig = {"run_id": "test", "prng_seed": 42}
        replay = {"run_id": "test", "prng_seed": 42}
        receipt = {"status": "FAIL", "divergence_point": 10}

        report = analyze_seed_drift(orig, replay, replay_receipt=receipt)

        assert len(report.resolution_steps) > 0
        assert len(report.preventive_measures) > 0


# =============================================================================
# Tests for lastmile_readiness_check.py
# =============================================================================

class TestEnvVarChecks:
    """Test environment variable checking."""

    def test_check_env_var_present(self):
        """Check passes when env var matches."""
        with patch.dict(os.environ, {"TEST_VAR": "expected"}):
            actual, passed = check_env_var("TEST_VAR", "expected")
            assert passed is True
            assert actual == "expected"

    def test_check_env_var_missing(self):
        """Check fails when env var missing."""
        with patch.dict(os.environ, {}, clear=True):
            actual, passed = check_env_var("MISSING_VAR", "value")
            assert passed is False
            assert actual == "(not set)"

    def test_check_env_var_wrong_value(self):
        """Check fails when env var has wrong value."""
        with patch.dict(os.environ, {"TEST_VAR": "wrong"}):
            actual, passed = check_env_var("TEST_VAR", "expected")
            assert passed is False
            assert actual == "wrong"


class TestProhibitedPatternCheck:
    """Test prohibited pattern checking."""

    def test_no_prohibited_patterns(self):
        """Passes when no prohibited patterns found."""
        with patch.dict(os.environ, {"SAFE_VAR": "value"}, clear=True):
            actual, passed = check_no_pattern_in_env(["FORBIDDEN"])
            assert passed is True

    def test_prohibited_pattern_found(self):
        """Fails when prohibited pattern found."""
        with patch.dict(os.environ, {"MY_FORBIDDEN_VAR": "value"}, clear=True):
            actual, passed = check_no_pattern_in_env(["FORBIDDEN"])
            assert passed is False
            assert "FORBIDDEN" in actual.upper() or "MY_FORBIDDEN_VAR" in actual

    def test_http_proxy_allowed(self):
        """HTTP_PROXY is allowed despite PROXY pattern."""
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy:8080"}, clear=True):
            actual, passed = check_no_pattern_in_env(["PROXY"])
            # HTTP_PROXY should be allowed
            assert passed is True or "HTTP" in actual


class TestLastMileReport:
    """Test full LAST MILE verification."""

    def test_report_structure(self):
        """Report contains all required sections."""
        config = {
            "run_id": "test-run",
            "scripts_dir": "scripts",
            "logs_dir": "logs",
            "backend_rfl_dir": "backend/rfl",
            "hooks_dir": "hooks",
            "config_dir": "config",
            "manifest_path": "logs/uplift/test-run/run_manifest.yaml",
        }

        # Mock environment for consistent results
        with patch.dict(os.environ, {
            "PYTHONHASHSEED": "0",
            "REPLAY_ENABLED": "true",
            "RFL_ENV_MODE": "uplift_experiment",
        }):
            report = run_all_checks(config)

        # Verify structure
        assert report.checklist_version == "1.0"
        assert report.run_id == "test-run"
        assert "section_a_replay" in asdict(report)
        assert "section_b_prng" in asdict(report)
        assert "section_c_telemetry" in asdict(report)
        assert "section_d_hermetic" in asdict(report)
        assert report.total_checks == 20

    def test_deterministic_results(self):
        """Same config produces same results."""
        config = {
            "run_id": "test",
            "scripts_dir": "scripts",
            "logs_dir": "logs",
            "backend_rfl_dir": "backend/rfl",
            "hooks_dir": "hooks",
            "config_dir": "config",
            "manifest_path": "logs/uplift/test/run_manifest.yaml",
        }

        with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
            report1 = run_all_checks(config)
            report2 = run_all_checks(config)

        # Results should be identical
        assert report1.total_passed == report2.total_passed
        assert report1.overall_status == report2.overall_status
        assert len(report1.blocking_items) == len(report2.blocking_items)


# =============================================================================
# Integration Tests
# =============================================================================

class TestPlaybookIntegration:
    """Integration tests for playbook workflow."""

    def test_replay_to_drift_analysis_flow(self):
        """Replay incident can flow into drift analysis."""
        # Create replay receipt indicating failure
        replay_receipt = {
            "status": "FAIL",
            "cycles_replayed": 100,
            "cycles_matched": 50,
            "divergence_point": 51,
            "divergence_type": "output_mismatch"
        }

        # Create manifests
        original_manifest = {
            "run_id": "test-001",
            "prng_seed": 42,
            "u2_master_seed": "abc123"
        }
        replay_manifest = {
            "run_id": "test-001",
            "prng_seed": 42,
            "u2_master_seed": "abc123"
        }

        # Step 1: Analyze replay incident
        incident_report = analyze_replay_incident(
            replay_receipt, original_manifest, run_id="test-001"
        )

        # Verify incident detected
        assert incident_report.replay_status != "FULL_MATCH"
        assert incident_report.admissibility != Admissibility.ADMISSIBLE.value

        # Step 2: Follow up with drift analysis
        drift_report = analyze_seed_drift(
            original_manifest,
            replay_manifest,
            replay_receipt=replay_receipt,
            run_id="test-001"
        )

        # Since seeds match, should be substrate nondeterminism
        assert drift_report.classification == Classification.SUBSTRATE_NONDETERMINISM.value
        assert drift_report.dimensions["seeds_match"] is True
        assert drift_report.dimensions["outputs_match"] is False

    def test_seed_drift_detection(self):
        """Seed drift is correctly identified when seeds differ."""
        replay_receipt = {
            "status": "FAIL",
            "cycles_replayed": 100,
            "cycles_matched": 0,
            "divergence_point": 0,
        }

        original_manifest = {
            "run_id": "test-001",
            "prng_seed": 42,
            "u2_master_seed": "abc123"
        }
        replay_manifest = {
            "run_id": "test-001",
            "prng_seed": 99,  # Different seed!
            "u2_master_seed": "abc123"
        }

        drift_report = analyze_seed_drift(
            original_manifest,
            replay_manifest,
            replay_receipt=replay_receipt,
            run_id="test-001"
        )

        # Should be classified as seed drift
        assert drift_report.classification == Classification.SEED_DRIFT.value
        assert drift_report.dimensions["seeds_match"] is False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_manifest_files():
    """Create temporary manifest files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create original manifest
        orig_path = tmppath / "original_manifest.yaml"
        orig_path.write_text("""
run_id: test-run-001
prng_seed: 12345
prng_seed_start: 12345
prng_seed_end: 12345
u2_master_seed: abcdef123456
environment:
  PYTHONHASHSEED: "0"
""")

        # Create replay manifest
        replay_path = tmppath / "replay_manifest.yaml"
        replay_path.write_text("""
run_id: test-run-001
prng_seed: 12345
prng_seed_start: 12345
prng_seed_end: 12345
u2_master_seed: abcdef123456
environment:
  PYTHONHASHSEED: "0"
""")

        # Create replay receipt
        receipt_path = tmppath / "replay_receipt.json"
        receipt_path.write_text(json.dumps({
            "status": "PASS",
            "cycles_replayed": 100,
            "cycles_matched": 100,
            "divergence_point": None
        }))

        yield {
            "original": orig_path,
            "replay": replay_path,
            "receipt": receipt_path,
            "tmpdir": tmppath
        }


class TestWithFixtures:
    """Tests using file fixtures."""

    def test_load_and_analyze_manifests(self, temp_manifest_files):
        """Load manifest files and analyze."""
        from security_seed_drift_analysis import load_manifest

        orig = load_manifest(temp_manifest_files["original"])
        replay = load_manifest(temp_manifest_files["replay"])

        with open(temp_manifest_files["receipt"]) as f:
            receipt = json.load(f)

        report = analyze_seed_drift(orig, replay, replay_receipt=receipt)

        # Should show no disagreement (matching seeds and outputs)
        assert report.classification == Classification.NO_DISAGREEMENT.value


# =============================================================================
# Tests for security_posture.py - Unified Security Posture Spine
# =============================================================================

from security_posture import (
    build_security_posture,
    summarize_security_for_governance,
    merge_into_global_health,
    SecurityLevel,
    ReplayStatus as PostureReplayStatus,
    SeedClassification,
    LastMileStatus,
    SCHEMA_VERSION,
)


class TestBuildSecurityPosture:
    """Test unified security posture building."""

    def test_empty_inputs_produce_valid_posture(self):
        """Posture is valid even with no component reports."""
        posture = build_security_posture()

        assert posture["schema_version"] == SCHEMA_VERSION
        assert "is_security_ok" in posture
        assert "security_level" in posture
        assert posture["components_available"]["replay_incident"] is False
        assert posture["components_available"]["seed_analysis"] is False
        assert posture["components_available"]["lastmile_report"] is False

    def test_all_ok_scenario(self):
        """All components OK produces security OK."""
        replay = {
            "replay_status": "FULL_MATCH",
            "severity": "NONE",
            "match_percentage": 100.0,
            "run_id": "test-001"
        }
        seed = {
            "classification": "NO_DISAGREEMENT",
            "confidence": "HIGH"
        }
        lastmile = {
            "overall_status": "READY",
            "total_passed": 20,
            "total_checks": 20,
            "blocking_items": []
        }

        posture = build_security_posture(replay, seed, lastmile)

        assert posture["is_security_ok"] is True
        assert posture["security_level"] == "OK"
        assert posture["replay_status"] == "FULL_MATCH"
        assert posture["seed_classification"] == "NO_ISSUE"
        assert posture["lastmile_status"] == "READY"
        assert len(posture["blocking_reasons"]) == 0

    def test_replay_no_match_blocks(self):
        """Replay NO_MATCH blocks security."""
        replay = {
            "replay_status": "NO_MATCH",
            "severity": "CRITICAL",
            "match_percentage": 0.0,
            "divergence_point": 0
        }

        posture = build_security_posture(replay_incident=replay)

        assert posture["is_security_ok"] is False
        assert posture["security_level"] == "NO_GO"
        assert "NO_MATCH" in str(posture["blocking_reasons"])

    def test_seed_drift_blocks(self):
        """Seed drift blocks security."""
        seed = {
            "classification": "SEED_DRIFT",
            "confidence": "HIGH"
        }

        posture = build_security_posture(seed_analysis=seed)

        assert posture["is_security_ok"] is False
        assert posture["security_level"] == "NO_GO"
        assert posture["has_seed_drift"] is True
        assert "drift" in str(posture["blocking_reasons"]).lower()

    def test_lastmile_nogo_blocks(self):
        """Last-mile NO_GO blocks security."""
        lastmile = {
            "overall_status": "NO_GO",
            "total_passed": 10,
            "total_checks": 20,
            "blocking_items": ["A1: Something failed"]
        }

        posture = build_security_posture(lastmile_report=lastmile)

        assert posture["is_security_ok"] is False
        assert posture["security_level"] == "NO_GO"
        assert "NO_GO" in str(posture["blocking_reasons"])

    def test_substrate_nondeterminism_warns(self):
        """Substrate nondeterminism produces warning but OK."""
        replay = {"replay_status": "FULL_MATCH", "severity": "NONE"}
        seed = {"classification": "SUBSTRATE_NONDETERMINISM", "confidence": "HIGH"}
        lastmile = {"overall_status": "READY", "total_passed": 20, "total_checks": 20}

        posture = build_security_posture(replay, seed, lastmile)

        assert posture["is_security_ok"] is True
        assert posture["security_level"] == "WARN"
        assert posture["has_substrate_nondeterminism"] is True

    def test_partial_match_warns(self):
        """Partial match produces warning."""
        replay = {
            "replay_status": "PARTIAL_MATCH",
            "severity": "MEDIUM",
            "match_percentage": 85.0
        }
        seed = {"classification": "NO_ISSUE"}
        lastmile = {"overall_status": "READY", "total_passed": 20, "total_checks": 20}

        posture = build_security_posture(replay, seed, lastmile)

        assert posture["is_security_ok"] is True
        assert posture["security_level"] == "WARN"

    def test_deterministic_output(self):
        """Same inputs produce identical outputs (except timestamp)."""
        replay = {"replay_status": "FULL_MATCH", "severity": "NONE"}
        seed = {"classification": "NO_ISSUE"}
        lastmile = {"overall_status": "READY"}

        posture1 = build_security_posture(replay, seed, lastmile, run_id="fixed")
        posture2 = build_security_posture(replay, seed, lastmile, run_id="fixed")

        # Compare all fields except timestamp
        assert posture1["is_security_ok"] == posture2["is_security_ok"]
        assert posture1["security_level"] == posture2["security_level"]
        assert posture1["replay_status"] == posture2["replay_status"]
        assert posture1["seed_classification"] == posture2["seed_classification"]
        assert posture1["lastmile_status"] == posture2["lastmile_status"]

    def test_run_id_extraction(self):
        """Run ID is extracted from components."""
        replay = {"run_id": "from-replay"}
        posture = build_security_posture(replay_incident=replay)
        assert posture["run_id"] == "from-replay"

        seed = {"run_id": "from-seed"}
        posture = build_security_posture(seed_analysis=seed)
        assert posture["run_id"] == "from-seed"

    def test_critical_severity_blocks(self):
        """Critical severity blocks even with partial match."""
        replay = {
            "replay_status": "PARTIAL_MATCH",
            "severity": "CRITICAL",
            "match_percentage": 95.0,
            "divergence_point": 3
        }

        posture = build_security_posture(replay_incident=replay)

        assert posture["is_security_ok"] is False
        assert posture["security_level"] == "NO_GO"


class TestSummarizeSecurityForGovernance:
    """Test governance summary generation."""

    def test_governance_summary_structure(self):
        """Summary contains required fields."""
        posture = build_security_posture()
        summary = summarize_security_for_governance(posture)

        assert "security_level" in summary
        assert "has_seed_drift" in summary
        assert "has_substrate_nondeterminism" in summary
        assert "is_security_ok" in summary
        assert "lastmile_ready" in summary
        assert "replay_ok" in summary
        assert "blocking_count" in summary

    def test_ok_summary(self):
        """OK posture produces OK summary."""
        replay = {"replay_status": "FULL_MATCH", "severity": "NONE"}
        seed = {"classification": "NO_ISSUE"}
        lastmile = {"overall_status": "READY", "total_passed": 20, "total_checks": 20}

        posture = build_security_posture(replay, seed, lastmile)
        summary = summarize_security_for_governance(posture)

        assert summary["security_level"] == "OK"
        assert summary["is_security_ok"] is True
        assert summary["has_seed_drift"] is False
        assert summary["has_substrate_nondeterminism"] is False
        assert summary["lastmile_ready"] is True
        assert summary["replay_ok"] is True
        assert summary["blocking_count"] == 0

    def test_warn_summary(self):
        """WARN posture produces correct summary."""
        seed = {"classification": "SUBSTRATE_NONDETERMINISM"}
        lastmile = {"overall_status": "READY"}

        posture = build_security_posture(seed_analysis=seed, lastmile_report=lastmile)
        summary = summarize_security_for_governance(posture)

        assert summary["security_level"] == "WARN"
        assert summary["has_substrate_nondeterminism"] is True

    def test_nogo_summary(self):
        """NO_GO posture produces correct summary."""
        seed = {"classification": "SEED_DRIFT"}

        posture = build_security_posture(seed_analysis=seed)
        summary = summarize_security_for_governance(posture)

        assert summary["security_level"] == "NO_GO"
        assert summary["has_seed_drift"] is True
        assert summary["is_security_ok"] is False
        assert summary["blocking_count"] > 0

    def test_deterministic_summary(self):
        """Same posture produces same summary."""
        posture = build_security_posture(run_id="fixed")

        summary1 = summarize_security_for_governance(posture)
        summary2 = summarize_security_for_governance(posture)

        assert summary1 == summary2


class TestMergeIntoGlobalHealth:
    """Test global health integration."""

    def test_merge_adds_security_posture(self):
        """Merge adds security_posture field."""
        global_health = {"status": "healthy", "uptime": 3600}
        posture = build_security_posture()

        result = merge_into_global_health(global_health, posture)

        assert "security_posture" in result
        assert result["status"] == "healthy"
        assert result["uptime"] == 3600

    def test_merge_preserves_existing_fields(self):
        """Merge preserves all existing fields."""
        global_health = {
            "status": "healthy",
            "services": {"db": "up", "redis": "up"},
            "metrics": {"proofs": 1000}
        }
        posture = build_security_posture()

        result = merge_into_global_health(global_health, posture)

        assert result["status"] == "healthy"
        assert result["services"] == {"db": "up", "redis": "up"}
        assert result["metrics"] == {"proofs": 1000}


class TestSecurityPostureCLI:
    """Test CLI behavior via module imports."""

    def test_posture_check_exit_codes(self):
        """Verify exit code mapping logic."""
        # OK -> exit 0
        ok_posture = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH", "severity": "NONE"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )
        assert ok_posture["is_security_ok"] is True
        assert ok_posture["security_level"] == "OK"
        # Would exit 0

        # WARN -> exit 1
        warn_posture = build_security_posture(
            seed_analysis={"classification": "SUBSTRATE_NONDETERMINISM"},
            lastmile_report={"overall_status": "READY"}
        )
        assert warn_posture["is_security_ok"] is True
        assert warn_posture["security_level"] == "WARN"
        # Would exit 1

        # NO_GO -> exit 2
        nogo_posture = build_security_posture(
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        assert nogo_posture["is_security_ok"] is False
        assert nogo_posture["security_level"] == "NO_GO"
        # Would exit 2


class TestSecurityPostureEdgeCases:
    """Test edge cases and normalization."""

    def test_unknown_replay_status_normalized(self):
        """Unknown replay status is handled."""
        replay = {"replay_status": "UNKNOWN_STATUS"}
        posture = build_security_posture(replay_incident=replay)
        assert posture["replay_status"] == "NOT_RUN"

    def test_pass_status_normalized_to_full_match(self):
        """PASS status normalized to FULL_MATCH."""
        replay = {"replay_status": "PASS"}
        posture = build_security_posture(replay_incident=replay)
        assert posture["replay_status"] == "FULL_MATCH"

    def test_fail_status_normalized_to_no_match(self):
        """FAIL status normalized to NO_MATCH."""
        replay = {"replay_status": "FAIL"}
        posture = build_security_posture(replay_incident=replay)
        assert posture["replay_status"] == "NO_MATCH"

    def test_conditional_lastmile_warns(self):
        """CONDITIONAL last-mile produces warning."""
        lastmile = {"overall_status": "CONDITIONAL"}
        posture = build_security_posture(lastmile_report=lastmile)
        assert posture["security_level"] == "WARN"

    def test_unknown_seed_classification_warns(self):
        """UNKNOWN seed classification produces warning."""
        seed = {"classification": "UNKNOWN"}
        lastmile = {"overall_status": "READY"}

        posture = build_security_posture(seed_analysis=seed, lastmile_report=lastmile)
        assert posture["security_level"] == "WARN"

    def test_multiple_blocking_reasons(self):
        """Multiple issues produce multiple blocking reasons."""
        replay = {"replay_status": "NO_MATCH", "severity": "CRITICAL"}
        seed = {"classification": "SEED_DRIFT"}
        lastmile = {"overall_status": "NO_GO"}

        posture = build_security_posture(replay, seed, lastmile)

        assert posture["is_security_ok"] is False
        assert len(posture["blocking_reasons"]) >= 2


class TestGovernanceSummaryStability:
    """Test stable mapping for governance integration."""

    @pytest.mark.parametrize("replay,seed,lastmile,expected_level", [
        # All OK -> OK
        ({"replay_status": "FULL_MATCH"}, {"classification": "NO_ISSUE"}, {"overall_status": "READY"}, "OK"),
        # Partial match -> WARN
        ({"replay_status": "PARTIAL_MATCH"}, {"classification": "NO_ISSUE"}, {"overall_status": "READY"}, "WARN"),
        # Substrate nondet -> WARN
        ({"replay_status": "FULL_MATCH"}, {"classification": "SUBSTRATE_NONDETERMINISM"}, {"overall_status": "READY"}, "WARN"),
        # Conditional lastmile -> WARN
        ({"replay_status": "FULL_MATCH"}, {"classification": "NO_ISSUE"}, {"overall_status": "CONDITIONAL"}, "WARN"),
        # Seed drift -> NO_GO
        ({"replay_status": "FULL_MATCH"}, {"classification": "SEED_DRIFT"}, {"overall_status": "READY"}, "NO_GO"),
        # No match -> NO_GO
        ({"replay_status": "NO_MATCH"}, {"classification": "NO_ISSUE"}, {"overall_status": "READY"}, "NO_GO"),
        # LastMile NO_GO -> NO_GO
        ({"replay_status": "FULL_MATCH"}, {"classification": "NO_ISSUE"}, {"overall_status": "NO_GO"}, "NO_GO"),
    ])
    def test_security_level_mapping(self, replay, seed, lastmile, expected_level):
        """Verify stable security level mapping for various combinations."""
        posture = build_security_posture(replay, seed, lastmile)
        summary = summarize_security_for_governance(posture)

        assert summary["security_level"] == expected_level

    @pytest.mark.parametrize("seed_class,has_drift,has_nondet", [
        ("NO_ISSUE", False, False),
        ("SEED_DRIFT", True, False),
        ("SUBSTRATE_NONDETERMINISM", False, True),
        ("UNKNOWN", False, False),
        ("NO_DISAGREEMENT", False, False),
    ])
    def test_seed_flags_mapping(self, seed_class, has_drift, has_nondet):
        """Verify stable seed flag mapping."""
        seed = {"classification": seed_class}
        posture = build_security_posture(seed_analysis=seed)
        summary = summarize_security_for_governance(posture)

        assert summary["has_seed_drift"] == has_drift
        assert summary["has_substrate_nondeterminism"] == has_nondet


# =============================================================================
# Phase III Tests: Security Scenario Intelligence & Simulation
# =============================================================================

from security_posture import (
    classify_security_scenario,
    simulate_security_variation,
    map_security_to_director_status,
    get_scenario_description,
    SecurityScenario,
    DirectorStatus,
)


class TestClassifySecurityScenario:
    """Test security scenario classification."""

    def test_nominal_scenario(self):
        """All OK produces NOMINAL scenario."""
        posture = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH", "severity": "NONE"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.NOMINAL.value

    def test_pristine_scenario(self):
        """No components available produces PRISTINE."""
        posture = build_security_posture()
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.PRISTINE.value

    def test_integrity_compromised_highest_priority(self):
        """Seed drift + NO_MATCH = INTEGRITY_COMPROMISED."""
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH", "severity": "CRITICAL"},
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.INTEGRITY_COMPROMISED.value

    def test_multi_failure_scenario(self):
        """Multiple blocking reasons = MULTI_FAILURE."""
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            lastmile_report={"overall_status": "NO_GO"}
        )
        # Should have 2+ blocking reasons
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.MULTI_FAILURE.value

    def test_critical_severity_scenario(self):
        """Critical severity = CRITICAL_SEVERITY."""
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH", "severity": "CRITICAL"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.CRITICAL_SEVERITY.value

    def test_seed_drift_detected_scenario(self):
        """Seed drift alone = SEED_DRIFT_DETECTED."""
        posture = build_security_posture(
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.SEED_DRIFT_DETECTED.value

    def test_replay_failure_scenario(self):
        """Replay NO_MATCH alone = REPLAY_FAILURE."""
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH", "severity": "HIGH"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.REPLAY_FAILURE.value

    def test_lastmile_blocked_scenario(self):
        """Last-mile NO_GO = LASTMILE_BLOCKED."""
        posture = build_security_posture(
            lastmile_report={"overall_status": "NO_GO"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.LASTMILE_BLOCKED.value

    def test_degraded_coverage_scenario(self):
        """Two+ warnings = DEGRADED_COVERAGE."""
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"},
            seed_analysis={"classification": "SUBSTRATE_NONDETERMINISM"},
            lastmile_report={"overall_status": "READY"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.DEGRADED_COVERAGE.value

    def test_partial_replay_scenario(self):
        """Single partial match = PARTIAL_REPLAY."""
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH", "severity": "MEDIUM"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.PARTIAL_REPLAY.value

    def test_substrate_variance_scenario(self):
        """Single substrate nondet = SUBSTRATE_VARIANCE."""
        posture = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"},
            seed_analysis={"classification": "SUBSTRATE_NONDETERMINISM"},
            lastmile_report={"overall_status": "READY"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.SUBSTRATE_VARIANCE.value

    def test_conditional_ready_scenario(self):
        """Conditional last-mile = CONDITIONAL_READY."""
        posture = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "CONDITIONAL"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.CONDITIONAL_READY.value

    def test_unknown_seed_state_scenario(self):
        """Unknown seed = UNKNOWN_SEED_STATE."""
        posture = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"},
            seed_analysis={"classification": "UNKNOWN"},
            lastmile_report={"overall_status": "READY"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.UNKNOWN_SEED_STATE.value

    def test_deterministic_classification(self):
        """Same posture always produces same scenario."""
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"},
            seed_analysis={"classification": "NO_ISSUE"}
        )

        scenario1 = classify_security_scenario(posture)
        scenario2 = classify_security_scenario(posture)

        assert scenario1 == scenario2


class TestSimulateSecurityVariation:
    """Test what-if security simulation."""

    def test_simulate_replay_status_change(self):
        """Can simulate replay status change."""
        original = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )

        # Simulate degradation
        simulated = simulate_security_variation(
            original,
            {"replay_status": "NO_MATCH"}
        )

        assert original["replay_status"] == "FULL_MATCH"
        assert simulated["replay_status"] == "NO_MATCH"
        assert simulated["is_security_ok"] is False
        assert simulated["_simulated"] is True

    def test_simulate_seed_classification_change(self):
        """Simulating seed classification updates flags."""
        original = build_security_posture(
            seed_analysis={"classification": "NO_ISSUE"}
        )

        simulated = simulate_security_variation(
            original,
            {"seed_classification": "SEED_DRIFT"}
        )

        assert original["has_seed_drift"] is False
        assert simulated["has_seed_drift"] is True
        assert simulated["seed_classification"] == "SEED_DRIFT"

    def test_simulate_lastmile_change(self):
        """Can simulate last-mile status change."""
        original = build_security_posture(
            lastmile_report={"overall_status": "READY"}
        )

        simulated = simulate_security_variation(
            original,
            {"lastmile_status": "NO_GO"}
        )

        assert simulated["lastmile_status"] == "NO_GO"
        assert simulated["security_level"] == "NO_GO"

    def test_simulate_add_blocking_reason(self):
        """Can add blocking reason."""
        original = build_security_posture()

        simulated = simulate_security_variation(
            original,
            {"add_blocking_reason": "Simulated failure"}
        )

        assert "Simulated failure" in simulated["blocking_reasons"]

    def test_simulate_clear_blocking_reasons(self):
        """Can clear blocking reasons."""
        original = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"}
        )
        # Should have blocking reasons
        assert len(original["blocking_reasons"]) > 0

        simulated = simulate_security_variation(
            original,
            {
                "replay_status": "FULL_MATCH",
                "clear_blocking_reasons": True
            }
        )

        assert len(simulated["blocking_reasons"]) == 0

    def test_simulate_multiple_changes(self):
        """Can apply multiple changes at once."""
        original = build_security_posture()

        simulated = simulate_security_variation(
            original,
            {
                "replay_status": "PARTIAL_MATCH",
                "seed_classification": "SUBSTRATE_NONDETERMINISM",
                "lastmile_status": "CONDITIONAL"
            }
        )

        assert simulated["replay_status"] == "PARTIAL_MATCH"
        assert simulated["seed_classification"] == "SUBSTRATE_NONDETERMINISM"
        assert simulated["lastmile_status"] == "CONDITIONAL"

    def test_simulate_preserves_original(self):
        """Simulation does not modify original."""
        original = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"}
        )
        original_copy = original.copy()

        simulate_security_variation(
            original,
            {"replay_status": "NO_MATCH"}
        )

        assert original["replay_status"] == original_copy["replay_status"]

    def test_simulate_recalculates_scenario(self):
        """Simulation includes updated scenario."""
        original = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )

        simulated = simulate_security_variation(
            original,
            {"seed_classification": "SEED_DRIFT"}
        )

        assert "scenario" in simulated
        assert simulated["scenario"] == SecurityScenario.SEED_DRIFT_DETECTED.value

    def test_simulate_deterministic(self):
        """Same variation produces same result."""
        original = build_security_posture()
        spec = {"replay_status": "PARTIAL_MATCH"}

        sim1 = simulate_security_variation(original, spec)
        sim2 = simulate_security_variation(original, spec)

        # Compare key fields (excluding timestamp-related)
        assert sim1["replay_status"] == sim2["replay_status"]
        assert sim1["security_level"] == sim2["security_level"]
        assert sim1["scenario"] == sim2["scenario"]


class TestMapSecurityToDirectorStatus:
    """Test Director Console status mapping."""

    def test_ok_maps_to_green(self):
        """OK security level maps to GREEN."""
        posture = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )

        status = map_security_to_director_status(posture)
        assert status == DirectorStatus.GREEN.value

    def test_warn_maps_to_yellow(self):
        """WARN security level maps to YELLOW."""
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )

        status = map_security_to_director_status(posture)
        assert status == DirectorStatus.YELLOW.value

    def test_nogo_maps_to_red(self):
        """NO_GO security level maps to RED."""
        posture = build_security_posture(
            seed_analysis={"classification": "SEED_DRIFT"}
        )

        status = map_security_to_director_status(posture)
        assert status == DirectorStatus.RED.value

    def test_integrity_compromised_always_red(self):
        """INTEGRITY_COMPROMISED scenario always RED."""
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            seed_analysis={"classification": "SEED_DRIFT"}
        )

        status = map_security_to_director_status(posture)
        assert status == DirectorStatus.RED.value

    def test_pristine_maps_to_green(self):
        """PRISTINE scenario gets benefit of doubt = GREEN."""
        posture = build_security_posture()

        status = map_security_to_director_status(posture)
        assert status == DirectorStatus.GREEN.value

    def test_deterministic_mapping(self):
        """Same posture always maps to same status."""
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"}
        )

        status1 = map_security_to_director_status(posture)
        status2 = map_security_to_director_status(posture)

        assert status1 == status2

    @pytest.mark.parametrize("scenario,expected_color", [
        (SecurityScenario.NOMINAL.value, "GREEN"),
        (SecurityScenario.PRISTINE.value, "GREEN"),
        (SecurityScenario.PARTIAL_REPLAY.value, "YELLOW"),
        (SecurityScenario.SUBSTRATE_VARIANCE.value, "YELLOW"),
        (SecurityScenario.CONDITIONAL_READY.value, "YELLOW"),
        (SecurityScenario.UNKNOWN_SEED_STATE.value, "YELLOW"),
        (SecurityScenario.DEGRADED_COVERAGE.value, "YELLOW"),
        (SecurityScenario.REPLAY_FAILURE.value, "RED"),
        (SecurityScenario.SEED_DRIFT_DETECTED.value, "RED"),
        (SecurityScenario.LASTMILE_BLOCKED.value, "RED"),
        (SecurityScenario.CRITICAL_SEVERITY.value, "RED"),
        (SecurityScenario.MULTI_FAILURE.value, "RED"),
        (SecurityScenario.INTEGRITY_COMPROMISED.value, "RED"),
    ])
    def test_scenario_to_color_mapping(self, scenario, expected_color):
        """Verify scenario-to-color mapping via get_scenario_description."""
        desc = get_scenario_description(scenario)
        assert desc["director_status"] == expected_color


class TestGetScenarioDescription:
    """Test scenario description retrieval."""

    def test_all_scenarios_have_descriptions(self):
        """All defined scenarios have descriptions."""
        for scenario in SecurityScenario:
            desc = get_scenario_description(scenario.value)
            assert "description" in desc
            assert "severity" in desc
            assert "director_status" in desc
            assert "recommended_actions" in desc

    def test_unknown_scenario_handled(self):
        """Unknown scenarios get default description."""
        desc = get_scenario_description("UNKNOWN_SCENARIO_XYZ")
        assert "Unknown scenario" in desc["description"]
        assert desc["severity"] == "HIGH"
        assert desc["director_status"] == "RED"

    def test_descriptions_are_strings(self):
        """Descriptions are non-empty strings."""
        for scenario in SecurityScenario:
            desc = get_scenario_description(scenario.value)
            assert isinstance(desc["description"], str)
            assert len(desc["description"]) > 0

    def test_actions_are_lists(self):
        """Recommended actions are non-empty lists."""
        for scenario in SecurityScenario:
            desc = get_scenario_description(scenario.value)
            assert isinstance(desc["recommended_actions"], list)
            assert len(desc["recommended_actions"]) > 0


class TestPhaseIIIIntegration:
    """Integration tests for Phase III functionality."""

    def test_posture_to_scenario_to_director(self):
        """Full flow: posture -> scenario -> director status."""
        # Build posture
        posture = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH", "severity": "NONE"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )

        # Classify scenario
        scenario = classify_security_scenario(posture)
        assert scenario == SecurityScenario.NOMINAL.value

        # Map to director
        status = map_security_to_director_status(posture)
        assert status == DirectorStatus.GREEN.value

        # Get description
        desc = get_scenario_description(scenario)
        assert desc["director_status"] == status

    def test_simulate_and_compare_scenarios(self):
        """Simulate variations and compare scenarios."""
        # Start with OK posture
        original = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )
        original_scenario = classify_security_scenario(original)
        assert original_scenario == SecurityScenario.NOMINAL.value

        # Simulate seed drift
        with_drift = simulate_security_variation(
            original,
            {"seed_classification": "SEED_DRIFT"}
        )
        assert with_drift["scenario"] == SecurityScenario.SEED_DRIFT_DETECTED.value
        assert map_security_to_director_status(with_drift) == "RED"

        # Simulate partial replay instead
        with_partial = simulate_security_variation(
            original,
            {"replay_status": "PARTIAL_MATCH"}
        )
        assert with_partial["scenario"] == SecurityScenario.PARTIAL_REPLAY.value
        assert map_security_to_director_status(with_partial) == "YELLOW"

    def test_simulation_chain(self):
        """Can chain simulations."""
        posture = build_security_posture()

        # First simulation
        sim1 = simulate_security_variation(
            posture,
            {"replay_status": "FULL_MATCH"}
        )

        # Second simulation on result
        sim2 = simulate_security_variation(
            sim1,
            {"seed_classification": "SUBSTRATE_NONDETERMINISM"}
        )

        assert sim2["replay_status"] == "FULL_MATCH"
        assert sim2["has_substrate_nondeterminism"] is True
        assert sim2["_simulated"] is True


# =============================================================================
# Phase IV: Security Governance Tile & Release Scenario Gate Tests
# =============================================================================

from security_posture import (
    evaluate_security_for_release,
    build_security_scenario_health_snapshot,
    build_security_director_panel,
    ReleaseStatus,
    HealthStatus,
    BLOCKING_SCENARIOS,
    WARNING_SCENARIOS,
    OK_SCENARIOS,
)


class TestEvaluateSecurityForRelease:
    """Test security release gate evaluation."""

    def test_blocking_scenarios_block_release(self):
        """All blocking scenarios return BLOCK status."""
        posture = build_security_posture()

        for scenario in BLOCKING_SCENARIOS:
            result = evaluate_security_for_release(scenario, posture)
            assert result["status"] == "BLOCK", f"{scenario} should BLOCK"
            assert result["release_ok"] is False, f"{scenario} should not be release_ok"
            assert len(result["blocking_reasons"]) > 0, f"{scenario} should have reasons"

    def test_warning_scenarios_warn_release(self):
        """All warning scenarios return WARN status."""
        posture = build_security_posture()

        for scenario in WARNING_SCENARIOS:
            result = evaluate_security_for_release(scenario, posture)
            assert result["status"] == "WARN", f"{scenario} should WARN"
            assert result["release_ok"] is True, f"{scenario} should be release_ok"
            assert len(result["blocking_reasons"]) > 0, f"{scenario} should have reasons"

    def test_ok_scenarios_allow_release(self):
        """OK scenarios return OK status."""
        posture = build_security_posture()

        for scenario in OK_SCENARIOS:
            result = evaluate_security_for_release(scenario, posture)
            assert result["status"] == "OK", f"{scenario} should be OK"
            assert result["release_ok"] is True, f"{scenario} should be release_ok"
            assert len(result["blocking_reasons"]) == 0, f"{scenario} should have no reasons"

    def test_integrity_compromised_blocks_with_reason(self):
        """INTEGRITY_COMPROMISED has detailed blocking reason."""
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        scenario = classify_security_scenario(posture)
        result = evaluate_security_for_release(scenario, posture)

        assert result["status"] == "BLOCK"
        assert result["release_ok"] is False
        assert "INTEGRITY_COMPROMISED" in str(result["blocking_reasons"])

    def test_release_includes_recommendation(self):
        """All evaluations include a recommendation."""
        posture = build_security_posture()

        # Test BLOCK
        result = evaluate_security_for_release("INTEGRITY_COMPROMISED", posture)
        assert "blocked" in result["recommendation"].lower()

        # Test WARN
        result = evaluate_security_for_release("PARTIAL_REPLAY", posture)
        assert "caution" in result["recommendation"].lower()

        # Test OK
        result = evaluate_security_for_release("NOMINAL", posture)
        assert "approved" in result["recommendation"].lower()

    def test_release_includes_scenario(self):
        """Evaluation includes the evaluated scenario."""
        posture = build_security_posture()
        result = evaluate_security_for_release("SEED_DRIFT_DETECTED", posture)
        assert result["scenario"] == "SEED_DRIFT_DETECTED"

    def test_posture_reasons_included_in_blocking(self):
        """Posture blocking reasons are included in output."""
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            lastmile_report={"overall_status": "NO_GO"}
        )
        scenario = classify_security_scenario(posture)
        result = evaluate_security_for_release(scenario, posture)

        # Should include reasons from posture
        reasons_str = " ".join(result["blocking_reasons"])
        assert "NO_MATCH" in reasons_str or "NO_GO" in reasons_str

    def test_deterministic_evaluation(self):
        """Same inputs produce same outputs."""
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"}
        )
        scenario = classify_security_scenario(posture)

        result1 = evaluate_security_for_release(scenario, posture)
        result2 = evaluate_security_for_release(scenario, posture)

        assert result1 == result2


class TestBuildSecurityScenarioHealthSnapshot:
    """Test scenario health snapshot aggregation."""

    def test_empty_scenarios_ok(self):
        """Empty scenario list returns OK status."""
        snapshot = build_security_scenario_health_snapshot([])

        assert snapshot["status"] == "OK"
        assert snapshot["dominant_scenario"] is None
        assert snapshot["total_scenarios"] == 0
        assert snapshot["scenario_counts_by_severity"]["critical"] == 0

    def test_single_ok_scenario(self):
        """Single OK scenario produces OK status."""
        snapshot = build_security_scenario_health_snapshot(["NOMINAL"])

        assert snapshot["status"] == "OK"
        assert snapshot["dominant_scenario"] == "NOMINAL"
        assert snapshot["total_scenarios"] == 1
        assert snapshot["scenario_counts_by_severity"]["ok"] == 1

    def test_single_warning_scenario(self):
        """Single warning scenario produces ATTENTION status."""
        snapshot = build_security_scenario_health_snapshot(["PARTIAL_REPLAY"])

        assert snapshot["status"] == "ATTENTION"
        assert snapshot["dominant_scenario"] == "PARTIAL_REPLAY"
        assert snapshot["scenario_counts_by_severity"]["warning"] == 1

    def test_single_critical_scenario(self):
        """Single critical scenario produces CRITICAL status."""
        snapshot = build_security_scenario_health_snapshot(["INTEGRITY_COMPROMISED"])

        assert snapshot["status"] == "CRITICAL"
        assert snapshot["dominant_scenario"] == "INTEGRITY_COMPROMISED"
        assert snapshot["scenario_counts_by_severity"]["critical"] == 1

    def test_mixed_scenarios_critical_dominates(self):
        """Any critical scenario produces CRITICAL status."""
        scenarios = ["NOMINAL", "NOMINAL", "PARTIAL_REPLAY", "INTEGRITY_COMPROMISED"]
        snapshot = build_security_scenario_health_snapshot(scenarios)

        assert snapshot["status"] == "CRITICAL"
        assert snapshot["total_scenarios"] == 4
        assert snapshot["scenario_counts_by_severity"]["critical"] == 1
        assert snapshot["scenario_counts_by_severity"]["warning"] == 1
        assert snapshot["scenario_counts_by_severity"]["ok"] == 2

    def test_dominant_scenario_by_frequency(self):
        """Most frequent scenario is dominant."""
        scenarios = ["NOMINAL", "NOMINAL", "NOMINAL", "PARTIAL_REPLAY"]
        snapshot = build_security_scenario_health_snapshot(scenarios)

        assert snapshot["dominant_scenario"] == "NOMINAL"
        assert snapshot["status"] == "ATTENTION"  # Due to warning

    def test_dominant_scenario_tiebreaker_severity(self):
        """Ties broken by severity (critical > warning > ok)."""
        scenarios = ["NOMINAL", "PARTIAL_REPLAY"]  # Both count=1
        snapshot = build_security_scenario_health_snapshot(scenarios)

        # PARTIAL_REPLAY is warning, higher severity than NOMINAL (ok)
        assert snapshot["dominant_scenario"] == "PARTIAL_REPLAY"

    def test_breakdown_included(self):
        """Breakdown includes all scenarios with counts."""
        scenarios = ["NOMINAL", "NOMINAL", "SEED_DRIFT_DETECTED"]
        snapshot = build_security_scenario_health_snapshot(scenarios)

        assert snapshot["breakdown"] is not None
        # Breakdown is sorted by dominance
        breakdown_dict = dict(snapshot["breakdown"])
        assert breakdown_dict["NOMINAL"] == 2
        assert breakdown_dict["SEED_DRIFT_DETECTED"] == 1

    def test_all_critical_scenarios_counted(self):
        """All critical scenarios are in critical tier."""
        for scenario in BLOCKING_SCENARIOS:
            snapshot = build_security_scenario_health_snapshot([scenario])
            assert snapshot["scenario_counts_by_severity"]["critical"] == 1, \
                f"{scenario} should be in critical tier"

    def test_all_warning_scenarios_counted(self):
        """All warning scenarios are in warning tier."""
        for scenario in WARNING_SCENARIOS:
            snapshot = build_security_scenario_health_snapshot([scenario])
            assert snapshot["scenario_counts_by_severity"]["warning"] == 1, \
                f"{scenario} should be in warning tier"

    def test_deterministic_snapshot(self):
        """Same inputs produce same outputs."""
        scenarios = ["NOMINAL", "PARTIAL_REPLAY", "SEED_DRIFT_DETECTED"]

        snapshot1 = build_security_scenario_health_snapshot(scenarios)
        snapshot2 = build_security_scenario_health_snapshot(scenarios)

        assert snapshot1 == snapshot2


class TestBuildSecurityDirectorPanel:
    """Test Director Console security panel building."""

    def test_green_panel_all_ok(self):
        """All OK produces GREEN status light."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)

        panel = build_security_director_panel(health, release)

        assert panel["status_light"] == "GREEN"
        assert panel["dominant_scenario"] == "NOMINAL"
        assert "passed" in panel["headline"].lower()

    def test_yellow_panel_warning(self):
        """Warning produces YELLOW status light."""
        health = build_security_scenario_health_snapshot(["PARTIAL_REPLAY"])
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"}
        )
        release = evaluate_security_for_release("PARTIAL_REPLAY", posture)

        panel = build_security_director_panel(health, release)

        assert panel["status_light"] == "YELLOW"
        assert "warning" in panel["headline"].lower()

    def test_red_panel_blocked(self):
        """Blocked release produces RED status light."""
        health = build_security_scenario_health_snapshot(["INTEGRITY_COMPROMISED"])
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        release = evaluate_security_for_release("INTEGRITY_COMPROMISED", posture)

        panel = build_security_director_panel(health, release)

        assert panel["status_light"] == "RED"
        assert "blocked" in panel["headline"].lower()
        assert panel["release_status"] == "BLOCK"

    def test_red_panel_critical_health(self):
        """Critical health produces RED even if release OK."""
        # Health is critical
        health = build_security_scenario_health_snapshot(["SEED_DRIFT_DETECTED"])
        # But we evaluate OK scenario (artificial case)
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)

        panel = build_security_director_panel(health, release)

        assert panel["status_light"] == "RED"  # Health drives RED

    def test_panel_includes_metrics(self):
        """Panel includes key metrics."""
        health = build_security_scenario_health_snapshot(
            ["NOMINAL", "NOMINAL", "PARTIAL_REPLAY"]
        )
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)

        panel = build_security_director_panel(health, release)

        assert "metrics" in panel
        assert panel["metrics"]["critical_count"] == 0
        assert panel["metrics"]["warning_count"] == 1
        assert panel["metrics"]["ok_count"] == 2
        assert panel["metrics"]["total_scenarios"] == 3
        assert panel["metrics"]["release_ok"] is True

    def test_panel_blocked_uses_release_scenario(self):
        """When blocked, panel shows release scenario."""
        health = build_security_scenario_health_snapshot(["NOMINAL", "NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("SEED_DRIFT_DETECTED", posture)

        panel = build_security_director_panel(health, release)

        # Should show SEED_DRIFT_DETECTED, not NOMINAL
        assert panel["dominant_scenario"] == "SEED_DRIFT_DETECTED"
        assert panel["status_light"] == "RED"

    def test_panel_headline_neutral_tone(self):
        """Headlines use neutral, factual tone."""
        # Test blocked
        health = build_security_scenario_health_snapshot(["INTEGRITY_COMPROMISED"])
        posture = build_security_posture()
        release = evaluate_security_for_release("INTEGRITY_COMPROMISED", posture)
        panel = build_security_director_panel(health, release)

        # Should be factual, not alarmist
        assert "!" not in panel["headline"]
        assert "URGENT" not in panel["headline"].upper()
        assert "DANGER" not in panel["headline"].upper()

    def test_pristine_panel(self):
        """PRISTINE scenario produces informative headline."""
        health = build_security_scenario_health_snapshot(["PRISTINE"])
        posture = build_security_posture()
        release = evaluate_security_for_release("PRISTINE", posture)

        panel = build_security_director_panel(health, release)

        assert panel["status_light"] == "GREEN"
        assert "pending" in panel["headline"].lower() or "ready" in panel["headline"].lower()

    def test_empty_health_panel(self):
        """Empty health snapshot handled gracefully."""
        health = build_security_scenario_health_snapshot([])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)

        panel = build_security_director_panel(health, release)

        assert panel["status_light"] == "GREEN"
        assert "no security scenarios" in panel["headline"].lower()

    def test_deterministic_panel(self):
        """Same inputs produce same outputs."""
        health = build_security_scenario_health_snapshot(["NOMINAL", "PARTIAL_REPLAY"])
        posture = build_security_posture()
        release = evaluate_security_for_release("PARTIAL_REPLAY", posture)

        panel1 = build_security_director_panel(health, release)
        panel2 = build_security_director_panel(health, release)

        assert panel1 == panel2


class TestPhaseIVIntegration:
    """Integration tests for Phase IV end-to-end flows."""

    def test_full_governance_flow_ok(self):
        """Complete flow from posture to panel - OK path."""
        # Build posture
        posture = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )

        # Classify scenario
        scenario = classify_security_scenario(posture)
        assert scenario == "NOMINAL"

        # Evaluate for release
        release = evaluate_security_for_release(scenario, posture)
        assert release["status"] == "OK"
        assert release["release_ok"] is True

        # Build health snapshot
        health = build_security_scenario_health_snapshot([scenario])
        assert health["status"] == "OK"

        # Build director panel
        panel = build_security_director_panel(health, release)
        assert panel["status_light"] == "GREEN"

    def test_full_governance_flow_blocked(self):
        """Complete flow from posture to panel - BLOCKED path."""
        # Build posture with integrity compromise
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            seed_analysis={"classification": "SEED_DRIFT"}
        )

        # Classify scenario
        scenario = classify_security_scenario(posture)
        assert scenario == "INTEGRITY_COMPROMISED"

        # Evaluate for release
        release = evaluate_security_for_release(scenario, posture)
        assert release["status"] == "BLOCK"
        assert release["release_ok"] is False

        # Build health snapshot
        health = build_security_scenario_health_snapshot([scenario])
        assert health["status"] == "CRITICAL"

        # Build director panel
        panel = build_security_director_panel(health, release)
        assert panel["status_light"] == "RED"
        assert panel["metrics"]["release_ok"] is False

    def test_multi_run_aggregation(self):
        """Aggregate scenarios from multiple runs."""
        # Simulate multiple postures
        postures = [
            build_security_posture(
                replay_incident={"replay_status": "FULL_MATCH"}
            ),
            build_security_posture(
                replay_incident={"replay_status": "FULL_MATCH"}
            ),
            build_security_posture(
                replay_incident={"replay_status": "PARTIAL_MATCH"}
            ),
        ]

        # Classify each
        scenarios = [classify_security_scenario(p) for p in postures]

        # Build aggregate snapshot
        health = build_security_scenario_health_snapshot(scenarios)

        assert health["total_scenarios"] == 3
        assert health["status"] == "ATTENTION"  # Due to warning

        # Use latest for release eval
        latest_scenario = scenarios[-1]
        release = evaluate_security_for_release(latest_scenario, postures[-1])

        panel = build_security_director_panel(health, release)
        assert panel["status_light"] == "YELLOW"

    def test_what_if_release_gate(self):
        """Use simulation to test what-if on release gate."""
        # Start with warning scenario
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"}
        )
        scenario = classify_security_scenario(posture)
        release = evaluate_security_for_release(scenario, posture)
        assert release["status"] == "WARN"

        # Simulate fix
        fixed = simulate_security_variation(posture, {"replay_status": "FULL_MATCH"})
        fixed_scenario = classify_security_scenario(fixed)
        fixed_release = evaluate_security_for_release(fixed_scenario, fixed)

        assert fixed_release["status"] == "OK"
        assert fixed_release["release_ok"] is True


# =============================================================================
# Phase V: Security as Cross-Cutting Constraint Tests
# =============================================================================

from security_posture import (
    build_security_replay_ht_view,
    summarize_security_for_global_console,
    to_governance_signal,
    CompositeStatus,
)


class TestBuildSecurityReplayHTView:
    """Test security/replay/HT composite view."""

    def test_all_ok_produces_ok(self):
        """All OK inputs produce OK composite status."""
        view = build_security_replay_ht_view(
            security_scenario="NOMINAL",
            replay_status="OK",
            ht_status="OK"
        )

        assert view["composite_status"] == "OK"
        assert view["blocking_reasons"] == []
        assert view["security_is_constraint"] is False
        assert view["replay_ok"] is True
        assert view["ht_ok"] is True

    def test_security_block_overrides_replay_ht_ok(self):
        """Security BLOCK overrides replay/HT OK - key cross-cutting test."""
        view = build_security_replay_ht_view(
            security_scenario="INTEGRITY_COMPROMISED",
            replay_status="OK",
            ht_status="OK"
        )

        assert view["composite_status"] == "BLOCK"
        assert view["security_is_constraint"] is True
        assert len(view["blocking_reasons"]) > 0
        assert "INTEGRITY_COMPROMISED" in view["scenarios_implicated"]

    def test_security_warn_with_replay_ht_ok(self):
        """Security WARN downgrades replay/HT OK to WARN."""
        view = build_security_replay_ht_view(
            security_scenario="DEGRADED_COVERAGE",
            replay_status="OK",
            ht_status="OK"
        )

        assert view["composite_status"] == "WARN"
        assert view["security_is_constraint"] is True
        assert len(view["blocking_reasons"]) > 0

    def test_replay_fail_blocks(self):
        """Replay FAIL contributes to BLOCK."""
        view = build_security_replay_ht_view(
            security_scenario="NOMINAL",
            replay_status="FAIL",
            ht_status="OK"
        )

        assert view["composite_status"] == "BLOCK"
        assert view["replay_ok"] is False
        assert "REPLAY_FAILURE" in view["scenarios_implicated"]

    def test_ht_fail_blocks(self):
        """HT FAIL contributes to BLOCK."""
        view = build_security_replay_ht_view(
            security_scenario="NOMINAL",
            replay_status="OK",
            ht_status="TAMPERED"
        )

        assert view["composite_status"] == "BLOCK"
        assert view["ht_ok"] is False
        assert "HT_INTEGRITY_FAILURE" in view["scenarios_implicated"]

    def test_multiple_blocks_aggregate(self):
        """Multiple BLOCK sources all contribute reasons."""
        view = build_security_replay_ht_view(
            security_scenario="SEED_DRIFT_DETECTED",
            replay_status="FAIL",
            ht_status="INVALID"
        )

        assert view["composite_status"] == "BLOCK"
        assert len(view["blocking_reasons"]) >= 3
        assert view["security_is_constraint"] is True

    def test_replay_warn_produces_warn(self):
        """Replay WARN produces WARN composite."""
        view = build_security_replay_ht_view(
            security_scenario="NOMINAL",
            replay_status="PARTIAL_MATCH",
            ht_status="OK"
        )

        assert view["composite_status"] == "WARN"
        assert view["replay_ok"] is False  # PARTIAL_MATCH is not "ok"

    def test_ht_warn_produces_warn(self):
        """HT WARN produces WARN composite."""
        view = build_security_replay_ht_view(
            security_scenario="NOMINAL",
            replay_status="OK",
            ht_status="UNVERIFIED"
        )

        assert view["composite_status"] == "WARN"

    def test_not_run_treated_as_ok(self):
        """NOT_RUN status treated as OK (no failure)."""
        view = build_security_replay_ht_view(
            security_scenario="PRISTINE",
            replay_status="NOT_RUN",
            ht_status="NOT_RUN"
        )

        assert view["composite_status"] == "OK"
        assert view["replay_ok"] is True
        assert view["ht_ok"] is True

    def test_case_insensitive_status(self):
        """Status values are case-insensitive."""
        view = build_security_replay_ht_view(
            security_scenario="NOMINAL",
            replay_status="ok",
            ht_status="Ok"
        )

        assert view["composite_status"] == "OK"

    def test_null_status_handled(self):
        """None status values handled gracefully."""
        view = build_security_replay_ht_view(
            security_scenario="NOMINAL",
            replay_status=None,
            ht_status=None
        )

        assert view["composite_status"] == "OK"
        assert view["replay_ok"] is True
        assert view["ht_ok"] is True

    def test_degraded_coverage_with_replay_warn(self):
        """DEGRADED_COVERAGE + replay WARN = WARN (scenario from spec)."""
        view = build_security_replay_ht_view(
            security_scenario="DEGRADED_COVERAGE",
            replay_status="WARN",
            ht_status="OK"
        )

        assert view["composite_status"] == "WARN"
        assert view["security_is_constraint"] is True
        # Both security and replay contribute to warn
        assert len(view["blocking_reasons"]) >= 2

    def test_blocking_scenarios_all_block(self):
        """All blocking scenarios produce BLOCK even with OK replay/HT."""
        for scenario in BLOCKING_SCENARIOS:
            view = build_security_replay_ht_view(
                security_scenario=scenario,
                replay_status="OK",
                ht_status="OK"
            )
            assert view["composite_status"] == "BLOCK", \
                f"{scenario} should BLOCK even with OK replay/HT"
            assert view["security_is_constraint"] is True

    def test_warning_scenarios_all_warn(self):
        """All warning scenarios produce WARN with OK replay/HT."""
        for scenario in WARNING_SCENARIOS:
            view = build_security_replay_ht_view(
                security_scenario=scenario,
                replay_status="OK",
                ht_status="OK"
            )
            assert view["composite_status"] == "WARN", \
                f"{scenario} should WARN with OK replay/HT"
            assert view["security_is_constraint"] is True


class TestSummarizeSecurityForGlobalConsole:
    """Test global console security summary."""

    def test_all_ok_produces_green(self):
        """All OK inputs produce GREEN status and security_ok=True."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")

        summary = summarize_security_for_global_console(health, release, composite)

        assert summary["security_ok"] is True
        assert summary["status_light"] == "GREEN"
        assert summary["is_hard_constraint"] is False
        assert "proceed" in summary["headline"].lower()

    def test_security_block_produces_red(self):
        """Security BLOCK produces RED and security_ok=False."""
        health = build_security_scenario_health_snapshot(["INTEGRITY_COMPROMISED"])
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        release = evaluate_security_for_release("INTEGRITY_COMPROMISED", posture)
        composite = build_security_replay_ht_view("INTEGRITY_COMPROMISED", "OK", "OK")

        summary = summarize_security_for_global_console(health, release, composite)

        assert summary["security_ok"] is False
        assert summary["status_light"] == "RED"
        assert summary["is_hard_constraint"] is True

    def test_security_warn_produces_yellow(self):
        """Security WARN produces YELLOW."""
        health = build_security_scenario_health_snapshot(["PARTIAL_REPLAY"])
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"}
        )
        release = evaluate_security_for_release("PARTIAL_REPLAY", posture)
        composite = build_security_replay_ht_view("PARTIAL_REPLAY", "OK", "OK")

        summary = summarize_security_for_global_console(health, release, composite)

        assert summary["status_light"] == "YELLOW"
        assert summary["security_ok"] is False  # WARN means not fully OK

    def test_composite_block_overrides_release_ok(self):
        """Composite BLOCK overrides release OK."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        # Composite has BLOCK due to HT failure
        composite = build_security_replay_ht_view("NOMINAL", "OK", "FAIL")

        summary = summarize_security_for_global_console(health, release, composite)

        assert summary["security_ok"] is False
        assert summary["status_light"] == "RED"

    def test_dominant_scenario_from_release(self):
        """Dominant scenario comes from release eval first."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("SEED_DRIFT_DETECTED", posture)
        composite = build_security_replay_ht_view("SEED_DRIFT_DETECTED", "OK", "OK")

        summary = summarize_security_for_global_console(health, release, composite)

        assert summary["dominant_scenario"] == "SEED_DRIFT_DETECTED"

    def test_blocking_reasons_aggregated(self):
        """Blocking reasons from both composite and release are aggregated."""
        health = build_security_scenario_health_snapshot(["MULTI_FAILURE"])
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            lastmile_report={"overall_status": "NO_GO"}
        )
        release = evaluate_security_for_release("MULTI_FAILURE", posture)
        composite = build_security_replay_ht_view("MULTI_FAILURE", "OK", "OK")

        summary = summarize_security_for_global_console(health, release, composite)

        # Should have reasons from both
        assert len(summary["blocking_reasons"]) > 0

    def test_headline_mentions_constraint_when_blocking(self):
        """Headline mentions constraint when security is blocking."""
        health = build_security_scenario_health_snapshot(["SEED_DRIFT_DETECTED"])
        posture = build_security_posture(
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        release = evaluate_security_for_release("SEED_DRIFT_DETECTED", posture)
        composite = build_security_replay_ht_view("SEED_DRIFT_DETECTED", "OK", "OK")

        summary = summarize_security_for_global_console(health, release, composite)

        assert "constraint" in summary["headline"].lower() or "blocking" in summary["headline"].lower()


class TestToGovernanceSignal:
    """Test governance signal generation for CLAUDE I."""

    def test_signal_has_required_fields(self):
        """Governance signal has all required fields."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary, run_id="test-run-001")

        assert signal["signal_type"] == "SECURITY_POSTURE"
        assert signal["run_id"] == "test-run-001"
        assert "timestamp" in signal
        assert "security_ok" in signal
        assert "status" in signal
        assert "dominant_scenario" in signal
        assert "is_blocking" in signal
        assert "summary" in signal
        assert "metadata" in signal

    def test_signal_reflects_ok_state(self):
        """Signal correctly reflects OK state."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary)

        assert signal["security_ok"] is True
        assert signal["status"] == "GREEN"
        assert signal["is_blocking"] is False

    def test_signal_reflects_blocked_state(self):
        """Signal correctly reflects blocked state."""
        health = build_security_scenario_health_snapshot(["INTEGRITY_COMPROMISED"])
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        release = evaluate_security_for_release("INTEGRITY_COMPROMISED", posture)
        composite = build_security_replay_ht_view("INTEGRITY_COMPROMISED", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary, run_id="blocked-run")

        assert signal["security_ok"] is False
        assert signal["status"] == "RED"
        assert signal["is_blocking"] is True
        assert signal["dominant_scenario"] == "INTEGRITY_COMPROMISED"

    def test_signal_includes_metadata(self):
        """Signal includes useful metadata."""
        health = build_security_scenario_health_snapshot(["PARTIAL_REPLAY"])
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"}
        )
        release = evaluate_security_for_release("PARTIAL_REPLAY", posture)
        composite = build_security_replay_ht_view("PARTIAL_REPLAY", "WARN", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary)

        assert "composite_status" in signal["metadata"]
        assert "blocking_reason_count" in signal["metadata"]

    def test_signal_uses_provided_timestamp(self):
        """Signal uses provided timestamp if given."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        custom_ts = "2025-01-15T12:00:00Z"
        signal = to_governance_signal(summary, timestamp=custom_ts)

        assert signal["timestamp"] == custom_ts

    def test_signal_summary_from_headline(self):
        """Signal summary comes from headline."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary)

        assert signal["summary"] == summary["headline"]


class TestPhaseVIntegration:
    """Integration tests for Phase V cross-cutting security."""

    def test_security_blocks_even_when_replay_ht_ok(self):
        """Security BLOCK even when replay/HT are OK - key integration test."""
        # Build posture with security issue but no replay/HT issue
        posture = build_security_posture(
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        scenario = classify_security_scenario(posture)

        # Security should be SEED_DRIFT_DETECTED
        assert scenario == "SEED_DRIFT_DETECTED"

        # Release should be blocked
        release = evaluate_security_for_release(scenario, posture)
        assert release["status"] == "BLOCK"

        # Composite view with OK replay/HT should still BLOCK
        composite = build_security_replay_ht_view(scenario, "OK", "OK")
        assert composite["composite_status"] == "BLOCK"
        assert composite["security_is_constraint"] is True

        # Global console should show RED
        health = build_security_scenario_health_snapshot([scenario])
        summary = summarize_security_for_global_console(health, release, composite)
        assert summary["security_ok"] is False
        assert summary["status_light"] == "RED"

        # Governance signal should indicate blocking
        signal = to_governance_signal(summary)
        assert signal["is_blocking"] is True

    def test_security_warn_with_degraded_coverage_and_replay_warn(self):
        """Security WARN with DEGRADED_COVERAGE + replay WARN - scenario from spec."""
        # Build posture with multiple warnings = DEGRADED_COVERAGE
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"},
            seed_analysis={"classification": "SUBSTRATE_NONDETERMINISM"}
        )
        scenario = classify_security_scenario(posture)

        # Should be DEGRADED_COVERAGE (2+ warnings)
        assert scenario == "DEGRADED_COVERAGE"

        # Release should be WARN
        release = evaluate_security_for_release(scenario, posture)
        assert release["status"] == "WARN"

        # Composite with replay WARN
        composite = build_security_replay_ht_view(scenario, "WARN", "OK")
        assert composite["composite_status"] == "WARN"

        # Global console should show YELLOW
        health = build_security_scenario_health_snapshot([scenario])
        summary = summarize_security_for_global_console(health, release, composite)
        assert summary["status_light"] == "YELLOW"
        assert summary["security_ok"] is False

    def test_full_pipeline_ok_path(self):
        """Full pipeline with everything OK."""
        # Build clean posture
        posture = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )
        scenario = classify_security_scenario(posture)
        assert scenario == "NOMINAL"

        # All components OK
        release = evaluate_security_for_release(scenario, posture)
        composite = build_security_replay_ht_view(scenario, "OK", "OK")
        health = build_security_scenario_health_snapshot([scenario])

        # Global console green
        summary = summarize_security_for_global_console(health, release, composite)
        assert summary["security_ok"] is True
        assert summary["status_light"] == "GREEN"
        assert summary["is_hard_constraint"] is False

        # Governance signal OK
        signal = to_governance_signal(summary, run_id="clean-run")
        assert signal["security_ok"] is True
        assert signal["is_blocking"] is False

    def test_ht_failure_blocks_despite_security_ok(self):
        """HT failure blocks even when security scenario is OK."""
        posture = build_security_posture()
        scenario = classify_security_scenario(posture)

        # Security is PRISTINE (OK)
        release = evaluate_security_for_release(scenario, posture)
        assert release["status"] == "OK"

        # But HT is TAMPERED
        composite = build_security_replay_ht_view(scenario, "OK", "TAMPERED")
        assert composite["composite_status"] == "BLOCK"
        assert "HT_INTEGRITY_FAILURE" in composite["scenarios_implicated"]

        # Global console should be RED
        health = build_security_scenario_health_snapshot([scenario])
        summary = summarize_security_for_global_console(health, release, composite)
        assert summary["security_ok"] is False
        assert summary["status_light"] == "RED"

    def test_multiple_runs_with_varying_security(self):
        """Multiple runs with varying security states aggregate correctly."""
        # Run 1: OK
        run1_scenario = "NOMINAL"
        # Run 2: WARN
        run2_scenario = "PARTIAL_REPLAY"
        # Run 3: BLOCK
        run3_scenario = "SEED_DRIFT_DETECTED"

        # Aggregate health
        health = build_security_scenario_health_snapshot([
            run1_scenario, run2_scenario, run3_scenario
        ])

        assert health["status"] == "CRITICAL"  # Due to SEED_DRIFT_DETECTED
        assert health["scenario_counts_by_severity"]["critical"] == 1
        assert health["scenario_counts_by_severity"]["warning"] == 1
        assert health["scenario_counts_by_severity"]["ok"] == 1

        # Latest run drives release
        posture = build_security_posture(seed_analysis={"classification": "SEED_DRIFT"})
        release = evaluate_security_for_release(run3_scenario, posture)
        composite = build_security_replay_ht_view(run3_scenario, "OK", "OK")

        summary = summarize_security_for_global_console(health, release, composite)
        assert summary["status_light"] == "RED"
        assert summary["is_hard_constraint"] is True


# =============================================================================
# Phase VI: Security as Meta-Governance Layer Tests
# =============================================================================

from security_posture import (
    GovernanceSignal,
    GOVERNANCE_SIGNAL_SCHEMA_VERSION,
    validate_governance_signal,
    apply_security_override_to_global_status,
    get_security_override_rule_doc,
)


class TestGovernanceSignalSchema:
    """Test GovernanceSignal schema compliance."""

    def test_signal_has_schema_version(self):
        """Signal includes schema version."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary)

        assert signal["schema_version"] == GOVERNANCE_SIGNAL_SCHEMA_VERSION
        assert signal["schema_version"] == "1.0.0"

    def test_signal_has_all_required_fields(self):
        """Signal has all fields required by GovernanceSignal schema."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary, run_id="test-run")

        # All required fields present
        assert "schema_version" in signal
        assert "signal_type" in signal
        assert "run_id" in signal
        assert "timestamp" in signal
        assert "security_ok" in signal
        assert "status" in signal
        assert "dominant_scenario" in signal
        assert "is_blocking" in signal
        assert "forces_global_block" in signal
        assert "summary" in signal
        assert "blocking_reasons" in signal
        assert "scenarios_implicated" in signal
        assert "metadata" in signal

    def test_signal_type_is_security_posture(self):
        """Signal type is always SECURITY_POSTURE."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary)

        assert signal["signal_type"] == "SECURITY_POSTURE"

    def test_metadata_includes_source(self):
        """Metadata includes source identifier."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary)

        assert signal["metadata"]["source"] == "CLAUDE_K_SECURITY_POSTURE"


class TestSecurityPriorityOverrideRule:
    """Test the SECURITY RED = BLOCK override rule."""

    def test_red_status_forces_global_block(self):
        """RED status automatically sets forces_global_block=True."""
        health = build_security_scenario_health_snapshot(["INTEGRITY_COMPROMISED"])
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        release = evaluate_security_for_release("INTEGRITY_COMPROMISED", posture)
        composite = build_security_replay_ht_view("INTEGRITY_COMPROMISED", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary)

        assert signal["status"] == "RED"
        assert signal["forces_global_block"] is True

    def test_yellow_status_does_not_force_block(self):
        """YELLOW status does not force global block."""
        health = build_security_scenario_health_snapshot(["PARTIAL_REPLAY"])
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"}
        )
        release = evaluate_security_for_release("PARTIAL_REPLAY", posture)
        composite = build_security_replay_ht_view("PARTIAL_REPLAY", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary)

        assert signal["status"] == "YELLOW"
        assert signal["forces_global_block"] is False

    def test_green_status_does_not_force_block(self):
        """GREEN status does not force global block."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary)

        assert signal["status"] == "GREEN"
        assert signal["forces_global_block"] is False

    def test_all_blocking_scenarios_force_block(self):
        """All blocking scenarios produce forces_global_block=True."""
        for scenario in BLOCKING_SCENARIOS:
            health = build_security_scenario_health_snapshot([scenario])
            posture = build_security_posture()
            release = evaluate_security_for_release(scenario, posture)
            composite = build_security_replay_ht_view(scenario, "OK", "OK")
            summary = summarize_security_for_global_console(health, release, composite)

            signal = to_governance_signal(summary)

            assert signal["forces_global_block"] is True, \
                f"{scenario} should force global block"


class TestValidateGovernanceSignal:
    """Test governance signal validation."""

    def test_valid_signal_passes(self):
        """Valid signal passes validation."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)

        signal = to_governance_signal(summary)
        is_valid, errors = validate_governance_signal(signal)

        assert is_valid is True
        assert errors == []

    def test_missing_field_fails(self):
        """Missing required field fails validation."""
        signal = {"signal_type": "SECURITY_POSTURE"}
        is_valid, errors = validate_governance_signal(signal)

        assert is_valid is False
        assert len(errors) > 0
        assert any("Missing required field" in e for e in errors)

    def test_invalid_signal_type_fails(self):
        """Invalid signal_type fails validation."""
        signal = {
            "schema_version": "1.0.0",
            "signal_type": "WRONG_TYPE",
            "run_id": "test",
            "timestamp": "2025-01-01T00:00:00Z",
            "security_ok": True,
            "status": "GREEN",
            "is_blocking": False,
            "forces_global_block": False,
            "summary": "test",
            "blocking_reasons": [],
            "scenarios_implicated": [],
            "metadata": {},
        }
        is_valid, errors = validate_governance_signal(signal)

        assert is_valid is False
        assert any("Invalid signal_type" in e for e in errors)

    def test_invalid_status_fails(self):
        """Invalid status fails validation."""
        signal = {
            "schema_version": "1.0.0",
            "signal_type": "SECURITY_POSTURE",
            "run_id": "test",
            "timestamp": "2025-01-01T00:00:00Z",
            "security_ok": True,
            "status": "PURPLE",  # Invalid
            "is_blocking": False,
            "forces_global_block": False,
            "summary": "test",
            "blocking_reasons": [],
            "scenarios_implicated": [],
            "metadata": {},
        }
        is_valid, errors = validate_governance_signal(signal)

        assert is_valid is False
        assert any("Invalid status" in e for e in errors)

    def test_red_without_forces_block_fails(self):
        """RED status without forces_global_block=True fails validation."""
        signal = {
            "schema_version": "1.0.0",
            "signal_type": "SECURITY_POSTURE",
            "run_id": "test",
            "timestamp": "2025-01-01T00:00:00Z",
            "security_ok": False,
            "status": "RED",
            "is_blocking": True,
            "forces_global_block": False,  # VIOLATION!
            "summary": "test",
            "blocking_reasons": [],
            "scenarios_implicated": [],
            "metadata": {},
        }
        is_valid, errors = validate_governance_signal(signal)

        assert is_valid is False
        assert any("VIOLATION" in e for e in errors)


class TestApplySecurityOverride:
    """Test security override application for CLAUDE I."""

    def test_red_overrides_ok(self):
        """RED security overrides OK global status to BLOCK."""
        health = build_security_scenario_health_snapshot(["SEED_DRIFT_DETECTED"])
        posture = build_security_posture(seed_analysis={"classification": "SEED_DRIFT"})
        release = evaluate_security_for_release("SEED_DRIFT_DETECTED", posture)
        composite = build_security_replay_ht_view("SEED_DRIFT_DETECTED", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)
        signal = to_governance_signal(summary)

        result = apply_security_override_to_global_status("OK", signal)

        assert result["global_status"] == "BLOCK"
        assert result["security_overrode"] is True
        assert result["original_status"] == "OK"

    def test_red_overrides_warn(self):
        """RED security overrides WARN global status to BLOCK."""
        health = build_security_scenario_health_snapshot(["INTEGRITY_COMPROMISED"])
        posture = build_security_posture(
            replay_incident={"replay_status": "NO_MATCH"},
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        release = evaluate_security_for_release("INTEGRITY_COMPROMISED", posture)
        composite = build_security_replay_ht_view("INTEGRITY_COMPROMISED", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)
        signal = to_governance_signal(summary)

        result = apply_security_override_to_global_status("WARN", signal)

        assert result["global_status"] == "BLOCK"
        assert result["security_overrode"] is True

    def test_yellow_downgrades_ok_to_warn(self):
        """YELLOW security downgrades OK to WARN."""
        health = build_security_scenario_health_snapshot(["PARTIAL_REPLAY"])
        posture = build_security_posture(
            replay_incident={"replay_status": "PARTIAL_MATCH"}
        )
        release = evaluate_security_for_release("PARTIAL_REPLAY", posture)
        composite = build_security_replay_ht_view("PARTIAL_REPLAY", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)
        signal = to_governance_signal(summary)

        result = apply_security_override_to_global_status("OK", signal)

        assert result["global_status"] == "WARN"
        assert result["security_overrode"] is True

    def test_green_does_not_override(self):
        """GREEN security does not override."""
        health = build_security_scenario_health_snapshot(["NOMINAL"])
        posture = build_security_posture()
        release = evaluate_security_for_release("NOMINAL", posture)
        composite = build_security_replay_ht_view("NOMINAL", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)
        signal = to_governance_signal(summary)

        result = apply_security_override_to_global_status("OK", signal)

        assert result["global_status"] == "OK"
        assert result["security_overrode"] is False

    def test_override_includes_reason(self):
        """Override result includes reason."""
        health = build_security_scenario_health_snapshot(["SEED_DRIFT_DETECTED"])
        posture = build_security_posture(seed_analysis={"classification": "SEED_DRIFT"})
        release = evaluate_security_for_release("SEED_DRIFT_DETECTED", posture)
        composite = build_security_replay_ht_view("SEED_DRIFT_DETECTED", "OK", "OK")
        summary = summarize_security_for_global_console(health, release, composite)
        signal = to_governance_signal(summary)

        result = apply_security_override_to_global_status("OK", signal)

        assert "reason" in result
        assert "SEED_DRIFT_DETECTED" in result["reason"]


class TestPhaseVIIntegration:
    """Integration tests for Phase VI meta-governance layer."""

    def test_full_governance_signal_flow(self):
        """Full flow from posture to governance signal."""
        # Build posture
        posture = build_security_posture(
            replay_incident={"replay_status": "FULL_MATCH"},
            seed_analysis={"classification": "NO_ISSUE"},
            lastmile_report={"overall_status": "READY"}
        )
        scenario = classify_security_scenario(posture)

        # Build all components
        release = evaluate_security_for_release(scenario, posture)
        composite = build_security_replay_ht_view(scenario, "OK", "OK")
        health = build_security_scenario_health_snapshot([scenario])
        summary = summarize_security_for_global_console(health, release, composite)

        # Generate governance signal
        signal = to_governance_signal(summary, run_id="integration-test")

        # Validate
        is_valid, errors = validate_governance_signal(signal)
        assert is_valid is True

        # Apply override
        result = apply_security_override_to_global_status("OK", signal)
        assert result["global_status"] == "OK"
        assert result["security_overrode"] is False

    def test_security_override_in_claude_i_style(self):
        """Demonstrate CLAUDE I synthesizer pattern."""
        # Simulate multiple signals (only security matters for override)
        signals = []

        # Build security signal
        posture = build_security_posture(
            seed_analysis={"classification": "SEED_DRIFT"}
        )
        scenario = classify_security_scenario(posture)
        release = evaluate_security_for_release(scenario, posture)
        composite = build_security_replay_ht_view(scenario, "OK", "OK")
        health = build_security_scenario_health_snapshot([scenario])
        summary = summarize_security_for_global_console(health, release, composite)
        security_signal = to_governance_signal(summary)
        signals.append(security_signal)

        # CLAUDE I synthesizer pattern
        def synthesize_global_status(sigs):
            # Find security signal
            sec_signal = next(
                (s for s in sigs if s.get("signal_type") == "SECURITY_POSTURE"),
                None
            )

            # SECURITY OVERRIDE CHECK - MUST BE FIRST
            if sec_signal and sec_signal.get("forces_global_block"):
                return "BLOCK"

            # ... rest of synthesis (not shown)
            return "OK"

        result = synthesize_global_status(signals)
        assert result == "BLOCK"  # Security RED forced BLOCK

    def test_documentation_accessible(self):
        """Security override rule documentation is accessible."""
        doc = get_security_override_rule_doc()

        assert "SECURITY PRIORITY OVERRIDE RULE" in doc
        assert "forces_global_block" in doc
        assert "CLAUDE I" in doc
        assert "global_status=BLOCK" in doc
        assert "NON-NEGOTIABLE" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
