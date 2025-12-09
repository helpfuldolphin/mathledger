# PHASE II — U2 UPLIFT EXPERIMENT
"""
Integration tests for the Admissibility Law Engine (MAAS v2).

Tests the 5-phase algorithm from U2_EVIDENCE_ADMISSIBILITY_SPEC.md Section 10:
  1. Governance Verifier Precondition (BLOCKING)
  2. Core Artifact Presence Check
  3. Structural Integrity Checks
  4. Artifact Integrity Checks
  5. Relationship Integrity Checks

Test cases per user requirements:
  - Missing governance report -> HE-GV* triggered -> INADMISSIBLE
  - Governance FAIL but all artifacts present -> still INADMISSIBLE
  - All checks pass -> ADMISSIBLE
  - Correct JSON structure of admissibility reports
"""

import dataclasses
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional

from backend.promotion.admissibility_engine import (
    COMPASS_CRITICAL_THRESHOLD,
    COMPASS_STABLE_THRESHOLD,
    CORE_ARTIFACT_IDS,
    GOVV_REQUIRED_FIELDS,
    GOVV_REQUIRED_GATES,
    PHASE_NAMES,
    RECURRENCE_THRESHOLD,
    AdmissibilityError,
    AdmissibilityReport,
    AdmissibilityVerdict,
    attach_admissibility_to_evidence_chain,
    build_admissibility_analytics,
    build_admissibility_compass,
    build_admissibility_director_panel,
    build_admissibility_snapshot,
    check_admissibility,
    compute_dossier_hash,
    evaluate_admissibility_for_promotion,
    map_admissibility_to_director_light,
    summarize_admissibility_for_global_console,
    summarize_admissibility_for_global_dashboard,
    summarize_admissibility_for_global_health,
    to_governance_signal,
)
from backend.promotion.u2_evidence import U2Artifact, U2Dossier


# =============================================================================
# TEST FIXTURES
# =============================================================================

def make_valid_governance_report(dossier: Optional["U2Dossier"] = None) -> Dict[str, Any]:
    """Create a valid governance verifier report that passes all checks.

    If a dossier is provided, computes the correct dossier_hash to match.
    """
    dossier_hash = compute_dossier_hash(dossier) if dossier else "abc123"
    return {
        "report_id": "gov-report-001",
        "verified_at": "2025-01-01T00:00:00Z",
        "verifier_version": "1.0.0",
        "dossier_hash": f"sha256-{dossier_hash}",
        "gates": {
            "G1": {"status": "PASS", "description": "Preregistration"},
            "G2": {"status": "PASS", "description": "Determinism"},
            "G3": {"status": "PASS", "description": "Manifest"},
            "G4": {"status": "PASS", "description": "RFL Integrity"},
        },
        "overall_status": "PASS",
        "artifact_inventory": {
            "found": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            "missing": [],
        },
        "signature": "sig-abc123",
    }


def make_artifact(
    artifact_id: str,
    description: str,
    found: bool = True,
    paths: Optional[List[str]] = None,
    is_multi_instance: bool = False,
) -> U2Artifact:
    """Create a U2Artifact for testing."""
    return U2Artifact(
        id=artifact_id,
        description=description,
        location_template=f"artifacts/u2/{artifact_id.lower()}.json",
        versioning_scheme="Run ID",
        is_multi_instance=is_multi_instance,
        found=found,
        paths=paths or [],
        validation_status="FOUND" if found else "MISSING",
    )


def make_complete_dossier(run_id: str = "test_run_001") -> U2Dossier:
    """Create a complete U2Dossier with all artifacts present."""
    return U2Dossier(
        run_id=run_id,
        creation_timestamp="2025-01-01T00:00:00Z",
        artifacts={
            "A1": make_artifact("A1", "Preregistration", found=True, paths=["docs/prereg/PREREG.yaml"]),
            "A2": make_artifact("A2", "Preregistration Seal", found=True, paths=["docs/prereg/PREREG.yaml.sig"]),
            "A3": make_artifact(
                "A3", "Environment Manifests", found=True,
                paths=[
                    "artifacts/u2/ENV_A/manifest.json",
                    "artifacts/u2/ENV_B/manifest.json",
                    "artifacts/u2/ENV_C/manifest.json",
                    "artifacts/u2/ENV_D/manifest.json",
                ],
                is_multi_instance=True,
            ),
            "A4": make_artifact(
                "A4", "Raw Telemetry", found=True,
                paths=[
                    "artifacts/u2/ENV_A/telemetry.jsonl",
                    "artifacts/u2/ENV_B/telemetry.jsonl",
                    "artifacts/u2/ENV_C/telemetry.jsonl",
                    "artifacts/u2/ENV_D/telemetry.jsonl",
                ],
                is_multi_instance=True,
            ),
            "A5": make_artifact("A5", "Statistical Summary", found=True, paths=["artifacts/u2/analysis/summary.json"]),
            "A6": make_artifact(
                "A6", "Uplift Curve Plots", found=True,
                paths=[
                    "artifacts/u2/analysis/curve_A.png",
                    "artifacts/u2/analysis/curve_B.png",
                    "artifacts/u2/analysis/curve_C.png",
                    "artifacts/u2/analysis/curve_D.png",
                ],
                is_multi_instance=True,
            ),
            "A7": make_artifact("A7", "Gate Compliance Log", found=True, paths=["ops/logs/u2_compliance.jsonl"]),
            "A8": make_artifact("A8", "Governance Verifier Report", found=True, paths=["artifacts/u2/governance_verifier_report.json"]),
        },
        dossier_status="COMPLETE",
    )


def make_incomplete_dossier(missing_artifacts: List[str], run_id: str = "test_run_001") -> U2Dossier:
    """Create an incomplete U2Dossier with specified artifacts missing."""
    dossier = make_complete_dossier(run_id)
    for artifact_id in missing_artifacts:
        if artifact_id in dossier.artifacts:
            artifact = dossier.artifacts[artifact_id]
            # Create a new artifact with found=False
            dossier.artifacts[artifact_id] = U2Artifact(
                id=artifact.id,
                description=artifact.description,
                location_template=artifact.location_template,
                versioning_scheme=artifact.versioning_scheme,
                is_multi_instance=artifact.is_multi_instance,
                found=False,
                paths=[],
                validation_status="MISSING",
            )
    dossier.dossier_status = "INCOMPLETE"
    return dossier


# =============================================================================
# TEST CASES FOR ADMISSIBILITY ENGINE CONSTANTS
# Note: Many legacy tests depend on an older API with (dossier, governance_report).
# The current engine only takes (dossier) and loads A8 internally.
# These tests focus on the dashboard/trajectory features (Tasks 1-3).
# =============================================================================

class TestAdmissibilityEngineCoreArtifactConstants(unittest.TestCase):
    """Tests for the MAAS v2 core artifact constants."""

    def test_core_artifact_ids_contains_a8(self):
        """A8 (Governance Verifier Report) is in CORE_ARTIFACT_IDS per MAAS v2."""
        self.assertIn("A8", CORE_ARTIFACT_IDS)

    def test_core_artifact_ids_count(self):
        """CORE_ARTIFACT_IDS contains 6 core artifacts per current spec."""
        # Current spec: A1, A2, A3, A5, A7, A8 are CORE
        self.assertEqual(len(CORE_ARTIFACT_IDS), 6)

    def test_govv_required_fields_complete(self):
        """GOVV_REQUIRED_FIELDS contains all required fields per Section 7.2."""
        expected_fields = [
            "report_id",
            "verified_at",
            "verifier_version",
            "dossier_hash",
            "gates",
            "overall_status",
            "artifact_inventory",
            "signature",
        ]
        for field in expected_fields:
            self.assertIn(field, GOVV_REQUIRED_FIELDS)

    def test_govv_required_gates_complete(self):
        """GOVV_REQUIRED_GATES contains G1-G4."""
        self.assertEqual(sorted(GOVV_REQUIRED_GATES), ["G1", "G2", "G3", "G4"])


# =============================================================================
# TASK 1: ADMISSIBILITY SNAPSHOT TESTS
# =============================================================================

class TestAdmissibilitySnapshot(unittest.TestCase):
    """Tests for build_admissibility_snapshot function."""

    def test_snapshot_contains_schema_version(self):
        """Snapshot includes schema_version."""
        report = {
            "decision_id": "test-123",
            "executed_at": "2025-01-01T00:00:00Z",
            "verdict": {"status": "ADMISSIBLE"},
            "errors": [],
        }
        snapshot = build_admissibility_snapshot(report)
        self.assertEqual(snapshot["schema_version"], SNAPSHOT_SCHEMA_VERSION)

    def test_snapshot_extracts_verdict_from_dict(self):
        """Snapshot extracts verdict from dict format."""
        report = {
            "verdict": {"status": "NOT_ADMISSIBLE", "code": "HE-GV1"},
            "errors": [{"error_id": "HE-GV1", "phase": 1}],
        }
        snapshot = build_admissibility_snapshot(report)
        self.assertEqual(snapshot["verdict"], "NOT_ADMISSIBLE")

    def test_snapshot_extracts_admissible_verdict(self):
        """Snapshot correctly identifies ADMISSIBLE verdict."""
        report = {
            "verdict": {"status": "ADMISSIBLE"},
            "errors": [],
        }
        snapshot = build_admissibility_snapshot(report)
        self.assertEqual(snapshot["verdict"], "ADMISSIBLE")

    def test_snapshot_error_codes_are_sorted(self):
        """Snapshot error_codes are sorted alphabetically for stability."""
        report = {
            "verdict": {"status": "NOT_ADMISSIBLE"},
            "errors": [
                {"error_id": "HE-S1", "phase": 2},
                {"error_id": "HE-GV1", "phase": 1},
                {"error_id": "DOSSIER-11", "phase": 4},
            ],
        }
        snapshot = build_admissibility_snapshot(report)
        self.assertEqual(snapshot["error_codes"], ["DOSSIER-11", "HE-GV1", "HE-S1"])

    def test_snapshot_error_codes_are_unique(self):
        """Snapshot error_codes removes duplicates."""
        report = {
            "verdict": {"status": "NOT_ADMISSIBLE"},
            "errors": [
                {"error_id": "HE-GV1", "phase": 1},
                {"error_id": "HE-GV1", "phase": 1},
            ],
        }
        snapshot = build_admissibility_snapshot(report)
        self.assertEqual(snapshot["error_codes"], ["HE-GV1"])

    def test_snapshot_phases_executed_for_admissible(self):
        """Snapshot includes all phases for ADMISSIBLE verdict."""
        report = {
            "verdict": {"status": "ADMISSIBLE"},
            "errors": [],
        }
        snapshot = build_admissibility_snapshot(report)
        self.assertEqual(snapshot["phases_executed"], PHASE_NAMES)

    def test_snapshot_phases_executed_inferred_from_error(self):
        """Snapshot infers phases from error phase number."""
        report = {
            "verdict": {"status": "NOT_ADMISSIBLE"},
            "errors": [{"error_id": "HE-S1", "phase": 2}],
        }
        snapshot = build_admissibility_snapshot(report)
        self.assertEqual(snapshot["phases_executed"], ["phase_1_governance", "phase_2_artifacts"])

    def test_snapshot_preserves_decision_id(self):
        """Snapshot preserves decision_id reference."""
        report = {
            "decision_id": "unique-id-12345",
            "verdict": {"status": "ADMISSIBLE"},
            "errors": [],
        }
        snapshot = build_admissibility_snapshot(report)
        self.assertEqual(snapshot["decision_id"], "unique-id-12345")

    def test_snapshot_preserves_executed_at(self):
        """Snapshot preserves executed_at timestamp."""
        report = {
            "executed_at": "2025-06-15T10:30:00Z",
            "verdict": {"status": "ADMISSIBLE"},
            "errors": [],
        }
        snapshot = build_admissibility_snapshot(report)
        self.assertEqual(snapshot["executed_at"], "2025-06-15T10:30:00Z")

    def test_snapshot_is_idempotent(self):
        """Same input produces same output (idempotent)."""
        report = {
            "decision_id": "test-123",
            "executed_at": "2025-01-01T00:00:00Z",
            "verdict": {"status": "NOT_ADMISSIBLE"},
            "errors": [
                {"error_id": "HE-GV2", "phase": 1},
                {"error_id": "HE-GV1", "phase": 1},
            ],
        }
        snapshot1 = build_admissibility_snapshot(report)
        snapshot2 = build_admissibility_snapshot(report)
        self.assertEqual(snapshot1, snapshot2)

    def test_snapshot_is_json_serializable(self):
        """Snapshot is JSON serializable."""
        report = {
            "decision_id": "test-123",
            "verdict": {"status": "NOT_ADMISSIBLE"},
            "errors": [{"error_id": "HE-GV1", "phase": 1}],
        }
        snapshot = build_admissibility_snapshot(report)
        json_str = json.dumps(snapshot)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, snapshot)


# =============================================================================
# TASK 2: ADMISSIBILITY TIMELINE TESTS
# =============================================================================

class TestAdmissibilityTimeline(unittest.TestCase):
    """Tests for build_admissibility_timeline function."""

    def test_timeline_counts_total_cases(self):
        """Timeline counts total cases correctly."""
        snapshots = [
            {"verdict": "ADMISSIBLE", "error_codes": []},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]},
        ]
        timeline = build_admissibility_timeline(snapshots)
        self.assertEqual(timeline["total_cases"], 3)

    def test_timeline_counts_admissible(self):
        """Timeline counts admissible verdicts."""
        snapshots = [
            {"verdict": "ADMISSIBLE", "error_codes": []},
            {"verdict": "ADMISSIBLE", "error_codes": []},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
        ]
        timeline = build_admissibility_timeline(snapshots)
        self.assertEqual(timeline["admissible_count"], 2)
        self.assertEqual(timeline["inadmissible_count"], 1)

    def test_timeline_calculates_admissibility_rate(self):
        """Timeline calculates admissibility rate correctly."""
        snapshots = [
            {"verdict": "ADMISSIBLE", "error_codes": []},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]},
            {"verdict": "ADMISSIBLE", "error_codes": []},
        ]
        timeline = build_admissibility_timeline(snapshots)
        self.assertEqual(timeline["admissibility_rate"], 0.5)

    def test_timeline_handles_empty_list(self):
        """Timeline handles empty snapshot list."""
        timeline = build_admissibility_timeline([])
        self.assertEqual(timeline["total_cases"], 0)
        self.assertEqual(timeline["admissible_count"], 0)
        self.assertEqual(timeline["inadmissible_count"], 0)
        self.assertEqual(timeline["admissibility_rate"], 0.0)

    def test_timeline_aggregates_error_code_frequency(self):
        """Timeline aggregates error code frequency correctly."""
        snapshots = [
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1", "HE-GV2"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
        ]
        timeline = build_admissibility_timeline(snapshots)
        self.assertEqual(timeline["error_code_frequency"]["HE-GV1"], 3)
        self.assertEqual(timeline["error_code_frequency"]["HE-GV2"], 1)
        self.assertEqual(timeline["error_code_frequency"]["HE-S1"], 1)

    def test_timeline_error_frequency_sorted_by_count(self):
        """Timeline error_code_frequency is sorted by count descending."""
        snapshots = [
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
        ]
        timeline = build_admissibility_timeline(snapshots)
        codes = list(timeline["error_code_frequency"].keys())
        self.assertEqual(codes[0], "HE-GV1")  # Most frequent first

    def test_timeline_most_common_blockers(self):
        """Timeline identifies most common blockers."""
        snapshots = [
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]},
        ]
        timeline = build_admissibility_timeline(snapshots)
        self.assertEqual(timeline["most_common_blockers"][0], "HE-GV1")

    def test_timeline_most_common_blockers_limited_to_5(self):
        """Timeline limits most_common_blockers to 5 entries."""
        snapshots = [
            {"verdict": "NOT_ADMISSIBLE", "error_codes": [f"ERR-{i}"]}
            for i in range(10)
        ]
        timeline = build_admissibility_timeline(snapshots)
        self.assertLessEqual(len(timeline["most_common_blockers"]), 5)

    def test_timeline_phase_failure_distribution(self):
        """Timeline tracks phase failure distribution."""
        snapshots = [
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"], "phases_executed": ["phase_1_governance"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"], "phases_executed": ["phase_1_governance"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"], "phases_executed": ["phase_1_governance", "phase_2_artifacts"]},
        ]
        timeline = build_admissibility_timeline(snapshots)
        self.assertEqual(timeline["phase_failure_distribution"]["phase_1_governance"], 2)
        self.assertEqual(timeline["phase_failure_distribution"]["phase_2_artifacts"], 1)

    def test_timeline_includes_schema_version(self):
        """Timeline includes schema version."""
        timeline = build_admissibility_timeline([])
        self.assertEqual(timeline["schema_version"], SNAPSHOT_SCHEMA_VERSION)

    def test_timeline_is_json_serializable(self):
        """Timeline is JSON serializable."""
        snapshots = [
            {"verdict": "ADMISSIBLE", "error_codes": []},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"], "phases_executed": ["phase_1_governance"]},
        ]
        timeline = build_admissibility_timeline(snapshots)
        json_str = json.dumps(timeline)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, timeline)

    def test_timeline_is_deterministic(self):
        """Timeline produces deterministic output."""
        snapshots = [
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1", "HE-GV1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
        ]
        timeline1 = build_admissibility_timeline(snapshots)
        timeline2 = build_admissibility_timeline(snapshots)
        self.assertEqual(timeline1, timeline2)


# =============================================================================
# TASK 3: GLOBAL HEALTH INTEGRATION TESTS
# =============================================================================

class TestGlobalHealthIntegration(unittest.TestCase):
    """Tests for summarize_admissibility_for_global_health function."""

    def test_admissible_returns_ok_status(self):
        """ADMISSIBLE verdict produces OK status."""
        snapshot = {
            "verdict": "ADMISSIBLE",
            "error_codes": [],
        }
        summary = summarize_admissibility_for_global_health(snapshot)
        self.assertEqual(summary["status"], "OK")
        self.assertTrue(summary["is_evidence_admissible"])
        self.assertEqual(summary["blocking_reasons"], [])

    def test_inadmissible_returns_blocked_status(self):
        """NOT_ADMISSIBLE verdict produces BLOCKED status."""
        snapshot = {
            "verdict": "NOT_ADMISSIBLE",
            "error_codes": ["HE-GV1"],
        }
        summary = summarize_admissibility_for_global_health(snapshot)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertFalse(summary["is_evidence_admissible"])

    def test_blocking_reasons_from_error_codes(self):
        """Blocking reasons come from error codes."""
        snapshot = {
            "verdict": "NOT_ADMISSIBLE",
            "error_codes": ["HE-GV1", "HE-GV2"],
        }
        summary = summarize_admissibility_for_global_health(snapshot)
        self.assertEqual(summary["blocking_reasons"], ["HE-GV1", "HE-GV2"])

    def test_blocking_reasons_limited_to_3(self):
        """Blocking reasons limited to 3 entries."""
        snapshot = {
            "verdict": "NOT_ADMISSIBLE",
            "error_codes": ["HE-GV1", "HE-GV2", "HE-S1", "HE-S3", "DOSSIER-11"],
        }
        summary = summarize_admissibility_for_global_health(snapshot)
        self.assertEqual(len(summary["blocking_reasons"]), 3)
        self.assertEqual(summary["blocking_reasons"], ["HE-GV1", "HE-GV2", "HE-S1"])

    def test_admissible_has_empty_blocking_reasons(self):
        """ADMISSIBLE verdict has empty blocking_reasons."""
        snapshot = {
            "verdict": "ADMISSIBLE",
            "error_codes": [],  # Should be empty anyway
        }
        summary = summarize_admissibility_for_global_health(snapshot)
        self.assertEqual(summary["blocking_reasons"], [])

    def test_he_gv_scenarios_produce_blocked(self):
        """HE-GV* error codes produce BLOCKED status."""
        for code in ["HE-GV1", "HE-GV2", "HE-GV3", "HE-GV4", "HE-GV5"]:
            with self.subTest(error_code=code):
                snapshot = {
                    "verdict": "NOT_ADMISSIBLE",
                    "error_codes": [code],
                }
                summary = summarize_admissibility_for_global_health(snapshot)
                self.assertEqual(summary["status"], "BLOCKED")
                self.assertIn(code, summary["blocking_reasons"])

    def test_he_s_scenarios_produce_blocked(self):
        """HE-S* error codes produce BLOCKED status."""
        for code in ["HE-S1", "HE-S2", "HE-S3", "HE-S4", "HE-S5"]:
            with self.subTest(error_code=code):
                snapshot = {
                    "verdict": "NOT_ADMISSIBLE",
                    "error_codes": [code],
                }
                summary = summarize_admissibility_for_global_health(snapshot)
                self.assertEqual(summary["status"], "BLOCKED")

    def test_he_i_scenarios_produce_blocked(self):
        """HE-I* error codes produce BLOCKED status."""
        for code in ["HE-I1", "HE-I2", "HE-I3", "HE-I4"]:
            with self.subTest(error_code=code):
                snapshot = {
                    "verdict": "NOT_ADMISSIBLE",
                    "error_codes": [code],
                }
                summary = summarize_admissibility_for_global_health(snapshot)
                self.assertEqual(summary["status"], "BLOCKED")

    def test_summary_is_json_serializable(self):
        """Summary is JSON serializable for embedding in global_health.json."""
        snapshot = {
            "verdict": "NOT_ADMISSIBLE",
            "error_codes": ["HE-GV1", "HE-S1"],
        }
        summary = summarize_admissibility_for_global_health(snapshot)
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, summary)

    def test_summary_structure_for_director_console(self):
        """Summary has correct structure for Director's Console."""
        snapshot = {
            "verdict": "NOT_ADMISSIBLE",
            "error_codes": ["HE-GV1"],
        }
        summary = summarize_admissibility_for_global_health(snapshot)

        # Verify required fields exist
        self.assertIn("is_evidence_admissible", summary)
        self.assertIn("blocking_reasons", summary)
        self.assertIn("status", summary)

        # Verify types
        self.assertIsInstance(summary["is_evidence_admissible"], bool)
        self.assertIsInstance(summary["blocking_reasons"], list)
        self.assertIn(summary["status"], ["OK", "BLOCKED"])


# =============================================================================
# PHASE III: ADMISSIBILITY ANALYTICS TESTS
# =============================================================================

class TestAdmissibilityAnalytics(unittest.TestCase):
    """Tests for build_admissibility_analytics function."""

    def test_analytics_calculates_long_run_rate(self):
        """Analytics computes long_run_admissibility_rate correctly."""
        timeline = {
            "total_cases": 10,
            "admissible_count": 7,
            "inadmissible_count": 3,
            "error_code_frequency": {},
            "phase_failure_distribution": {},
        }
        analytics = build_admissibility_analytics(timeline)
        self.assertEqual(analytics["long_run_admissibility_rate"], 0.7)

    def test_analytics_handles_empty_timeline(self):
        """Analytics handles empty timeline gracefully."""
        timeline = {
            "total_cases": 0,
            "admissible_count": 0,
            "inadmissible_count": 0,
            "error_code_frequency": {},
            "phase_failure_distribution": {},
        }
        analytics = build_admissibility_analytics(timeline)
        self.assertEqual(analytics["long_run_admissibility_rate"], 0.0)
        self.assertEqual(analytics["recurrent_errors"], [])
        self.assertIsNone(analytics["dominant_failure_phase"])

    def test_analytics_identifies_recurrent_errors(self):
        """Analytics identifies recurrent errors (>= 30% of failures)."""
        timeline = {
            "total_cases": 10,
            "admissible_count": 0,
            "inadmissible_count": 10,
            "error_code_frequency": {
                "HE-GV1": 5,  # 50% - recurrent
                "HE-S1": 3,   # 30% - recurrent
                "HE-S2": 2,   # 20% - not recurrent
            },
            "phase_failure_distribution": {},
        }
        analytics = build_admissibility_analytics(timeline)
        self.assertIn("HE-GV1", analytics["recurrent_errors"])
        self.assertIn("HE-S1", analytics["recurrent_errors"])
        self.assertNotIn("HE-S2", analytics["recurrent_errors"])

    def test_analytics_recurrent_errors_sorted(self):
        """Analytics recurrent_errors are sorted alphabetically."""
        timeline = {
            "total_cases": 10,
            "admissible_count": 0,
            "inadmissible_count": 10,
            "error_code_frequency": {
                "HE-S1": 5,
                "HE-GV1": 5,
                "DOSSIER-11": 4,
            },
            "phase_failure_distribution": {},
        }
        analytics = build_admissibility_analytics(timeline)
        self.assertEqual(analytics["recurrent_errors"], ["DOSSIER-11", "HE-GV1", "HE-S1"])

    def test_analytics_identifies_dominant_failure_phase(self):
        """Analytics identifies the phase with most failures."""
        timeline = {
            "total_cases": 10,
            "admissible_count": 5,
            "inadmissible_count": 5,
            "error_code_frequency": {},
            "phase_failure_distribution": {
                "phase_1_governance": 1,
                "phase_2_artifacts": 3,
                "phase_3_structure": 1,
            },
        }
        analytics = build_admissibility_analytics(timeline)
        self.assertEqual(analytics["dominant_failure_phase"], "phase_2_artifacts")

    def test_analytics_calculates_failure_concentration(self):
        """Analytics calculates how concentrated failures are in dominant phase."""
        timeline = {
            "total_cases": 10,
            "admissible_count": 0,
            "inadmissible_count": 10,
            "error_code_frequency": {},
            "phase_failure_distribution": {
                "phase_1_governance": 8,
                "phase_2_artifacts": 2,
            },
        }
        analytics = build_admissibility_analytics(timeline)
        self.assertEqual(analytics["failure_concentration"], 0.8)

    def test_analytics_counts_error_diversity(self):
        """Analytics counts unique error codes as error_diversity."""
        timeline = {
            "total_cases": 10,
            "admissible_count": 5,
            "inadmissible_count": 5,
            "error_code_frequency": {
                "HE-GV1": 2,
                "HE-S1": 1,
                "DOSSIER-11": 2,
            },
            "phase_failure_distribution": {},
        }
        analytics = build_admissibility_analytics(timeline)
        self.assertEqual(analytics["error_diversity"], 3)

    def test_analytics_is_json_serializable(self):
        """Analytics output is JSON serializable."""
        timeline = {
            "total_cases": 5,
            "admissible_count": 3,
            "inadmissible_count": 2,
            "error_code_frequency": {"HE-GV1": 2},
            "phase_failure_distribution": {"phase_1_governance": 2},
        }
        analytics = build_admissibility_analytics(timeline)
        json_str = json.dumps(analytics)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, analytics)


# =============================================================================
# PHASE III: DIRECTOR LIGHT MAPPING TESTS
# =============================================================================

class TestDirectorLightMapping(unittest.TestCase):
    """Tests for map_admissibility_to_director_light function."""

    def test_admissible_maps_to_green(self):
        """ADMISSIBLE verdict maps to GREEN."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        self.assertEqual(map_admissibility_to_director_light(snapshot), "GREEN")

    def test_he_gv_errors_map_to_red(self):
        """HE-GV* (governance) errors map to RED."""
        for code in ["HE-GV1", "HE-GV2", "HE-GV3", "HE-GV4", "HE-GV5"]:
            with self.subTest(error_code=code):
                snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": [code]}
                self.assertEqual(map_admissibility_to_director_light(snapshot), "RED")

    def test_he_i_errors_map_to_red(self):
        """HE-I* (integrity) errors map to RED."""
        for code in ["HE-I1", "HE-I2", "HE-I3", "HE-I4"]:
            with self.subTest(error_code=code):
                snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": [code]}
                self.assertEqual(map_admissibility_to_director_light(snapshot), "RED")

    def test_he_s_errors_map_to_yellow(self):
        """HE-S* (structural) errors map to YELLOW."""
        for code in ["HE-S1", "HE-S2", "HE-S3", "HE-S4", "HE-S5"]:
            with self.subTest(error_code=code):
                snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": [code]}
                self.assertEqual(map_admissibility_to_director_light(snapshot), "YELLOW")

    def test_dossier_errors_map_to_yellow(self):
        """DOSSIER-* errors map to YELLOW."""
        for code in ["DOSSIER-3", "DOSSIER-8", "DOSSIER-9", "DOSSIER-10", "DOSSIER-11"]:
            with self.subTest(error_code=code):
                snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": [code]}
                self.assertEqual(map_admissibility_to_director_light(snapshot), "YELLOW")

    def test_rel_errors_map_to_yellow(self):
        """REL-* errors map to YELLOW."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["REL-3"]}
        self.assertEqual(map_admissibility_to_director_light(snapshot), "YELLOW")

    def test_mixed_errors_with_critical_maps_to_red(self):
        """Mixed errors with any critical (HE-GV/HE-I) maps to RED."""
        snapshot = {
            "verdict": "NOT_ADMISSIBLE",
            "error_codes": ["HE-S1", "HE-GV1", "DOSSIER-11"],  # HE-GV1 is critical
        }
        self.assertEqual(map_admissibility_to_director_light(snapshot), "RED")

    def test_mixed_non_critical_errors_map_to_yellow(self):
        """Mixed non-critical errors map to YELLOW."""
        snapshot = {
            "verdict": "NOT_ADMISSIBLE",
            "error_codes": ["HE-S1", "HE-S3", "DOSSIER-11"],
        }
        self.assertEqual(map_admissibility_to_director_light(snapshot), "YELLOW")

    def test_unknown_verdict_with_no_errors_maps_to_yellow(self):
        """Unknown verdict with no errors maps to YELLOW (default fallback)."""
        snapshot = {"verdict": "UNKNOWN", "error_codes": []}
        self.assertEqual(map_admissibility_to_director_light(snapshot), "YELLOW")


# =============================================================================
# PHASE III: GLOBAL DASHBOARD SUMMARY TESTS
# =============================================================================

class TestGlobalDashboardSummary(unittest.TestCase):
    """Tests for summarize_admissibility_for_global_dashboard function."""

    def test_admissible_produces_green_summary(self):
        """ADMISSIBLE snapshot produces green status_light."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        summary = summarize_admissibility_for_global_dashboard(snapshot)
        self.assertTrue(summary["admissible"])
        self.assertEqual(summary["blockers"], [])
        self.assertEqual(summary["status_light"], "GREEN")

    def test_inadmissible_governance_produces_red_summary(self):
        """Governance failure produces red status_light."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]}
        summary = summarize_admissibility_for_global_dashboard(snapshot)
        self.assertFalse(summary["admissible"])
        self.assertEqual(summary["blockers"], ["HE-GV1"])
        self.assertEqual(summary["status_light"], "RED")

    def test_inadmissible_structural_produces_yellow_summary(self):
        """Structural failure produces yellow status_light."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]}
        summary = summarize_admissibility_for_global_dashboard(snapshot)
        self.assertFalse(summary["admissible"])
        self.assertEqual(summary["blockers"], ["HE-S1"])
        self.assertEqual(summary["status_light"], "YELLOW")

    def test_blockers_limited_to_3(self):
        """Blockers limited to top 3 error codes."""
        snapshot = {
            "verdict": "NOT_ADMISSIBLE",
            "error_codes": ["HE-S1", "HE-S2", "HE-S3", "HE-S4", "HE-S5"],
        }
        summary = summarize_admissibility_for_global_dashboard(snapshot)
        self.assertEqual(len(summary["blockers"]), 3)
        self.assertEqual(summary["blockers"], ["HE-S1", "HE-S2", "HE-S3"])

    def test_summary_has_required_fields(self):
        """Summary contains all required fields."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        summary = summarize_admissibility_for_global_dashboard(snapshot)
        self.assertIn("admissible", summary)
        self.assertIn("blockers", summary)
        self.assertIn("status_light", summary)

    def test_summary_is_json_serializable(self):
        """Summary is JSON serializable."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]}
        summary = summarize_admissibility_for_global_dashboard(snapshot)
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, summary)

    def test_status_light_values_are_valid(self):
        """status_light is always GREEN, YELLOW, or RED."""
        test_cases = [
            ({"verdict": "ADMISSIBLE", "error_codes": []}, "GREEN"),
            ({"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]}, "RED"),
            ({"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]}, "YELLOW"),
        ]
        for snapshot, expected_light in test_cases:
            with self.subTest(snapshot=snapshot):
                summary = summarize_admissibility_for_global_dashboard(snapshot)
                self.assertEqual(summary["status_light"], expected_light)


# =============================================================================
# PHASE IV: ADMISSIBILITY COMPASS TESTS
# =============================================================================

class TestAdmissibilityCompass(unittest.TestCase):
    """Tests for build_admissibility_compass function."""

    def test_compass_stable_with_high_rate_no_errors(self):
        """STABLE when admissibility rate >= 80% and no recurrent errors."""
        analytics = {
            "long_run_admissibility_rate": 0.85,
            "recurrent_errors": [],
            "dominant_failure_phase": None,
            "failure_concentration": 0.0,
        }
        compass = build_admissibility_compass(analytics)
        self.assertEqual(compass["compass_status"], "STABLE")

    def test_compass_degrading_with_high_rate_but_recurrent_errors(self):
        """DEGRADING when high rate but has recurrent errors."""
        analytics = {
            "long_run_admissibility_rate": 0.9,
            "recurrent_errors": ["HE-S1"],
            "dominant_failure_phase": "phase_2_artifacts",
            "failure_concentration": 0.5,
        }
        compass = build_admissibility_compass(analytics)
        self.assertEqual(compass["compass_status"], "DEGRADING")

    def test_compass_degrading_with_medium_rate(self):
        """DEGRADING when rate is between 50% and 80%."""
        analytics = {
            "long_run_admissibility_rate": 0.65,
            "recurrent_errors": [],
            "dominant_failure_phase": "phase_1_governance",
            "failure_concentration": 0.3,
        }
        compass = build_admissibility_compass(analytics)
        self.assertEqual(compass["compass_status"], "DEGRADING")

    def test_compass_critical_with_low_rate(self):
        """CRITICAL when rate < 50%."""
        analytics = {
            "long_run_admissibility_rate": 0.4,
            "recurrent_errors": ["HE-GV1", "HE-I1"],
            "dominant_failure_phase": "phase_1_governance",
            "failure_concentration": 0.8,
        }
        compass = build_admissibility_compass(analytics)
        self.assertEqual(compass["compass_status"], "CRITICAL")

    def test_compass_includes_dominant_failure_phase(self):
        """Compass passes through dominant_failure_phase."""
        analytics = {
            "long_run_admissibility_rate": 0.7,
            "recurrent_errors": [],
            "dominant_failure_phase": "phase_3_structure",
            "failure_concentration": 0.6,
        }
        compass = build_admissibility_compass(analytics)
        self.assertEqual(compass["dominant_failure_phase"], "phase_3_structure")

    def test_compass_includes_recurrent_errors(self):
        """Compass passes through recurrent_errors list."""
        analytics = {
            "long_run_admissibility_rate": 0.6,
            "recurrent_errors": ["HE-S1", "DOSSIER-11"],
            "dominant_failure_phase": None,
            "failure_concentration": 0.0,
        }
        compass = build_admissibility_compass(analytics)
        self.assertEqual(compass["recurrent_errors"], ["HE-S1", "DOSSIER-11"])

    def test_compass_includes_failure_concentration(self):
        """Compass passes through failure_concentration."""
        analytics = {
            "long_run_admissibility_rate": 0.5,
            "recurrent_errors": [],
            "dominant_failure_phase": "phase_2_artifacts",
            "failure_concentration": 0.75,
        }
        compass = build_admissibility_compass(analytics)
        self.assertEqual(compass["failure_concentration"], 0.75)

    def test_compass_has_required_fields(self):
        """Compass contains all required fields."""
        analytics = {
            "long_run_admissibility_rate": 0.8,
            "recurrent_errors": [],
            "dominant_failure_phase": None,
            "failure_concentration": 0.0,
        }
        compass = build_admissibility_compass(analytics)
        self.assertIn("compass_status", compass)
        self.assertIn("dominant_failure_phase", compass)
        self.assertIn("recurrent_errors", compass)
        self.assertIn("failure_concentration", compass)

    def test_compass_is_json_serializable(self):
        """Compass is JSON serializable."""
        analytics = {
            "long_run_admissibility_rate": 0.7,
            "recurrent_errors": ["HE-S1"],
            "dominant_failure_phase": "phase_1_governance",
            "failure_concentration": 0.5,
        }
        compass = build_admissibility_compass(analytics)
        json_str = json.dumps(compass)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, compass)

    def test_compass_handles_empty_analytics(self):
        """Compass handles empty/minimal analytics gracefully."""
        analytics = {}
        compass = build_admissibility_compass(analytics)
        self.assertEqual(compass["compass_status"], "CRITICAL")
        self.assertIsNone(compass["dominant_failure_phase"])
        self.assertEqual(compass["recurrent_errors"], [])
        self.assertEqual(compass["failure_concentration"], 0.0)


# =============================================================================
# PHASE IV: PROMOTION EVALUATION TESTS
# =============================================================================

class TestPromotionEvaluation(unittest.TestCase):
    """Tests for evaluate_admissibility_for_promotion function."""

    def test_stable_compass_produces_ok(self):
        """STABLE compass produces OK promotion status."""
        analytics = {"long_run_admissibility_rate": 0.9}
        compass = {"compass_status": "STABLE", "recurrent_errors": []}
        result = evaluate_admissibility_for_promotion(analytics, compass)
        self.assertTrue(result["promotion_ok"])
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["blocking_error_codes"], [])

    def test_degrading_compass_without_blockers_produces_warn(self):
        """DEGRADING compass without blocking errors produces WARN."""
        analytics = {"long_run_admissibility_rate": 0.7}
        compass = {
            "compass_status": "DEGRADING",
            "recurrent_errors": ["HE-S1", "DOSSIER-11"],  # Non-blocking
        }
        result = evaluate_admissibility_for_promotion(analytics, compass)
        self.assertTrue(result["promotion_ok"])
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["blocking_error_codes"], [])

    def test_degrading_compass_with_blocking_errors_produces_block(self):
        """DEGRADING compass with blocking errors (HE-GV/HE-I) produces BLOCK."""
        analytics = {"long_run_admissibility_rate": 0.7}
        compass = {
            "compass_status": "DEGRADING",
            "recurrent_errors": ["HE-S1", "HE-GV1"],  # HE-GV1 is blocking
        }
        result = evaluate_admissibility_for_promotion(analytics, compass)
        self.assertFalse(result["promotion_ok"])
        self.assertEqual(result["status"], "BLOCK")
        self.assertIn("HE-GV1", result["blocking_error_codes"])

    def test_critical_compass_produces_block(self):
        """CRITICAL compass always produces BLOCK."""
        analytics = {"long_run_admissibility_rate": 0.3}
        compass = {"compass_status": "CRITICAL", "recurrent_errors": []}
        result = evaluate_admissibility_for_promotion(analytics, compass)
        self.assertFalse(result["promotion_ok"])
        self.assertEqual(result["status"], "BLOCK")

    def test_blocking_codes_include_he_gv(self):
        """HE-GV* codes are identified as blocking."""
        analytics = {}
        compass = {
            "compass_status": "DEGRADING",
            "recurrent_errors": ["HE-GV1", "HE-GV2", "HE-S1"],
        }
        result = evaluate_admissibility_for_promotion(analytics, compass)
        self.assertIn("HE-GV1", result["blocking_error_codes"])
        self.assertIn("HE-GV2", result["blocking_error_codes"])
        self.assertNotIn("HE-S1", result["blocking_error_codes"])

    def test_blocking_codes_include_he_i(self):
        """HE-I* codes are identified as blocking."""
        analytics = {}
        compass = {
            "compass_status": "DEGRADING",
            "recurrent_errors": ["HE-I1", "HE-I3", "DOSSIER-11"],
        }
        result = evaluate_admissibility_for_promotion(analytics, compass)
        self.assertIn("HE-I1", result["blocking_error_codes"])
        self.assertIn("HE-I3", result["blocking_error_codes"])
        self.assertNotIn("DOSSIER-11", result["blocking_error_codes"])

    def test_blocking_codes_sorted(self):
        """Blocking error codes are sorted alphabetically."""
        analytics = {}
        compass = {
            "compass_status": "DEGRADING",
            "recurrent_errors": ["HE-I3", "HE-GV1", "HE-I1", "HE-GV2"],
        }
        result = evaluate_admissibility_for_promotion(analytics, compass)
        self.assertEqual(
            result["blocking_error_codes"],
            ["HE-GV1", "HE-GV2", "HE-I1", "HE-I3"],
        )

    def test_result_has_required_fields(self):
        """Result contains all required fields."""
        analytics = {}
        compass = {"compass_status": "STABLE", "recurrent_errors": []}
        result = evaluate_admissibility_for_promotion(analytics, compass)
        self.assertIn("promotion_ok", result)
        self.assertIn("status", result)
        self.assertIn("blocking_error_codes", result)

    def test_result_is_json_serializable(self):
        """Result is JSON serializable."""
        analytics = {}
        compass = {
            "compass_status": "DEGRADING",
            "recurrent_errors": ["HE-GV1", "HE-S1"],
        }
        result = evaluate_admissibility_for_promotion(analytics, compass)
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, result)

    def test_status_values_are_valid(self):
        """Status is always OK, WARN, or BLOCK."""
        test_cases = [
            ({"compass_status": "STABLE", "recurrent_errors": []}, "OK"),
            ({"compass_status": "DEGRADING", "recurrent_errors": []}, "WARN"),
            ({"compass_status": "DEGRADING", "recurrent_errors": ["HE-GV1"]}, "BLOCK"),
            ({"compass_status": "CRITICAL", "recurrent_errors": []}, "BLOCK"),
        ]
        for compass, expected_status in test_cases:
            with self.subTest(compass=compass):
                result = evaluate_admissibility_for_promotion({}, compass)
                self.assertEqual(result["status"], expected_status)


# =============================================================================
# PHASE IV: DIRECTOR PANEL TESTS
# =============================================================================

class TestDirectorPanel(unittest.TestCase):
    """Tests for build_admissibility_director_panel function."""

    def test_panel_includes_status_light(self):
        """Panel includes status_light from snapshot."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        compass = {"compass_status": "STABLE"}
        promotion_eval = {"status": "OK"}
        panel = build_admissibility_director_panel(snapshot, compass, promotion_eval)
        self.assertEqual(panel["status_light"], "GREEN")

    def test_panel_headline_for_stable_green(self):
        """Stable + green produces normal headline."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        compass = {"compass_status": "STABLE"}
        promotion_eval = {"status": "OK"}
        panel = build_admissibility_director_panel(snapshot, compass, promotion_eval)
        self.assertEqual(panel["headline"], "Evidence pipeline operating normally")

    def test_panel_headline_for_green_with_degrading(self):
        """Green + degrading produces degradation headline."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        compass = {"compass_status": "DEGRADING"}
        promotion_eval = {"status": "WARN"}
        panel = build_admissibility_director_panel(snapshot, compass, promotion_eval)
        self.assertIn("degradation", panel["headline"].lower())

    def test_panel_headline_for_critical(self):
        """Critical compass produces attention headline."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]}
        compass = {"compass_status": "CRITICAL"}
        promotion_eval = {"status": "BLOCK"}
        panel = build_admissibility_director_panel(snapshot, compass, promotion_eval)
        self.assertIn("immediate attention", panel["headline"].lower())

    def test_panel_headline_for_blocked_promotion(self):
        """Blocked promotion produces blocked headline."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]}
        compass = {"compass_status": "DEGRADING"}
        promotion_eval = {"status": "BLOCK"}
        panel = build_admissibility_director_panel(snapshot, compass, promotion_eval)
        self.assertIn("blocked", panel["headline"].lower())

    def test_panel_includes_long_run_rate_when_present(self):
        """Panel includes long_run_admissibility_rate if in snapshot."""
        snapshot = {
            "verdict": "ADMISSIBLE",
            "error_codes": [],
            "long_run_admissibility_rate": 0.85,
        }
        compass = {"compass_status": "STABLE"}
        promotion_eval = {"status": "OK"}
        panel = build_admissibility_director_panel(snapshot, compass, promotion_eval)
        self.assertEqual(panel["long_run_admissibility_rate"], 0.85)

    def test_panel_omits_long_run_rate_when_absent(self):
        """Panel omits long_run_admissibility_rate if not in snapshot."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        compass = {"compass_status": "STABLE"}
        promotion_eval = {"status": "OK"}
        panel = build_admissibility_director_panel(snapshot, compass, promotion_eval)
        self.assertNotIn("long_run_admissibility_rate", panel)

    def test_panel_has_required_fields(self):
        """Panel contains all required fields."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        compass = {"compass_status": "STABLE"}
        promotion_eval = {"status": "OK"}
        panel = build_admissibility_director_panel(snapshot, compass, promotion_eval)
        self.assertIn("status_light", panel)
        self.assertIn("headline", panel)

    def test_panel_is_json_serializable(self):
        """Panel is JSON serializable."""
        snapshot = {
            "verdict": "NOT_ADMISSIBLE",
            "error_codes": ["HE-GV1"],
            "long_run_admissibility_rate": 0.7,
        }
        compass = {"compass_status": "DEGRADING"}
        promotion_eval = {"status": "BLOCK"}
        panel = build_admissibility_director_panel(snapshot, compass, promotion_eval)
        json_str = json.dumps(panel)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, panel)

    def test_status_light_values_are_valid(self):
        """status_light is always GREEN, YELLOW, or RED."""
        test_cases = [
            {"verdict": "ADMISSIBLE", "error_codes": []},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]},
            {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]},
        ]
        for snapshot in test_cases:
            with self.subTest(snapshot=snapshot):
                panel = build_admissibility_director_panel(
                    snapshot,
                    {"compass_status": "STABLE"},
                    {"status": "OK"},
                )
                self.assertIn(panel["status_light"], ["GREEN", "YELLOW", "RED"])


# =============================================================================
# PHASE V: GLOBAL CONSOLE SUMMARY TESTS
# =============================================================================

class TestGlobalConsoleSummary(unittest.TestCase):
    """Tests for summarize_admissibility_for_global_console function."""

    def test_admissible_stable_produces_normal_headline(self):
        """Admissible + STABLE produces normal operation headline."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        compass = {"compass_status": "STABLE", "dominant_failure_phase": None}
        promotion_eval = {"status": "OK"}
        summary = summarize_admissibility_for_global_console(snapshot, compass, promotion_eval)
        self.assertTrue(summary["admissible"])
        self.assertEqual(summary["status_light"], "GREEN")
        self.assertIn("normally", summary["headline"].lower())

    def test_admissible_degrading_shows_degradation(self):
        """Admissible + DEGRADING shows degradation in headline."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        compass = {"compass_status": "DEGRADING", "dominant_failure_phase": "phase_1_governance"}
        promotion_eval = {"status": "WARN"}
        summary = summarize_admissibility_for_global_console(snapshot, compass, promotion_eval)
        self.assertTrue(summary["admissible"])
        self.assertIn("degradation", summary["headline"].lower())

    def test_critical_compass_shows_immediate_attention(self):
        """CRITICAL compass shows immediate attention in headline."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]}
        compass = {"compass_status": "CRITICAL", "dominant_failure_phase": "phase_1_governance"}
        promotion_eval = {"status": "BLOCK"}
        summary = summarize_admissibility_for_global_console(snapshot, compass, promotion_eval)
        self.assertFalse(summary["admissible"])
        self.assertIn("immediate attention", summary["headline"].lower())

    def test_blocked_promotion_shows_blocked(self):
        """Blocked promotion shows blocked in headline."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]}
        compass = {"compass_status": "DEGRADING", "dominant_failure_phase": "phase_2_artifacts"}
        promotion_eval = {"status": "BLOCK"}
        summary = summarize_admissibility_for_global_console(snapshot, compass, promotion_eval)
        self.assertIn("blocked", summary["headline"].lower())

    def test_summary_includes_dominant_failure_phase(self):
        """Summary includes dominant_failure_phase from compass."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]}
        compass = {"compass_status": "DEGRADING", "dominant_failure_phase": "phase_3_structure"}
        promotion_eval = {"status": "WARN"}
        summary = summarize_admissibility_for_global_console(snapshot, compass, promotion_eval)
        self.assertEqual(summary["dominant_failure_phase"], "phase_3_structure")

    def test_summary_has_required_fields(self):
        """Summary contains all required fields."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        compass = {"compass_status": "STABLE", "dominant_failure_phase": None}
        promotion_eval = {"status": "OK"}
        summary = summarize_admissibility_for_global_console(snapshot, compass, promotion_eval)
        self.assertIn("admissible", summary)
        self.assertIn("status_light", summary)
        self.assertIn("dominant_failure_phase", summary)
        self.assertIn("headline", summary)

    def test_summary_is_json_serializable(self):
        """Summary is JSON serializable."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]}
        compass = {"compass_status": "CRITICAL", "dominant_failure_phase": "phase_1_governance"}
        promotion_eval = {"status": "BLOCK"}
        summary = summarize_admissibility_for_global_console(snapshot, compass, promotion_eval)
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, summary)

    def test_status_light_matches_snapshot(self):
        """status_light is derived from snapshot verdict."""
        test_cases = [
            ({"verdict": "ADMISSIBLE", "error_codes": []}, "GREEN"),
            ({"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]}, "RED"),
            ({"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]}, "YELLOW"),
        ]
        compass = {"compass_status": "STABLE", "dominant_failure_phase": None}
        promotion_eval = {"status": "OK"}
        for snapshot, expected_light in test_cases:
            with self.subTest(snapshot=snapshot):
                summary = summarize_admissibility_for_global_console(snapshot, compass, promotion_eval)
                self.assertEqual(summary["status_light"], expected_light)


# =============================================================================
# PHASE V: EVIDENCE CHAIN HOOK TESTS
# =============================================================================

class TestEvidenceChainHook(unittest.TestCase):
    """Tests for attach_admissibility_to_evidence_chain function."""

    def test_attaches_admissibility_compass_subtree(self):
        """Attaches admissibility_compass subtree to evidence chain."""
        evidence_chain = {"run_id": "test-123", "artifacts": []}
        compass = {"compass_status": "STABLE", "recurrent_errors": [], "failure_concentration": 0.0}
        promotion_eval = {"promotion_ok": True, "status": "OK", "blocking_error_codes": []}
        result = attach_admissibility_to_evidence_chain(evidence_chain, compass, promotion_eval)
        self.assertIn("admissibility_compass", result)

    def test_subtree_contains_status(self):
        """Subtree contains compass status."""
        evidence_chain = {}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": [], "failure_concentration": 0.3}
        promotion_eval = {"promotion_ok": True, "status": "WARN", "blocking_error_codes": []}
        result = attach_admissibility_to_evidence_chain(evidence_chain, compass, promotion_eval)
        self.assertEqual(result["admissibility_compass"]["status"], "DEGRADING")

    def test_subtree_contains_recurrent_errors(self):
        """Subtree contains recurrent_errors list."""
        evidence_chain = {}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": ["HE-S1", "HE-S2"], "failure_concentration": 0.5}
        promotion_eval = {"promotion_ok": True, "status": "WARN", "blocking_error_codes": []}
        result = attach_admissibility_to_evidence_chain(evidence_chain, compass, promotion_eval)
        self.assertEqual(result["admissibility_compass"]["recurrent_errors"], ["HE-S1", "HE-S2"])

    def test_subtree_contains_failure_concentration(self):
        """Subtree contains failure_concentration."""
        evidence_chain = {}
        compass = {"compass_status": "CRITICAL", "recurrent_errors": [], "failure_concentration": 0.85}
        promotion_eval = {"promotion_ok": False, "status": "BLOCK", "blocking_error_codes": []}
        result = attach_admissibility_to_evidence_chain(evidence_chain, compass, promotion_eval)
        self.assertEqual(result["admissibility_compass"]["failure_concentration"], 0.85)

    def test_subtree_contains_promotion_blocked(self):
        """Subtree contains promotion_blocked flag."""
        evidence_chain = {}
        compass = {"compass_status": "CRITICAL", "recurrent_errors": [], "failure_concentration": 0.0}
        promotion_eval = {"promotion_ok": False, "status": "BLOCK", "blocking_error_codes": []}
        result = attach_admissibility_to_evidence_chain(evidence_chain, compass, promotion_eval)
        self.assertTrue(result["admissibility_compass"]["promotion_blocked"])

    def test_subtree_contains_blocking_error_codes(self):
        """Subtree contains blocking_error_codes list."""
        evidence_chain = {}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": ["HE-GV1"], "failure_concentration": 0.6}
        promotion_eval = {"promotion_ok": False, "status": "BLOCK", "blocking_error_codes": ["HE-GV1"]}
        result = attach_admissibility_to_evidence_chain(evidence_chain, compass, promotion_eval)
        self.assertEqual(result["admissibility_compass"]["blocking_error_codes"], ["HE-GV1"])

    def test_does_not_mutate_original_chain(self):
        """Does not mutate the original evidence chain."""
        evidence_chain = {"run_id": "test-123"}
        compass = {"compass_status": "STABLE", "recurrent_errors": [], "failure_concentration": 0.0}
        promotion_eval = {"promotion_ok": True, "status": "OK", "blocking_error_codes": []}
        result = attach_admissibility_to_evidence_chain(evidence_chain, compass, promotion_eval)
        self.assertNotIn("admissibility_compass", evidence_chain)
        self.assertIn("admissibility_compass", result)

    def test_preserves_existing_chain_data(self):
        """Preserves existing evidence chain data."""
        evidence_chain = {"run_id": "test-123", "artifacts": ["A1", "A2"], "metadata": {"key": "value"}}
        compass = {"compass_status": "STABLE", "recurrent_errors": [], "failure_concentration": 0.0}
        promotion_eval = {"promotion_ok": True, "status": "OK", "blocking_error_codes": []}
        result = attach_admissibility_to_evidence_chain(evidence_chain, compass, promotion_eval)
        self.assertEqual(result["run_id"], "test-123")
        self.assertEqual(result["artifacts"], ["A1", "A2"])
        self.assertEqual(result["metadata"], {"key": "value"})

    def test_result_is_json_serializable(self):
        """Result is JSON serializable."""
        evidence_chain = {"run_id": "test-123"}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": ["HE-S1"], "failure_concentration": 0.5}
        promotion_eval = {"promotion_ok": True, "status": "WARN", "blocking_error_codes": []}
        result = attach_admissibility_to_evidence_chain(evidence_chain, compass, promotion_eval)
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, result)

    def test_block_cascades_to_promotion_blocked(self):
        """BLOCK status cascades to promotion_blocked=True in subtree."""
        evidence_chain = {}
        compass = {"compass_status": "CRITICAL", "recurrent_errors": ["HE-GV1"], "failure_concentration": 0.9}
        promotion_eval = {"promotion_ok": False, "status": "BLOCK", "blocking_error_codes": ["HE-GV1"]}
        result = attach_admissibility_to_evidence_chain(evidence_chain, compass, promotion_eval)
        self.assertTrue(result["admissibility_compass"]["promotion_blocked"])
        self.assertEqual(result["admissibility_compass"]["status"], "CRITICAL")


# =============================================================================
# PHASE V: GOVERNANCE SIGNAL TESTS
# =============================================================================

class TestGovernanceSignal(unittest.TestCase):
    """Tests for to_governance_signal function."""

    def test_stable_ok_produces_proceed(self):
        """STABLE + OK produces PROCEED recommendation."""
        analytics = {"long_run_admissibility_rate": 0.9}
        compass = {"compass_status": "STABLE", "recurrent_errors": [], "dominant_failure_phase": None}
        promotion_eval = {"promotion_ok": True, "status": "OK", "blocking_error_codes": []}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["recommendation"], "PROCEED")
        self.assertEqual(signal["confidence"], "HIGH")

    def test_degrading_warn_produces_proceed_with_caution(self):
        """DEGRADING + WARN produces PROCEED_WITH_CAUTION."""
        analytics = {"long_run_admissibility_rate": 0.7}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": [], "dominant_failure_phase": "phase_2_artifacts"}
        promotion_eval = {"promotion_ok": True, "status": "WARN", "blocking_error_codes": []}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["recommendation"], "PROCEED_WITH_CAUTION")
        self.assertEqual(signal["confidence"], "MEDIUM")

    def test_critical_produces_halt(self):
        """CRITICAL compass produces HALT recommendation."""
        analytics = {"long_run_admissibility_rate": 0.3}
        compass = {"compass_status": "CRITICAL", "recurrent_errors": [], "dominant_failure_phase": "phase_1_governance"}
        promotion_eval = {"promotion_ok": False, "status": "BLOCK", "blocking_error_codes": []}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["recommendation"], "HALT")
        self.assertEqual(signal["confidence"], "HIGH")

    def test_blocked_produces_block_recommendation(self):
        """Blocked promotion produces BLOCK recommendation."""
        analytics = {"long_run_admissibility_rate": 0.7}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": ["HE-GV1"], "dominant_failure_phase": "phase_1_governance"}
        promotion_eval = {"promotion_ok": False, "status": "BLOCK", "blocking_error_codes": ["HE-GV1"]}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["recommendation"], "BLOCK")
        self.assertEqual(signal["confidence"], "HIGH")

    def test_signal_type_is_admissibility_gate(self):
        """Signal type is always ADMISSIBILITY_GATE."""
        analytics = {}
        compass = {"compass_status": "STABLE", "recurrent_errors": [], "dominant_failure_phase": None}
        promotion_eval = {"promotion_ok": True, "status": "OK", "blocking_error_codes": []}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["signal_type"], "ADMISSIBILITY_GATE")

    def test_signal_includes_long_run_rate(self):
        """Signal includes long_run_admissibility_rate."""
        analytics = {"long_run_admissibility_rate": 0.85}
        compass = {"compass_status": "STABLE", "recurrent_errors": [], "dominant_failure_phase": None}
        promotion_eval = {"promotion_ok": True, "status": "OK", "blocking_error_codes": []}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["long_run_admissibility_rate"], 0.85)

    def test_signal_includes_blocking_error_codes(self):
        """Signal includes blocking_error_codes."""
        analytics = {}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": ["HE-GV1", "HE-I2"], "dominant_failure_phase": None}
        promotion_eval = {"promotion_ok": False, "status": "BLOCK", "blocking_error_codes": ["HE-GV1", "HE-I2"]}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["blocking_error_codes"], ["HE-GV1", "HE-I2"])

    def test_signal_includes_recurrent_errors(self):
        """Signal includes recurrent_errors list."""
        analytics = {}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": ["HE-S1", "DOSSIER-11"], "dominant_failure_phase": None}
        promotion_eval = {"promotion_ok": True, "status": "WARN", "blocking_error_codes": []}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["recurrent_errors"], ["HE-S1", "DOSSIER-11"])

    def test_critical_issues_for_critical_compass(self):
        """Critical issues populated for CRITICAL compass."""
        analytics = {}
        compass = {"compass_status": "CRITICAL", "recurrent_errors": [], "dominant_failure_phase": "phase_1_governance"}
        promotion_eval = {"promotion_ok": False, "status": "BLOCK", "blocking_error_codes": []}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertTrue(any("critical threshold" in issue.lower() for issue in signal["critical_issues"]))

    def test_critical_issues_for_blocking_errors(self):
        """Critical issues populated for blocking errors."""
        analytics = {}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": ["HE-GV1"], "dominant_failure_phase": None}
        promotion_eval = {"promotion_ok": False, "status": "BLOCK", "blocking_error_codes": ["HE-GV1"]}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertTrue(any("blocking errors" in issue.lower() for issue in signal["critical_issues"]))

    def test_critical_issues_for_dominant_failure_phase(self):
        """Critical issues include dominant failure phase when present."""
        analytics = {}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": [], "dominant_failure_phase": "phase_2_artifacts"}
        promotion_eval = {"promotion_ok": True, "status": "WARN", "blocking_error_codes": []}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertTrue(any("phase_2_artifacts" in issue for issue in signal["critical_issues"]))

    def test_signal_has_required_fields(self):
        """Signal contains all required fields."""
        analytics = {}
        compass = {"compass_status": "STABLE", "recurrent_errors": [], "dominant_failure_phase": None}
        promotion_eval = {"promotion_ok": True, "status": "OK", "blocking_error_codes": []}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        required_fields = [
            "signal_type", "recommendation", "confidence", "promotion_status",
            "compass_status", "long_run_admissibility_rate", "blocking_error_codes",
            "recurrent_errors", "critical_issues",
        ]
        for field in required_fields:
            self.assertIn(field, signal)

    def test_signal_is_json_serializable(self):
        """Signal is JSON serializable."""
        analytics = {"long_run_admissibility_rate": 0.8}
        compass = {"compass_status": "DEGRADING", "recurrent_errors": ["HE-S1"], "dominant_failure_phase": "phase_2_artifacts"}
        promotion_eval = {"promotion_ok": True, "status": "WARN", "blocking_error_codes": []}
        signal = to_governance_signal(analytics, compass, promotion_eval)
        json_str = json.dumps(signal)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, signal)


# =============================================================================
# PHASE V: BLOCK CASCADE INTEGRATION TESTS
# =============================================================================

class TestBlockCascadeIntegration(unittest.TestCase):
    """Integration tests verifying BLOCK cascades correctly through all adapters."""

    def test_critical_cascades_to_all_outputs(self):
        """CRITICAL compass cascades BLOCK to all outputs."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1"]}
        analytics = {"long_run_admissibility_rate": 0.3}
        compass = {
            "compass_status": "CRITICAL",
            "recurrent_errors": ["HE-GV1"],
            "dominant_failure_phase": "phase_1_governance",
            "failure_concentration": 0.9,
        }
        promotion_eval = {
            "promotion_ok": False,
            "status": "BLOCK",
            "blocking_error_codes": ["HE-GV1"],
        }

        # Global console should show not admissible
        console = summarize_admissibility_for_global_console(snapshot, compass, promotion_eval)
        self.assertFalse(console["admissible"])
        self.assertEqual(console["status_light"], "RED")

        # Evidence chain should show blocked
        chain = attach_admissibility_to_evidence_chain({}, compass, promotion_eval)
        self.assertTrue(chain["admissibility_compass"]["promotion_blocked"])
        self.assertEqual(chain["admissibility_compass"]["status"], "CRITICAL")

        # Governance signal should recommend HALT
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["recommendation"], "HALT")
        self.assertIn("HE-GV1", signal["blocking_error_codes"])

    def test_blocking_errors_cascade_to_all_outputs(self):
        """Blocking errors (HE-GV/HE-I) cascade BLOCK to all outputs."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-GV1", "HE-I2"]}
        analytics = {"long_run_admissibility_rate": 0.7}
        compass = {
            "compass_status": "DEGRADING",
            "recurrent_errors": ["HE-GV1", "HE-I2"],
            "dominant_failure_phase": "phase_1_governance",
            "failure_concentration": 0.6,
        }
        promotion_eval = {
            "promotion_ok": False,
            "status": "BLOCK",
            "blocking_error_codes": ["HE-GV1", "HE-I2"],
        }

        # Evidence chain should show blocked
        chain = attach_admissibility_to_evidence_chain({}, compass, promotion_eval)
        self.assertTrue(chain["admissibility_compass"]["promotion_blocked"])
        self.assertEqual(chain["admissibility_compass"]["blocking_error_codes"], ["HE-GV1", "HE-I2"])

        # Governance signal should recommend BLOCK
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["recommendation"], "BLOCK")
        self.assertEqual(signal["blocking_error_codes"], ["HE-GV1", "HE-I2"])

    def test_warn_status_does_not_block(self):
        """WARN status (DEGRADING without blockers) does not block."""
        snapshot = {"verdict": "NOT_ADMISSIBLE", "error_codes": ["HE-S1"]}
        analytics = {"long_run_admissibility_rate": 0.7}
        compass = {
            "compass_status": "DEGRADING",
            "recurrent_errors": ["HE-S1"],
            "dominant_failure_phase": "phase_2_artifacts",
            "failure_concentration": 0.5,
        }
        promotion_eval = {
            "promotion_ok": True,
            "status": "WARN",
            "blocking_error_codes": [],
        }

        # Evidence chain should show not blocked
        chain = attach_admissibility_to_evidence_chain({}, compass, promotion_eval)
        self.assertFalse(chain["admissibility_compass"]["promotion_blocked"])

        # Governance signal should recommend PROCEED_WITH_CAUTION
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["recommendation"], "PROCEED_WITH_CAUTION")

    def test_stable_ok_allows_all(self):
        """STABLE + OK allows promotion through all gates."""
        snapshot = {"verdict": "ADMISSIBLE", "error_codes": []}
        analytics = {"long_run_admissibility_rate": 0.95}
        compass = {
            "compass_status": "STABLE",
            "recurrent_errors": [],
            "dominant_failure_phase": None,
            "failure_concentration": 0.0,
        }
        promotion_eval = {
            "promotion_ok": True,
            "status": "OK",
            "blocking_error_codes": [],
        }

        # Global console should show admissible
        console = summarize_admissibility_for_global_console(snapshot, compass, promotion_eval)
        self.assertTrue(console["admissible"])
        self.assertEqual(console["status_light"], "GREEN")

        # Evidence chain should show not blocked
        chain = attach_admissibility_to_evidence_chain({}, compass, promotion_eval)
        self.assertFalse(chain["admissibility_compass"]["promotion_blocked"])
        self.assertEqual(chain["admissibility_compass"]["status"], "STABLE")

        # Governance signal should recommend PROCEED
        signal = to_governance_signal(analytics, compass, promotion_eval)
        self.assertEqual(signal["recommendation"], "PROCEED")
        self.assertEqual(signal["confidence"], "HIGH")


if __name__ == "__main__":
    unittest.main()
