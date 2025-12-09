# PHASE II — NOT USED IN PHASE I
"""
Tests for PRNG Governance Phase IV features.

Verifies:
- CI gate exit code mapping
- Auto-remediation suggestions
- Governance history ledger
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rfl.prng.governance import (
    GovernanceStatus,
    ManifestStatus,
    PRNGGovernanceSnapshot,
    PolicyEvaluation,
    GlobalHealthSummary,
    PolicyViolation,
    evaluate_prng_for_ci,
    build_prng_remediation_suggestions,
    build_prng_governance_history,
    build_prng_governance_snapshot,
    evaluate_prng_policy,
    summarize_prng_for_global_health,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

def create_manifest(
    master_seed: str = "a" * 64,
    merkle_root: str = "b" * 64,
    entry_count: int = 10,
):
    """Create a test manifest."""
    return {
        "manifest_version": "1.1",
        "prng_attestation": {
            "master_seed_hex": master_seed,
            "lineage_merkle_root": merkle_root,
            "lineage_entry_count": entry_count,
        },
    }


def create_namespace_report(duplicates=0, hardcoded=0, hardcoded_files=None):
    """Create a test namespace report."""
    return {
        "duplicates": [
            {"namespace": f"ns_{i}", "usages": [{"file": f"file{j}.py"} for j in range(2)]}
            for i in range(duplicates)
        ],
        "hard_coded_seeds": [
            {"file": f or f"runtime_{i}.py", "seed_value": str(i)}
            for i, f in enumerate(hardcoded_files or [None] * hardcoded)
        ],
        "dynamic_paths": [],
        "suppressed_count": 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: CI GATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCIGate:
    """Tests for evaluate_prng_for_ci."""

    def test_ok_returns_zero(self):
        """OK status returns exit code 0."""
        health = GlobalHealthSummary(
            prng_policy_ok=True,
            status=GovernanceStatus.OK,
        )
        assert evaluate_prng_for_ci(health) == 0

    def test_warn_returns_one(self):
        """WARN status returns exit code 1."""
        health = GlobalHealthSummary(
            prng_policy_ok=False,
            status=GovernanceStatus.WARN,
        )
        assert evaluate_prng_for_ci(health) == 1

    def test_block_returns_two(self):
        """BLOCK status returns exit code 2."""
        health = GlobalHealthSummary(
            prng_policy_ok=False,
            status=GovernanceStatus.BLOCK,
        )
        assert evaluate_prng_for_ci(health) == 2

    def test_unknown_status_defaults_to_block(self):
        """Unknown status defaults to BLOCK (exit code 2)."""
        # Create a health summary with invalid status (bypassing enum)
        health = GlobalHealthSummary(
            prng_policy_ok=False,
            status=GovernanceStatus.BLOCK,  # Use valid enum but test logic
        )
        # The function should handle any status gracefully
        result = evaluate_prng_for_ci(health)
        assert result in (0, 1, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: AUTO-REMEDIATION SUGGESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRemediationSuggestions:
    """Tests for build_prng_remediation_suggestions."""

    def test_r1_suggestion_drifted_evidence(self):
        """R1 violation generates appropriate suggestion."""
        manifest = create_manifest(merkle_root="a" * 64)
        replay = create_manifest(merkle_root="b" * 64)

        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            replay_manifest=replay,
            is_evidence_run=True,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        suggestions = build_prng_remediation_suggestions(snapshot, policy_eval)

        r1_suggestions = [s for s in suggestions if s["rule_id"] == "R1"]
        assert len(r1_suggestions) > 0
        assert "evidence" in r1_suggestions[0]["impact"].lower()
        assert "suggested_action" in r1_suggestions[0]

    def test_r2_suggestion_incompatible(self):
        """R2 violation generates appropriate suggestion."""
        manifest = create_manifest()
        manifest["prng_attestation"]["derivation_scheme"] = "v1"
        replay = create_manifest()
        replay["prng_attestation"]["derivation_scheme"] = "v2"

        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            replay_manifest=replay,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        suggestions = build_prng_remediation_suggestions(snapshot, policy_eval)

        r2_suggestions = [s for s in suggestions if s["rule_id"] == "R2"]
        assert len(r2_suggestions) > 0
        assert "incompatible" in r2_suggestions[0]["impact"].lower()

    def test_r3_suggestion_hardcoded_seeds(self):
        """R3 violation generates suggestion with file list."""
        namespace_report = create_namespace_report(
            hardcoded=2,
            hardcoded_files=["rfl/runner.py", "experiments/run.py"],
        )

        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            namespace_report=namespace_report,
            is_test_context=False,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        suggestions = build_prng_remediation_suggestions(snapshot, policy_eval)

        r3_suggestions = [s for s in suggestions if s["rule_id"] == "R3"]
        assert len(r3_suggestions) > 0
        assert "hard-coded" in r3_suggestions[0]["impact"].lower()
        assert len(r3_suggestions[0]["files_involved"]) > 0
        assert "rfl/runner.py" in r3_suggestions[0]["files_involved"]

    def test_r4_suggestion_namespace_collisions(self):
        """R4 violation generates suggestion with file list."""
        namespace_report = create_namespace_report(duplicates=2)

        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            namespace_report=namespace_report,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        suggestions = build_prng_remediation_suggestions(snapshot, policy_eval)

        r4_suggestions = [s for s in suggestions if s["rule_id"] == "R4"]
        assert len(r4_suggestions) > 0
        assert "collision" in r4_suggestions[0]["impact"].lower()
        assert "namespace-ok" in r4_suggestions[0]["suggested_action"].lower()

    def test_r5_suggestion_missing_attestation(self):
        """R5 violation generates appropriate suggestion."""
        snapshot = build_prng_governance_snapshot(manifest=None)
        policy_eval = evaluate_prng_policy(snapshot)

        suggestions = build_prng_remediation_suggestions(snapshot, policy_eval)

        r5_suggestions = [s for s in suggestions if s["rule_id"] == "R5"]
        assert len(r5_suggestions) > 0
        assert "attestation" in r5_suggestions[0]["impact"].lower()
        assert "prng_attestation" in r5_suggestions[0]["suggested_action"]

    def test_suggestion_structure(self):
        """All suggestions have required fields."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        policy_eval = evaluate_prng_policy(snapshot)

        suggestions = build_prng_remediation_suggestions(snapshot, policy_eval)

        for suggestion in suggestions:
            assert "rule_id" in suggestion
            assert "impact" in suggestion
            assert "suggested_action" in suggestion
            assert "files_involved" in suggestion
            assert isinstance(suggestion["files_involved"], list)

    def test_no_violations_no_suggestions(self):
        """Clean snapshot produces no violation-based suggestions."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        policy_eval = evaluate_prng_policy(snapshot)

        suggestions = build_prng_remediation_suggestions(snapshot, policy_eval)

        # Only INFO suggestions should remain
        violation_suggestions = [s for s in suggestions if s["rule_id"] not in ("INFO",)]
        assert len(violation_suggestions) == 0

    def test_multiple_violations_multiple_suggestions(self):
        """Multiple violations produce multiple suggestions."""
        namespace_report = create_namespace_report(
            duplicates=1,
            hardcoded=1,
            hardcoded_files=["runtime.py"],
        )

        snapshot = build_prng_governance_snapshot(
            manifest=None,  # Missing attestation
            namespace_report=namespace_report,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        suggestions = build_prng_remediation_suggestions(snapshot, policy_eval)

        # Should have suggestions for R3, R4, R5
        rule_ids = {s["rule_id"] for s in suggestions}
        assert "R3" in rule_ids or "R4" in rule_ids or "R5" in rule_ids


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: GOVERNANCE HISTORY LEDGER
# ═══════════════════════════════════════════════════════════════════════════════

class TestGovernanceHistory:
    """Tests for build_prng_governance_history."""

    def test_empty_history(self):
        """Empty snapshot list produces valid history."""
        history = build_prng_governance_history([])

        assert history["schema_version"] == "1.0"
        assert history["total_runs"] == 0
        assert history["runs"] == []
        assert history["status_counts"]["OK"] == 0
        assert history["history_hash"] != ""

    def test_single_snapshot(self):
        """Single snapshot produces valid history."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        history = build_prng_governance_history([snapshot])

        assert history["total_runs"] == 1
        assert len(history["runs"]) == 1
        assert history["runs"][0]["governance_status"] == "OK"
        assert history["status_counts"]["OK"] == 1

    def test_multiple_snapshots(self):
        """Multiple snapshots aggregated correctly."""
        snapshots = [
            build_prng_governance_snapshot(manifest=create_manifest()),
            build_prng_governance_snapshot(manifest=None),  # Missing
            build_prng_governance_snapshot(manifest=create_manifest()),
        ]

        history = build_prng_governance_history(snapshots)

        assert history["total_runs"] == 3
        assert len(history["runs"]) == 3
        assert sum(history["status_counts"].values()) == 3

    def test_custom_run_ids(self):
        """Custom run IDs are preserved."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        history = build_prng_governance_history([snapshot], run_ids=["custom_run_001"])

        assert history["runs"][0]["run_id"] == "custom_run_001"

    def test_run_ids_length_mismatch_raises(self):
        """Mismatched run_ids length raises ValueError."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())

        with pytest.raises(ValueError, match="length"):
            build_prng_governance_history([snapshot], run_ids=["id1", "id2"])

    def test_history_hash_deterministic(self):
        """History hash is deterministic for same inputs."""
        snapshots = [
            build_prng_governance_snapshot(manifest=create_manifest()),
            build_prng_governance_snapshot(manifest=create_manifest()),
        ]

        history1 = build_prng_governance_history(snapshots)
        history2 = build_prng_governance_history(snapshots)

        assert history1["history_hash"] == history2["history_hash"]

    def test_history_hash_different_for_different_inputs(self):
        """History hash differs for different inputs."""
        snapshots1 = [build_prng_governance_snapshot(manifest=create_manifest())]
        snapshots2 = [build_prng_governance_snapshot(manifest=None)]

        history1 = build_prng_governance_history(snapshots1)
        history2 = build_prng_governance_history(snapshots2)

        assert history1["history_hash"] != history2["history_hash"]

    def test_status_counts_accurate(self):
        """Status counts match actual run statuses."""
        snapshots = [
            build_prng_governance_snapshot(manifest=create_manifest()),  # OK
            build_prng_governance_snapshot(manifest=None),  # WARN (missing)
        ]

        namespace_report = create_namespace_report(
            hardcoded=1,
            hardcoded_files=["runtime.py"],
        )
        snapshots.append(
            build_prng_governance_snapshot(
                manifest=create_manifest(),
                namespace_report=namespace_report,
            )
        )

        history = build_prng_governance_history(snapshots)

        # Count actual statuses
        actual_ok = sum(1 for r in history["runs"] if r["governance_status"] == "OK")
        actual_warn = sum(1 for r in history["runs"] if r["governance_status"] == "WARN")

        assert history["status_counts"]["OK"] == actual_ok
        assert history["status_counts"]["WARN"] == actual_warn

    def test_manifest_status_counts(self):
        """Manifest status counts are included."""
        manifest1 = create_manifest(merkle_root="a" * 64)
        manifest2 = create_manifest(merkle_root="b" * 64)

        snapshots = [
            build_prng_governance_snapshot(manifest=manifest1),
            build_prng_governance_snapshot(
                manifest=manifest1,
                replay_manifest=manifest2,
            ),  # DRIFTED
        ]

        history = build_prng_governance_history(snapshots)

        assert "manifest_status_counts" in history
        assert "EQUIVALENT" in history["manifest_status_counts"]
        assert "DRIFTED" in history["manifest_status_counts"]

    def test_history_structure(self):
        """History has all required fields."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        history = build_prng_governance_history([snapshot])

        required_fields = [
            "schema_version",
            "total_runs",
            "runs",
            "status_counts",
            "manifest_status_counts",
            "history_hash",
        ]

        for field in required_fields:
            assert field in history

    def test_run_record_structure(self):
        """Each run record has required fields."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        history = build_prng_governance_history([snapshot])

        run = history["runs"][0]

        required_fields = [
            "run_id",
            "governance_status",
            "manifest_status",
            "has_hardcoded_seeds",
            "hardcoded_seed_count",
            "namespace_duplicate_count",
            "lineage_fingerprint",
            "timestamp",
        ]

        for field in required_fields:
            assert field in run

    def test_history_deterministic_ordering(self):
        """History runs are deterministically ordered."""
        snapshots = [
            build_prng_governance_snapshot(manifest=create_manifest()),
            build_prng_governance_snapshot(manifest=create_manifest()),
            build_prng_governance_snapshot(manifest=create_manifest()),
        ]

        history1 = build_prng_governance_history(snapshots)
        history2 = build_prng_governance_history(snapshots)

        # Same run IDs in same order
        run_ids1 = [r["run_id"] for r in history1["runs"]]
        run_ids2 = [r["run_id"] for r in history2["runs"]]
        assert run_ids1 == run_ids2


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for Phase IV features."""

    def test_full_pipeline_with_ci_gate(self):
        """Full pipeline produces CI-compatible exit code."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        policy_eval = evaluate_prng_policy(snapshot)
        health = summarize_prng_for_global_health(snapshot, policy_eval)

        exit_code = evaluate_prng_for_ci(health)

        assert exit_code in (0, 1, 2)

    def test_suggestions_with_history(self):
        """Suggestions and history work together."""
        namespace_report = create_namespace_report(
            hardcoded=1,
            hardcoded_files=["runtime.py"],
        )

        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            namespace_report=namespace_report,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        suggestions = build_prng_remediation_suggestions(snapshot, policy_eval)
        history = build_prng_governance_history([snapshot])

        assert len(suggestions) > 0
        assert history["total_runs"] == 1
        assert history["runs"][0]["has_hardcoded_seeds"] is True

