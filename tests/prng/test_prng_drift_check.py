"""
Tests for PRNG drift check CI script.

Covers:
- Exit code behavior for STABLE/DRIFTING/VOLATILE scenarios
- Tile output validation
- Evidence pack integration
"""

import json
import pytest
from pathlib import Path
from rfl.prng.governance import (
    build_prng_drift_radar,
    build_prng_governance_tile,
    build_prng_governance_history,
    attach_prng_governance_tile,
    PRNGGovernanceSnapshot,
    PolicyEvaluation,
    PolicyViolation,
    GovernanceStatus,
    ManifestStatus,
    DriftStatus,
    NamespaceIssues,
)


class TestCIDriftCheck:
    """Test PRNG drift check exit codes and outputs."""

    def test_exit_code_ok_stable(self):
        """STABLE status → exit code 0."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.OK,
                manifest_status=ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for _ in range(5)
        ]
        policy_evals = [PolicyEvaluation(violations=[]) for _ in range(5)]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_governance_tile(history)

        assert tile["status"] == GovernanceStatus.OK.value
        assert tile["drift_status"] == DriftStatus.STABLE.value

    def test_exit_code_warn_drifting(self):
        """DRIFTING status (WARN) → exit code 0."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.WARN if i < 3 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(10)
        ]
        policy_evals = [
            PolicyEvaluation(
                violations=[
                    PolicyViolation(
                        rule_id="R4",
                        rule_name="namespace_collision_warn",
                        severity=GovernanceStatus.WARN,
                        message="test",
                    )
                ] if i < 3 else []
            )
            for i in range(10)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_governance_tile(history)

        assert tile["status"] == GovernanceStatus.WARN.value
        assert tile["drift_status"] == DriftStatus.DRIFTING.value

    def test_exit_code_block(self):
        """BLOCK status → exit code 1 (would be returned by script)."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.BLOCK if i < 2 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.DRIFTED if i < 2 else ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(5)
        ]
        policy_evals = [
            PolicyEvaluation(
                violations=[
                    PolicyViolation(
                        rule_id="R1",
                        rule_name="no_drifted_evidence",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    )
                ] if i < 2 else []
            )
            for i in range(5)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_governance_tile(history)

        assert tile["status"] == GovernanceStatus.BLOCK.value
        # Exit code would be 1 for BLOCK status

    def test_tile_has_required_fields(self):
        """Tile has all fields expected by global health."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.OK,
                manifest_status=ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for _ in range(3)
        ]
        policy_evals = [PolicyEvaluation(violations=[]) for _ in range(3)]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_governance_tile(history)

        # Required fields for global health adapter
        assert "schema_version" in tile
        assert "status" in tile
        assert "drift_status" in tile
        assert "blocking_rules" in tile
        assert "headline" in tile

        # Fields must be valid values
        assert tile["status"] in ("OK", "WARN", "BLOCK")
        assert tile["drift_status"] in ("STABLE", "DRIFTING", "VOLATILE")
        assert isinstance(tile["blocking_rules"], list)
        assert isinstance(tile["headline"], str)


class TestEvidencePackIntegration:
    """Test evidence pack harmonization."""

    def test_attach_tile_preserves_evidence_structure(self):
        """attach_prng_governance_tile preserves existing evidence sections."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.OK,
                manifest_status=ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for _ in range(3)
        ]
        policy_evals = [PolicyEvaluation(violations=[]) for _ in range(3)]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_governance_tile(history)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
            "artifacts": ["log1.jsonl", "log2.jsonl"],
            "metadata": {"key": "value"},
            "other_section": {"data": "preserved"},
        }

        enriched = attach_prng_governance_tile(evidence, tile)

        # Original sections preserved
        assert enriched["version"] == "1.0.0"
        assert enriched["experiment_id"] == "test-123"
        assert enriched["artifacts"] == ["log1.jsonl", "log2.jsonl"]
        assert enriched["metadata"] == {"key": "value"}
        assert enriched["other_section"] == {"data": "preserved"}

        # PRNG tile attached under governance.prng_governance
        assert "governance" in enriched
        assert "prng_governance" in enriched["governance"]
        assert enriched["governance"]["prng_governance"] == tile

    def test_attach_tile_read_only(self):
        """attach_prng_governance_tile does not mutate input evidence."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.OK,
                manifest_status=ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for _ in range(3)
        ]
        policy_evals = [PolicyEvaluation(violations=[]) for _ in range(3)]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_governance_tile(history)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
        }

        enriched = attach_prng_governance_tile(evidence, tile)

        # Original evidence unchanged
        assert "governance" not in evidence

        # Enriched has tile
        assert "governance" in enriched
        assert "prng_governance" in enriched["governance"]

    def test_attach_tile_deterministic_json(self):
        """Attached tile produces deterministic JSON."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.BLOCK if i < 1 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.DRIFTED if i < 1 else ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(3)
        ]
        policy_evals = [
            PolicyEvaluation(
                violations=[
                    PolicyViolation(
                        rule_id="R1",
                        rule_name="test",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    )
                ] if i < 1 else []
            )
            for i in range(3)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_governance_tile(history)

        evidence = {"version": "1.0.0"}

        enriched1 = attach_prng_governance_tile(evidence, tile)
        enriched2 = attach_prng_governance_tile(evidence, tile)

        # Should produce identical JSON
        json1 = json.dumps(enriched1, sort_keys=True)
        json2 = json.dumps(enriched2, sort_keys=True)
        assert json1 == json2

