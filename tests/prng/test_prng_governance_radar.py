"""
Tests for PRNG Drift Radar and Governance Tile.

Covers:
- build_prng_drift_radar: STABLE, DRIFTING, VOLATILE classifications based on frequent violations
- build_prng_governance_tile: Status derivation, blocking rules, determinism
- attach_prng_governance_tile: Evidence chain integration
"""

import json
import pytest
from rfl.prng.governance import (
    build_prng_drift_radar,
    build_prng_governance_tile,
    attach_prng_governance_tile,
    build_prng_governance_history,
    build_prng_drift_ledger,
    attach_prng_drift_ledger_to_evidence,
    PRNGGovernanceSnapshot,
    PolicyEvaluation,
    PolicyViolation,
    GovernanceStatus,
    ManifestStatus,
    DriftStatus,
    NamespaceIssues,
)


class TestDriftRadar:
    """Test build_prng_drift_radar."""

    def test_empty_history_returns_stable(self):
        """Empty history should return STABLE status."""
        history = {
            "schema_version": "1.0",
            "total_runs": 0,
            "runs": [],
            "status_counts": {"OK": 0, "WARN": 0, "BLOCK": 0},
            "history_hash": "",
        }

        radar = build_prng_drift_radar(history)

        assert radar["schema_version"] == "1.0.0"
        assert radar["drift_status"] == DriftStatus.STABLE.value
        assert radar["total_runs"] == 0
        assert radar["frequent_violations"] == {}

    def test_single_run_no_violations_stable(self):
        """Single run with no violations → STABLE."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.OK,
                manifest_status=ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
        ]
        policy_evals = [PolicyEvaluation(violations=[])]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        radar = build_prng_drift_radar(history)

        assert radar["drift_status"] == DriftStatus.STABLE.value
        assert len(radar["frequent_violations"]) == 0
        assert radar["total_runs"] == 1

    def test_multiple_runs_repeated_rule_frequent(self):
        """Rule appearing in >=3 runs → included in frequent_violations."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.BLOCK if i < 3 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.DRIFTED if i < 3 else ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(10)
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
                ] if i < 3 else []
            )
            for i in range(10)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        radar = build_prng_drift_radar(history)

        assert "R1" in radar["frequent_violations"]
        assert radar["frequent_violations"]["R1"] == 3
        assert radar["total_runs"] == 10

    def test_frequent_violations_below_threshold_not_included(self):
        """Rule appearing in <3 runs → not in frequent_violations."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.BLOCK if i < 2 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.DRIFTED if i < 2 else ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(10)
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
            for i in range(10)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        radar = build_prng_drift_radar(history)

        assert "R1" not in radar["frequent_violations"]
        assert len(radar["frequent_violations"]) == 0

    def test_drift_status_stable_no_frequent_violations(self):
        """No frequent violations → STABLE."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.OK,
                manifest_status=ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for _ in range(10)
        ]
        policy_evals = [PolicyEvaluation(violations=[]) for _ in range(10)]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        radar = build_prng_drift_radar(history)

        assert radar["drift_status"] == DriftStatus.STABLE.value
        assert len(radar["frequent_violations"]) == 0

    def test_drift_status_drifting_one_frequent_violation(self):
        """1 frequent violation → DRIFTING."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.BLOCK if i < 3 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.DRIFTED if i < 3 else ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(10)
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
                ] if i < 3 else []
            )
            for i in range(10)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        radar = build_prng_drift_radar(history)

        assert radar["drift_status"] == DriftStatus.DRIFTING.value
        assert len(radar["frequent_violations"]) == 1

    def test_drift_status_drifting_two_frequent_violations(self):
        """2 frequent violations → DRIFTING."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.BLOCK if i < 3 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.DRIFTED if i < 3 else ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(10)
        ]
        policy_evals = [
            PolicyEvaluation(
                violations=[
                    PolicyViolation(
                        rule_id="R1" if i < 3 else "R2",
                        rule_name="test",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    )
                ] if i < 6 else []
            )
            for i in range(10)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        radar = build_prng_drift_radar(history)

        assert radar["drift_status"] == DriftStatus.DRIFTING.value
        assert len(radar["frequent_violations"]) == 2

    def test_drift_status_volatile_three_frequent_violations(self):
        """≥3 frequent violations → VOLATILE."""
        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.BLOCK if i < 3 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.DRIFTED if i < 3 else ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(10)
        ]
        policy_evals = [
            PolicyEvaluation(
                violations=[
                    PolicyViolation(
                        rule_id=f"R{i % 3 + 1}",
                        rule_name="test",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    )
                ] if i < 9 else []
            )
            for i in range(10)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        radar = build_prng_drift_radar(history)

        assert radar["drift_status"] == DriftStatus.VOLATILE.value
        assert len(radar["frequent_violations"]) == 3


class TestGovernanceTile:
    """Test build_prng_governance_tile."""

    def test_tile_block_history_blocking_rules_populated(self):
        """BLOCK history → blocking_rules populated, status BLOCK."""
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
                    ),
                    PolicyViolation(
                        rule_id="R2",
                        rule_name="incompatible_block",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    ),
                ] if i < 2 else []
            )
            for i in range(5)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_governance_tile(history)

        assert tile["status"] == GovernanceStatus.BLOCK.value
        assert "R1" in tile["blocking_rules"]
        assert "R2" in tile["blocking_rules"]
        assert len(tile["blocking_rules"]) == 2

    def test_tile_only_warn_violations_status_warn(self):
        """Only WARN violations → status WARN."""
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
        assert tile["drift_status"] == DriftStatus.DRIFTING.value  # R4 appears 3 times
        assert tile["blocking_rules"] == []

    def test_tile_no_violations_status_ok(self):
        """No violations → status OK."""
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
        assert tile["blocking_rules"] == []

    def test_tile_determinism_same_history_same_outputs(self):
        """Same history → same outputs (deterministic)."""
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
                        rule_id="R3" if i == 0 else "R1",
                        rule_name="test",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    ),
                    PolicyViolation(
                        rule_id="R2",
                        rule_name="test",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    ),
                ] if i < 2 else []
            )
            for i in range(5)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile1 = build_prng_governance_tile(history)
        tile2 = build_prng_governance_tile(history)

        # JSON serialization should be identical
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)
        assert json1 == json2

        # Blocking rules should be sorted
        assert tile1["blocking_rules"] == sorted(tile1["blocking_rules"])
        assert tile1["blocking_rules"] == ["R1", "R2", "R3"]

    def test_tile_schema_fields(self):
        """Tile contains all required schema fields."""
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

        assert "schema_version" in tile
        assert tile["schema_version"] == "1.0.0"
        assert "status" in tile
        assert "drift_status" in tile
        assert "blocking_rules" in tile
        assert "headline" in tile

    def test_tile_json_serializable(self):
        """Tile is JSON-serializable."""
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

        # Should not raise
        json_str = json.dumps(tile)
        assert isinstance(json_str, str)
        # Should be able to round-trip
        tile_roundtrip = json.loads(json_str)
        assert tile_roundtrip["status"] == tile["status"]


class TestEvidenceChainIntegration:
    """Test attach_prng_governance_tile."""

    def test_attach_tile_read_only(self):
        """attach_prng_governance_tile is read-only (doesn't mutate input)."""
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
            "artifacts": [],
        }

        # Attach tile
        result = attach_prng_governance_tile(evidence, tile)

        # Original evidence should be unchanged
        assert "governance" not in evidence

        # Result should have tile attached
        assert "governance" in result
        assert "prng_governance" in result["governance"]
        assert result["governance"]["prng_governance"] == tile

    def test_attach_tile_additive(self):
        """attach_prng_governance_tile is additive (preserves existing data)."""
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
            "artifacts": ["log1.jsonl"],
            "metadata": {"key": "value"},
        }

        result = attach_prng_governance_tile(evidence, tile)

        # All original fields preserved
        assert result["version"] == "1.0.0"
        assert result["experiment_id"] == "test-123"
        assert result["artifacts"] == ["log1.jsonl"]
        assert result["metadata"] == {"key": "value"}

        # Tile attached
        assert "governance" in result
        assert "prng_governance" in result["governance"]
        assert result["governance"]["prng_governance"] == tile

    def test_attach_tile_json_serializable(self):
        """Result from attach_prng_governance_tile is JSON-serializable."""
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

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
        }

        result = attach_prng_governance_tile(evidence, tile)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        # Should be able to round-trip
        result_roundtrip = json.loads(json_str)
        assert "governance" in result_roundtrip
        assert "prng_governance" in result_roundtrip["governance"]
        assert result_roundtrip["governance"]["prng_governance"]["status"] == tile["status"]


class TestGlobalHealthIntegration:
    """Test PRNG tile integration with global health surface."""

    def test_prng_tile_for_global_health_shape(self):
        """PRNG tile for global health has correct shape."""
        from backend.health.prng_governance_adapter import (
            build_prng_tile_for_global_health,
            StatusLight,
        )

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
        tile = build_prng_tile_for_global_health(history)

        assert "schema_version" in tile
        assert "status_light" in tile
        assert "drift_status" in tile
        assert "status" in tile
        assert "frequent_violations" in tile
        assert "blocking_rules" in tile
        assert "headline" in tile

    def test_status_light_mapping_stable_green(self):
        """STABLE drift status maps to GREEN status_light."""
        from backend.health.prng_governance_adapter import (
            build_prng_tile_for_global_health,
            StatusLight,
        )

        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.OK,
                manifest_status=ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for _ in range(10)
        ]
        policy_evals = [PolicyEvaluation(violations=[]) for _ in range(10)]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_tile_for_global_health(history)

        assert tile["drift_status"] == DriftStatus.STABLE.value
        assert tile["status_light"] == StatusLight.GREEN

    def test_status_light_mapping_drifting_yellow(self):
        """DRIFTING drift status maps to YELLOW status_light."""
        from backend.health.prng_governance_adapter import (
            build_prng_tile_for_global_health,
            StatusLight,
        )

        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.BLOCK if i < 3 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.DRIFTED if i < 3 else ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(10)
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
                ] if i < 3 else []
            )
            for i in range(10)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_tile_for_global_health(history)

        assert tile["drift_status"] == DriftStatus.DRIFTING.value
        assert tile["status_light"] == StatusLight.YELLOW

    def test_status_light_mapping_volatile_red(self):
        """VOLATILE drift status maps to RED status_light."""
        from backend.health.prng_governance_adapter import (
            build_prng_tile_for_global_health,
            StatusLight,
        )

        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.BLOCK if i < 3 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.DRIFTED if i < 3 else ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(10)
        ]
        policy_evals = [
            PolicyEvaluation(
                violations=[
                    PolicyViolation(
                        rule_id=f"R{i % 3 + 1}",
                        rule_name="test",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    )
                ] if i < 9 else []
            )
            for i in range(10)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile = build_prng_tile_for_global_health(history)

        assert tile["drift_status"] == DriftStatus.VOLATILE.value
        assert tile["status_light"] == StatusLight.RED

    def test_global_health_tile_determinism(self):
        """Global health tile is deterministic (same history → same output)."""
        from backend.health.prng_governance_adapter import build_prng_tile_for_global_health

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
                        rule_name="test",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    )
                ] if i < 2 else []
            )
            for i in range(5)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        tile1 = build_prng_tile_for_global_health(history)
        tile2 = build_prng_tile_for_global_health(history)

        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)
        assert json1 == json2


class TestEvidenceIntegration:
    """Test PRNG evidence integration."""

    def test_summarize_prng_for_evidence_shape(self):
        """summarize_prng_for_evidence returns correct shape."""
        from rfl.prng.governance import summarize_prng_for_evidence

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
        radar = build_prng_drift_radar(history)
        tile = build_prng_governance_tile(history, radar=radar)

        summary = summarize_prng_for_evidence(radar, tile)

        assert "schema_version" in summary
        assert "drift_status" in summary
        assert "rule_frequencies" in summary
        assert "blocking_rules" in summary
        assert "forensic_narrative" in summary
        assert "total_runs" in summary

    def test_evidence_summary_attaches_to_evidence(self):
        """Evidence summary can be attached to evidence pack."""
        from rfl.prng.governance import summarize_prng_for_evidence

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
                        rule_name="test",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    )
                ] if i < 2 else []
            )
            for i in range(5)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        radar = build_prng_drift_radar(history)
        tile = build_prng_governance_tile(history, radar=radar)

        summary = summarize_prng_for_evidence(radar, tile)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
        }

        # Attach summary
        if "governance" not in evidence:
            evidence["governance"] = {}
        evidence["governance"]["prng"] = summary

        assert "governance" in evidence
        assert "prng" in evidence["governance"]
        assert evidence["governance"]["prng"]["drift_status"] == summary["drift_status"]


class TestCIMonitor:
    """Test CI drift monitor script behavior."""

    def test_monitor_handles_empty_directory(self, tmp_path):
        """Monitor handles empty input directory gracefully."""
        import subprocess
        import sys

        input_dir = tmp_path / "runs"
        input_dir.mkdir()

        output_file = tmp_path / "prng_drift.json"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/prng_drift_monitor.py",
                "--input-dir",
                str(input_dir),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        # Should exit 0 (shadow mode)
        assert result.returncode == 0

        # Should create output file
        assert output_file.exists()

        # Should contain empty radar
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["radar"]["total_runs"] == 0
            assert data["radar"]["drift_status"] == DriftStatus.STABLE.value

    def test_monitor_processes_run_summaries(self, tmp_path):
        """Monitor processes run summaries correctly."""
        import subprocess
        import sys

        input_dir = tmp_path / "runs"
        input_dir.mkdir()

        # Create a run summary
        run_summary = {
            "run_id": "run_0001",
            "governance_status": "OK",
            "manifest_status": "EQUIVALENT",
            "namespace_issues": {},
            "violations": [],
        }

        with open(input_dir / "run_0001.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f)

        output_file = tmp_path / "prng_drift.json"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/prng_drift_monitor.py",
                "--input-dir",
                str(input_dir),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        # Should exit 0
        assert result.returncode == 0

        # Should create output file
        assert output_file.exists()

        # Should contain radar and tile
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "radar" in data
            assert "tile" in data
            assert data["radar"]["total_runs"] == 1

    def test_monitor_always_exits_zero(self, tmp_path):
        """Monitor always exits 0 even with BLOCK conditions (shadow mode)."""
        import subprocess
        import sys

        input_dir = tmp_path / "runs"
        input_dir.mkdir()

        # Create a run summary with BLOCK status
        run_summary = {
            "run_id": "run_0001",
            "governance_status": "BLOCK",
            "manifest_status": "DRIFTED",
            "namespace_issues": {},
            "violations": [
                {
                    "rule_id": "R1",
                    "kind": "no_drifted_evidence",
                    "severity": "BLOCK",
                    "message": "test",
                }
            ],
        }

        with open(input_dir / "run_0001.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f)

        output_file = tmp_path / "prng_drift.json"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/prng_drift_monitor.py",
                "--input-dir",
                str(input_dir),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        # Should exit 0 even with BLOCK (shadow mode)
        assert result.returncode == 0

        # Should print WARN/BLOCK conditions
        assert "PRNG" in result.stdout or "BLOCK" in result.stdout


class TestFirstLightPRNGSummary:
    """
    Test First Light PRNG summary functionality.
    
    NOTE: The First Light PRNG summary is intended to be low-weight in any future
    fusion logic—more like a stability footnote than a primary gate. This summary
    provides long-horizon context about PRNG regime stability across multiple runs,
    not a go/no-go signal for a single First Light run. It is purely observational
    and does not gate any decisions.
    """

    def test_build_first_light_prng_summary_shape(self):
        """build_first_light_prng_summary returns correct shape."""
        from rfl.prng.governance import build_first_light_prng_summary

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
        radar = build_prng_drift_radar(history)
        tile = build_prng_governance_tile(history, radar=radar)

        summary = build_first_light_prng_summary(radar, tile)

        assert "schema_version" in summary
        assert summary["schema_version"] == "1.0.0"
        assert "drift_status" in summary
        assert "frequent_violations" in summary
        assert "status" in summary
        assert "blocking_rules" in summary
        assert "total_runs" in summary

    def test_first_light_summary_attaches_to_evidence(self):
        """First Light summary can be attached to evidence via attach_prng_governance_tile."""
        from rfl.prng.governance import attach_prng_governance_tile

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
                        rule_name="test",
                        severity=GovernanceStatus.BLOCK,
                        message="test",
                    )
                ] if i < 2 else []
            )
            for i in range(5)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        radar = build_prng_drift_radar(history)
        tile = build_prng_governance_tile(history, radar=radar)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
        }

        # Attach with first_light_summary
        result = attach_prng_governance_tile(
            evidence, tile, radar=radar, include_first_light_summary=True
        )

        assert "governance" in result
        assert "prng_governance" in result["governance"]
        assert "first_light_summary" in result["governance"]["prng_governance"]
        
        first_light = result["governance"]["prng_governance"]["first_light_summary"]
        assert first_light["drift_status"] == radar["drift_status"]
        assert first_light["status"] == tile["status"]
        assert first_light["total_runs"] == radar["total_runs"]

    def test_first_light_summary_determinism(self):
        """First Light summary is deterministic (same inputs → same output)."""
        from rfl.prng.governance import build_first_light_prng_summary

        snapshots = [
            PRNGGovernanceSnapshot(
                governance_status=GovernanceStatus.BLOCK if i < 3 else GovernanceStatus.OK,
                manifest_status=ManifestStatus.DRIFTED if i < 3 else ManifestStatus.EQUIVALENT,
                namespace_issues=NamespaceIssues(),
            )
            for i in range(10)
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
                ] if i < 3 else []
            )
            for i in range(10)
        ]

        history = build_prng_governance_history(snapshots, policy_evaluations=policy_evals)
        radar = build_prng_drift_radar(history)
        tile = build_prng_governance_tile(history, radar=radar)

        summary1 = build_first_light_prng_summary(radar, tile)
        summary2 = build_first_light_prng_summary(radar, tile)

        json1 = json.dumps(summary1, sort_keys=True)
        json2 = json.dumps(summary2, sort_keys=True)
        assert json1 == json2

    def test_first_light_summary_json_serializable(self):
        """First Light summary is JSON-serializable."""
        from rfl.prng.governance import build_first_light_prng_summary

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
        radar = build_prng_drift_radar(history)
        tile = build_prng_governance_tile(history, radar=radar)

        summary = build_first_light_prng_summary(radar, tile)

        # Should not raise
        json_str = json.dumps(summary)
        assert isinstance(json_str, str)
        # Should be able to round-trip
        summary_roundtrip = json.loads(json_str)
        assert summary_roundtrip["drift_status"] == summary["drift_status"]

    def test_attach_without_first_light_summary(self):
        """attach_prng_governance_tile works without first_light_summary."""
        from rfl.prng.governance import attach_prng_governance_tile

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

        # Attach without first_light_summary
        result = attach_prng_governance_tile(evidence, tile, include_first_light_summary=False)

        assert "governance" in result
        assert "prng_governance" in result["governance"]
        # Should not have first_light_summary when not requested
        assert "first_light_summary" not in result["governance"]["prng_governance"]

    def test_attach_requires_radar_for_first_light_summary(self):
        """attach_prng_governance_tile raises ValueError if radar missing when requested."""
        from rfl.prng.governance import attach_prng_governance_tile

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

        # Should raise ValueError when radar is None but include_first_light_summary=True
        with pytest.raises(ValueError, match="radar is required"):
            attach_prng_governance_tile(
                evidence, tile, radar=None, include_first_light_summary=True
            )


class TestPRNGRegimeComparison:
    """Test PRNG regime comparison functionality (mock vs real)."""

    def test_build_prng_regime_comparison_shape(self):
        """build_prng_regime_comparison returns correct shape."""
        from rfl.prng.governance import build_prng_regime_comparison

        mock_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 10,
            "volatile_runs": 2,
            "drifting_runs": 3,
            "stable_runs": 5,
            "frequent_rules": {"R1": 3, "R2": 2},
        }
        real_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 10,
            "volatile_runs": 1,
            "drifting_runs": 2,
            "stable_runs": 7,
            "frequent_rules": {"R1": 5, "R3": 2},
        }

        comparison = build_prng_regime_comparison(mock_ledger, real_ledger)

        assert "schema_version" in comparison
        assert comparison["schema_version"] == "1.0.0"
        assert "mock_drift_status" in comparison
        assert "real_drift_status" in comparison
        assert "delta_volatile_runs" in comparison
        assert "delta_stable_runs" in comparison
        assert "rules_more_frequent_in_real" in comparison
        assert "rules_more_frequent_in_mock" in comparison

    def test_regime_comparison_determinism(self):
        """Regime comparison is deterministic (same inputs → same output)."""
        from rfl.prng.governance import build_prng_regime_comparison

        mock_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 5,
            "volatile_runs": 1,
            "drifting_runs": 2,
            "stable_runs": 2,
            "frequent_rules": {"R1": 2, "R2": 1},
        }
        real_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 5,
            "volatile_runs": 0,
            "drifting_runs": 1,
            "stable_runs": 4,
            "frequent_rules": {"R1": 3, "R3": 1},
        }

        comparison1 = build_prng_regime_comparison(mock_ledger, real_ledger)
        comparison2 = build_prng_regime_comparison(mock_ledger, real_ledger)

        json1 = json.dumps(comparison1, sort_keys=True)
        json2 = json.dumps(comparison2, sort_keys=True)
        assert json1 == json2

    def test_regime_comparison_json_serializable(self):
        """Regime comparison is JSON-serializable."""
        from rfl.prng.governance import build_prng_regime_comparison

        mock_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 3,
            "volatile_runs": 0,
            "drifting_runs": 0,
            "stable_runs": 3,
            "frequent_rules": {},
        }
        real_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 3,
            "volatile_runs": 0,
            "drifting_runs": 0,
            "stable_runs": 3,
            "frequent_rules": {},
        }

        comparison = build_prng_regime_comparison(mock_ledger, real_ledger)

        # Should not raise
        json_str = json.dumps(comparison)
        assert isinstance(json_str, str)
        # Should be able to round-trip
        comparison_roundtrip = json.loads(json_str)
        assert comparison_roundtrip["mock_drift_status"] == comparison["mock_drift_status"]

    def test_regime_comparison_handles_empty_ledgers(self):
        """Regime comparison handles empty ledgers gracefully."""
        from rfl.prng.governance import build_prng_regime_comparison

        empty_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 0,
            "volatile_runs": 0,
            "drifting_runs": 0,
            "stable_runs": 0,
            "frequent_rules": {},
        }

        comparison = build_prng_regime_comparison(empty_ledger, empty_ledger)

        assert comparison["mock_drift_status"] == "STABLE"
        assert comparison["real_drift_status"] == "STABLE"
        assert comparison["delta_volatile_runs"] == 0
        assert comparison["delta_stable_runs"] == 0
        assert comparison["rules_more_frequent_in_real"] == []
        assert comparison["rules_more_frequent_in_mock"] == []

    def test_regime_comparison_rule_deltas(self):
        """Regime comparison correctly identifies rule frequency differences."""
        from rfl.prng.governance import build_prng_regime_comparison

        mock_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 10,
            "volatile_runs": 0,
            "drifting_runs": 0,
            "stable_runs": 10,
            "frequent_rules": {"R1": 2, "R2": 5},
        }
        real_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 10,
            "volatile_runs": 0,
            "drifting_runs": 0,
            "stable_runs": 10,
            "frequent_rules": {"R1": 6, "R3": 3},
        }

        comparison = build_prng_regime_comparison(mock_ledger, real_ledger)

        # R1 appears more in real (6 > 2)
        assert "R1" in comparison["rules_more_frequent_in_real"]
        # R2 appears only in mock (5 > 0)
        assert "R2" in comparison["rules_more_frequent_in_mock"]
        # R3 appears only in real (3 > 0)
        assert "R3" in comparison["rules_more_frequent_in_real"]
        # Rules are sorted deterministically
        assert comparison["rules_more_frequent_in_real"] == sorted(comparison["rules_more_frequent_in_real"])
        assert comparison["rules_more_frequent_in_mock"] == sorted(comparison["rules_more_frequent_in_mock"])

    def test_regime_comparison_attaches_to_evidence(self):
        """Regime comparison attaches to evidence via attach_prng_drift_ledger_to_evidence."""
        from rfl.prng.governance import (
            attach_prng_drift_ledger_to_evidence,
            build_prng_drift_ledger,
        )

        # Create mock tiles
        mock_tiles = [
            {"drift_status": "STABLE", "blocking_rules": []},
            {"drift_status": "DRIFTING", "blocking_rules": ["R1"]},
            {"drift_status": "STABLE", "blocking_rules": []},
        ]
        real_tiles = [
            {"drift_status": "STABLE", "blocking_rules": []},
            {"drift_status": "STABLE", "blocking_rules": []},
            {"drift_status": "VOLATILE", "blocking_rules": ["R1", "R2"]},
        ]

        mock_ledger = build_prng_drift_ledger(mock_tiles)
        real_ledger = build_prng_drift_ledger(real_tiles)
        combined_ledger = build_prng_drift_ledger(mock_tiles + real_tiles)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
        }

        # Attach with regime comparison
        result = attach_prng_drift_ledger_to_evidence(
            evidence, combined_ledger, mock_ledger=mock_ledger, real_ledger=real_ledger
        )

        assert "governance" in result
        assert "prng_drift_ledger" in result["governance"]
        assert "first_light_prng_regime_comparison" in result["governance"]["prng_drift_ledger"]
        
        comparison = result["governance"]["prng_drift_ledger"]["first_light_prng_regime_comparison"]
        assert comparison["mock_drift_status"] in ("STABLE", "DRIFTING", "VOLATILE")
        assert comparison["real_drift_status"] in ("STABLE", "DRIFTING", "VOLATILE")
        assert "delta_volatile_runs" in comparison

    def test_regime_comparison_optional_in_evidence(self):
        """Regime comparison is optional in evidence attachment."""
        from rfl.prng.governance import (
            attach_prng_drift_ledger_to_evidence,
            build_prng_drift_ledger,
        )

        tiles = [
            {"drift_status": "STABLE", "blocking_rules": []},
            {"drift_status": "DRIFTING", "blocking_rules": ["R1"]},
        ]

        ledger = build_prng_drift_ledger(tiles)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
        }

        # Attach without regime comparison
        result = attach_prng_drift_ledger_to_evidence(evidence, ledger)

        assert "governance" in result
        assert "prng_drift_ledger" in result["governance"]
        # Should not have regime comparison when not provided
        assert "first_light_prng_regime_comparison" not in result["governance"]["prng_drift_ledger"]


class TestPRNGRegimeTimeseries:
    """Test PRNG regime timeseries functionality."""

    def test_build_prng_regime_timeseries_shape(self):
        """build_prng_regime_timeseries returns correct shape."""
        from rfl.prng.governance import build_prng_regime_timeseries

        tiles_window0 = [
            {"drift_status": "STABLE", "blocking_rules": []},
            {"drift_status": "DRIFTING", "blocking_rules": ["R1"]},
            {"drift_status": "STABLE", "blocking_rules": []},
        ]
        tiles_window1 = [
            {"drift_status": "VOLATILE", "blocking_rules": ["R1", "R2"]},
            {"drift_status": "DRIFTING", "blocking_rules": ["R1"]},
        ]
        tiles_window2 = [
            {"drift_status": "STABLE", "blocking_rules": []},
        ]

        ts = build_prng_regime_timeseries(
            [tiles_window0, tiles_window1, tiles_window2], window_size=20
        )

        assert "schema_version" in ts
        assert ts["schema_version"] == "1.0.0"
        assert "window_size" in ts
        assert ts["window_size"] == 20
        assert "windows" in ts
        assert len(ts["windows"]) == 3

        # Check first window structure
        window0 = ts["windows"][0]
        assert "window_index" in window0
        assert window0["window_index"] == 0
        assert "drift_status" in window0
        assert "frequent_rules_top5" in window0
        assert "volatile_count" in window0
        assert "drifting_count" in window0
        assert "stable_count" in window0

    def test_timeseries_determinism(self):
        """Timeseries builder is deterministic (same inputs → same output)."""
        from rfl.prng.governance import build_prng_regime_timeseries

        tiles_window0 = [
            {"drift_status": "STABLE", "blocking_rules": ["R1"]},
            {"drift_status": "DRIFTING", "blocking_rules": ["R1", "R2"]},
        ]
        tiles_window1 = [
            {"drift_status": "VOLATILE", "blocking_rules": ["R1", "R2", "R3"]},
        ]

        ts1 = build_prng_regime_timeseries([tiles_window0, tiles_window1], window_size=20)
        ts2 = build_prng_regime_timeseries([tiles_window0, tiles_window1], window_size=20)

        json1 = json.dumps(ts1, sort_keys=True)
        json2 = json.dumps(ts2, sort_keys=True)
        assert json1 == json2

    def test_timeseries_json_serializable(self):
        """Timeseries is JSON-serializable."""
        from rfl.prng.governance import build_prng_regime_timeseries

        tiles_window0 = [
            {"drift_status": "STABLE", "blocking_rules": []},
        ]

        ts = build_prng_regime_timeseries([tiles_window0], window_size=20)

        # Should not raise
        json_str = json.dumps(ts)
        assert isinstance(json_str, str)
        # Should be able to round-trip
        ts_roundtrip = json.loads(json_str)
        assert ts_roundtrip["window_size"] == ts["window_size"]

    def test_timeseries_handles_empty_windows(self):
        """Timeseries handles empty windows gracefully."""
        from rfl.prng.governance import build_prng_regime_timeseries

        tiles_window0 = [
            {"drift_status": "STABLE", "blocking_rules": []},
        ]
        tiles_window1 = []  # Empty window

        ts = build_prng_regime_timeseries([tiles_window0, tiles_window1], window_size=20)

        assert len(ts["windows"]) == 2
        window1 = ts["windows"][1]
        assert window1["window_index"] == 1
        assert window1["drift_status"] == "STABLE"  # Default for empty window
        assert window1["frequent_rules_top5"] == []
        assert window1["volatile_count"] == 0
        assert window1["drifting_count"] == 0
        assert window1["stable_count"] == 0

    def test_timeseries_frequent_rules_top5(self):
        """Timeseries correctly identifies top 5 frequent rules per window."""
        from rfl.prng.governance import build_prng_regime_timeseries

        # Window with many rules, R1 appears most, R2 second, etc.
        tiles_window = [
            {"drift_status": "STABLE", "blocking_rules": ["R1", "R1", "R1", "R2", "R2", "R3"]},
            {"drift_status": "STABLE", "blocking_rules": ["R1", "R2", "R4"]},
            {"drift_status": "STABLE", "blocking_rules": ["R1", "R5"]},
        ]

        ts = build_prng_regime_timeseries([tiles_window], window_size=20)

        window = ts["windows"][0]
        frequent_rules = window["frequent_rules_top5"]
        
        # R1 should be first (appears 5 times)
        assert frequent_rules[0] == "R1"
        # R2 should be second (appears 3 times)
        assert frequent_rules[1] == "R2"
        # Should be sorted deterministically (by count descending, then by rule_id)
        assert len(frequent_rules) <= 5

    def test_timeseries_attaches_to_cal_exp_report_governance_path(self):
        """Timeseries attaches to CAL-EXP report at governance path (canonical location)."""
        from rfl.prng.governance import (
            attach_prng_regime_timeseries_to_cal_exp_report,
            build_prng_regime_timeseries,
        )

        tiles_window0 = [
            {"drift_status": "STABLE", "blocking_rules": []},
            {"drift_status": "DRIFTING", "blocking_rules": ["R1"]},
        ]
        tiles_window1 = [
            {"drift_status": "VOLATILE", "blocking_rules": ["R1", "R2"]},
        ]

        ts = build_prng_regime_timeseries([tiles_window0, tiles_window1], window_size=20)

        report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "summary": {},
        }

        # Attach timeseries
        result = attach_prng_regime_timeseries_to_cal_exp_report(report, ts)

        # Should be in governance path (canonical location)
        assert "governance" in result
        assert "prng_regime_timeseries" in result["governance"]
        assert result["governance"]["prng_regime_timeseries"]["window_size"] == 20
        assert len(result["governance"]["prng_regime_timeseries"]["windows"]) == 2

    def test_timeseries_attachment_preserves_legacy_path(self):
        """Attachment preserves legacy path if it exists in input (backward compatibility)."""
        from rfl.prng.governance import (
            attach_prng_regime_timeseries_to_cal_exp_report,
            build_prng_regime_timeseries,
        )

        tiles_window0 = [
            {"drift_status": "STABLE", "blocking_rules": []},
        ]

        ts = build_prng_regime_timeseries([tiles_window0], window_size=20)

        # Report with legacy path
        report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "prng_regime_timeseries": {"legacy": "data"},  # Legacy path
        }

        # Attach timeseries
        result = attach_prng_regime_timeseries_to_cal_exp_report(report, ts)

        # Should have governance path (canonical)
        assert "governance" in result
        assert "prng_regime_timeseries" in result["governance"]
        # Legacy path should still exist (preserved for backward compatibility)
        assert "prng_regime_timeseries" in result
        assert result["prng_regime_timeseries"]["legacy"] == "data"

    def test_timeseries_attachment_non_mutating(self):
        """attach_prng_regime_timeseries_to_cal_exp_report does not mutate input."""
        from rfl.prng.governance import (
            attach_prng_regime_timeseries_to_cal_exp_report,
            build_prng_regime_timeseries,
        )

        tiles_window0 = [
            {"drift_status": "STABLE", "blocking_rules": []},
        ]

        ts = build_prng_regime_timeseries([tiles_window0], window_size=20)

        report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
        }

        # Attach timeseries
        result = attach_prng_regime_timeseries_to_cal_exp_report(report, ts)

        # Original report should not have governance or timeseries
        assert "governance" not in report
        assert "prng_regime_timeseries" not in report
        # Result should have timeseries in governance path (canonical location)
        assert "governance" in result
        assert "prng_regime_timeseries" in result["governance"]
        # Result should be a different dict
        assert result is not report

    def test_timeseries_summary_fields_computed(self):
        """Timeseries summary fields (first_window_status, last_window_status, status_changed, volatile_window_count) are computed correctly."""
        from rfl.prng.governance import build_prng_regime_timeseries

        # Create windows with different drift statuses
        tiles_window0 = [
            {"drift_status": "STABLE", "blocking_rules": []},
        ]
        tiles_window1 = [
            {"drift_status": "DRIFTING", "blocking_rules": ["R1"]},
        ]
        tiles_window2 = [
            {"drift_status": "VOLATILE", "blocking_rules": ["R1", "R2"]},
        ]
        tiles_window3 = [
            {"drift_status": "VOLATILE", "blocking_rules": ["R2"]},
        ]

        ts = build_prng_regime_timeseries(
            [tiles_window0, tiles_window1, tiles_window2, tiles_window3], window_size=20
        )

        assert "first_window_status" in ts
        assert ts["first_window_status"] == "STABLE"
        assert "last_window_status" in ts
        assert ts["last_window_status"] == "VOLATILE"
        assert "status_changed" in ts
        assert ts["status_changed"] is True  # Changed from STABLE to VOLATILE
        assert "volatile_window_count" in ts
        assert ts["volatile_window_count"] == 2  # windows 2 and 3 are VOLATILE

    def test_timeseries_summary_fields_no_change(self):
        """Timeseries summary fields correctly identify when status does not change."""
        from rfl.prng.governance import build_prng_regime_timeseries

        tiles_window0 = [
            {"drift_status": "STABLE", "blocking_rules": []},
        ]
        tiles_window1 = [
            {"drift_status": "STABLE", "blocking_rules": []},
        ]

        ts = build_prng_regime_timeseries([tiles_window0, tiles_window1], window_size=20)

        assert ts["first_window_status"] == "STABLE"
        assert ts["last_window_status"] == "STABLE"
        assert ts["status_changed"] is False
        assert ts["volatile_window_count"] == 0

    def test_timeseries_summary_fields_empty_windows(self):
        """Timeseries summary fields handle empty windows gracefully."""
        from rfl.prng.governance import build_prng_regime_timeseries

        ts = build_prng_regime_timeseries([], window_size=20)

        assert ts["first_window_status"] == "STABLE"  # Default
        assert ts["last_window_status"] == "STABLE"  # Default
        assert ts["status_changed"] is False
        assert ts["volatile_window_count"] == 0

    def test_timeseries_determinism_with_summary_fields(self):
        """Timeseries with summary fields is deterministic (same inputs → same output)."""
        from rfl.prng.governance import build_prng_regime_timeseries

        tiles_window0 = [
            {"drift_status": "STABLE", "blocking_rules": ["R1"]},
            {"drift_status": "DRIFTING", "blocking_rules": ["R1", "R2"]},
        ]
        tiles_window1 = [
            {"drift_status": "VOLATILE", "blocking_rules": ["R1", "R2", "R3"]},
        ]

        ts1 = build_prng_regime_timeseries([tiles_window0, tiles_window1], window_size=20)
        ts2 = build_prng_regime_timeseries([tiles_window0, tiles_window1], window_size=20)

        json1 = json.dumps(ts1, sort_keys=True)
        json2 = json.dumps(ts2, sort_keys=True)
        assert json1 == json2
        
        # Verify summary fields are identical
        assert ts1["first_window_status"] == ts2["first_window_status"]
        assert ts1["last_window_status"] == ts2["last_window_status"]
        assert ts1["status_changed"] == ts2["status_changed"]
        assert ts1["volatile_window_count"] == ts2["volatile_window_count"]


class TestPRNGRegimeTimeseriesStatusExtraction:
    """Test PRNG regime timeseries status extraction in generate_first_light_status."""

    def test_status_extraction_governance_path_preferred(self, tmp_path):
        """Status extraction prefers governance path over legacy path."""
        import json
        from scripts.generate_first_light_status import generate_status

        # Create a CAL-EXP report with both governance and legacy paths
        cal_exp_report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "governance": {
                "prng_regime_timeseries": {
                    "schema_version": "1.0.0",
                    "window_size": 20,
                    "windows": [
                        {"window_index": 0, "drift_status": "STABLE"},
                        {"window_index": 1, "drift_status": "DRIFTING"},
                    ],
                    "first_window_status": "STABLE",
                    "last_window_status": "DRIFTING",
                    "status_changed": True,
                    "volatile_window_count": 0,
                }
            },
            "prng_regime_timeseries": {  # Legacy path (should be ignored)
                "schema_version": "1.0.0",
                "window_size": 20,
                "windows": [
                    {"window_index": 0, "drift_status": "VOLATILE"},  # Different data
                ],
            },
        }

        evidence_pack_dir = tmp_path / "evidence_pack"
        evidence_pack_dir.mkdir()
        cal_exp_report_path = evidence_pack_dir / "cal_exp2_report.json"
        with open(cal_exp_report_path, "w", encoding="utf-8") as f:
            json.dump(cal_exp_report, f)
        
        # Create minimal manifest.json
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        # Create minimal P3/P4 dirs with required structure
        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        # Create minimal required files
        (p3_run / "synthetic_raw.jsonl").touch()
        (p4_run / "p4_summary.json").write_text(json.dumps({"mode": "SHADOW"}))

        # Generate status
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

        # Should extract from governance path (STABLE -> DRIFTING)
        assert status is not None
        signals = status.get("signals")
        assert signals is not None
        assert "prng_regime" in signals
        assert "timeseries" in signals["prng_regime"]
        timeseries_signal = signals["prng_regime"]["timeseries"]
        assert timeseries_signal["first_window_drift_status"] == "STABLE"
        assert timeseries_signal["last_window_drift_status"] == "DRIFTING"
        assert timeseries_signal["status_changed"] is True

    def test_status_extraction_legacy_path_fallback(self, tmp_path):
        """Status extraction falls back to legacy path if governance path not present."""
        import json
        from scripts.generate_first_light_status import generate_status

        # Create a CAL-EXP report with only legacy path
        cal_exp_report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "prng_regime_timeseries": {  # Legacy path only
                "schema_version": "1.0.0",
                "window_size": 20,
                "windows": [
                    {"window_index": 0, "drift_status": "STABLE"},
                    {"window_index": 1, "drift_status": "VOLATILE"},
                ],
            },
        }

        evidence_pack_dir = tmp_path / "evidence_pack"
        evidence_pack_dir.mkdir()
        cal_exp_report_path = evidence_pack_dir / "cal_exp2_report.json"
        with open(cal_exp_report_path, "w", encoding="utf-8") as f:
            json.dump(cal_exp_report, f)
        
        # Create minimal manifest.json
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        # Create minimal P3/P4 dirs with required structure
        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        # Create minimal required files
        (p3_run / "synthetic_raw.jsonl").touch()
        (p4_run / "p4_summary.json").write_text(json.dumps({"mode": "SHADOW"}))

        # Generate status
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

        # Should extract from legacy path (fallback)
        assert status is not None
        signals = status.get("signals")
        assert signals is not None
        assert "prng_regime" in signals
        assert "timeseries" in signals["prng_regime"]
        timeseries_signal = signals["prng_regime"]["timeseries"]
        assert timeseries_signal["first_window_drift_status"] == "STABLE"
        assert timeseries_signal["last_window_drift_status"] == "VOLATILE"

    def test_status_extraction_no_cal_exp_report(self, tmp_path):
        """Status extraction does not throw when CAL-EXP report is absent."""
        import json
        from scripts.generate_first_light_status import generate_status

        # Create evidence pack dir without CAL-EXP report
        evidence_pack_dir = tmp_path / "evidence_pack"
        evidence_pack_dir.mkdir()
        
        # Create minimal manifest.json
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        # Create minimal P3/P4 dirs with required structure
        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        # Create minimal required files
        (p3_run / "synthetic_raw.jsonl").touch()
        (p4_run / "p4_summary.json").write_text(json.dumps({"mode": "SHADOW"}))

        # Generate status - should not throw
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

        # Should not have PRNG timeseries signal (missing data is expected, not an error)
        # Status should be generated successfully (no exception)
        assert status is not None
        # PRNG regime may or may not be present (depends on other signals)
        # But if present, timeseries should not be there
        signals = status.get("signals")
        if signals and "prng_regime" in signals:
            assert "timeseries" not in signals["prng_regime"]

    def test_status_extraction_includes_summary_fields(self, tmp_path):
        """Status extraction includes status_changed and volatile_window_count in signals."""
        import json
        from scripts.generate_first_light_status import generate_status

        # Create a CAL-EXP report with summary fields
        cal_exp_report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "governance": {
                "prng_regime_timeseries": {
                    "schema_version": "1.0.0",
                    "window_size": 20,
                    "windows": [
                        {"window_index": 0, "drift_status": "STABLE"},
                        {"window_index": 1, "drift_status": "VOLATILE"},
                        {"window_index": 2, "drift_status": "VOLATILE"},
                    ],
                    "first_window_status": "STABLE",
                    "last_window_status": "VOLATILE",
                    "status_changed": True,
                    "volatile_window_count": 2,
                }
            },
        }

        evidence_pack_dir = tmp_path / "evidence_pack"
        evidence_pack_dir.mkdir()
        cal_exp_report_path = evidence_pack_dir / "cal_exp2_report.json"
        with open(cal_exp_report_path, "w", encoding="utf-8") as f:
            json.dump(cal_exp_report, f)
        
        # Create minimal manifest.json
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        # Create minimal P3/P4 dirs with required structure
        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        # Create minimal required files
        (p3_run / "synthetic_raw.jsonl").touch()
        (p4_run / "p4_summary.json").write_text(json.dumps({"mode": "SHADOW"}))

        # Generate status
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

        # Should include summary fields
        assert status is not None
        signals = status.get("signals")
        assert signals is not None
        assert "prng_regime" in signals
        assert "timeseries" in signals["prng_regime"]
        timeseries_signal = signals["prng_regime"]["timeseries"]
        assert "status_changed" in timeseries_signal
        assert timeseries_signal["status_changed"] is True
        assert "volatile_window_count" in timeseries_signal
        assert timeseries_signal["volatile_window_count"] == 2

    def test_status_extraction_source_governance_path(self, tmp_path):
        """Status extraction correctly identifies GOVERNANCE_PATH as extraction_source."""
        import json
        from scripts.generate_first_light_status import generate_status

        cal_exp_report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "governance": {
                "prng_regime_timeseries": {
                    "schema_version": "1.0.0",
                    "window_size": 20,
                    "windows": [
                        {"window_index": 0, "drift_status": "STABLE"},
                        {"window_index": 1, "drift_status": "DRIFTING"},
                    ],
                    "first_window_status": "STABLE",
                    "last_window_status": "DRIFTING",
                    "status_changed": True,
                    "volatile_window_count": 0,
                }
            },
        }

        evidence_pack_dir = tmp_path / "evidence_pack"
        evidence_pack_dir.mkdir()
        cal_exp_report_path = evidence_pack_dir / "cal_exp2_report.json"
        with open(cal_exp_report_path, "w", encoding="utf-8") as f:
            json.dump(cal_exp_report, f)
        
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        (p3_run / "synthetic_raw.jsonl").touch()
        (p4_run / "p4_summary.json").write_text(json.dumps({"mode": "SHADOW"}))

        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

        signals = status.get("signals")
        assert signals is not None
        assert "prng_regime" in signals
        assert "timeseries" in signals["prng_regime"]
        timeseries_signal = signals["prng_regime"]["timeseries"]
        assert timeseries_signal["extraction_source"] == "GOVERNANCE_PATH"
        assert timeseries_signal["schema_version"] == "1.0.0"

    def test_status_extraction_source_legacy_path(self, tmp_path):
        """Status extraction correctly identifies LEGACY_PATH as extraction_source."""
        import json
        from scripts.generate_first_light_status import generate_status

        cal_exp_report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "prng_regime_timeseries": {  # Legacy path only
                "schema_version": "1.0.0",
                "window_size": 20,
                "windows": [
                    {"window_index": 0, "drift_status": "STABLE"},
                    {"window_index": 1, "drift_status": "VOLATILE"},
                ],
                "first_window_status": "STABLE",
                "last_window_status": "VOLATILE",
                "status_changed": True,
                "volatile_window_count": 1,
            },
        }

        evidence_pack_dir = tmp_path / "evidence_pack"
        evidence_pack_dir.mkdir()
        cal_exp_report_path = evidence_pack_dir / "cal_exp2_report.json"
        with open(cal_exp_report_path, "w", encoding="utf-8") as f:
            json.dump(cal_exp_report, f)
        
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        (p3_run / "synthetic_raw.jsonl").touch()
        (p4_run / "p4_summary.json").write_text(json.dumps({"mode": "SHADOW"}))

        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

        signals = status.get("signals")
        assert signals is not None
        assert "prng_regime" in signals
        assert "timeseries" in signals["prng_regime"]
        timeseries_signal = signals["prng_regime"]["timeseries"]
        assert timeseries_signal["extraction_source"] == "LEGACY_PATH"
        assert timeseries_signal["schema_version"] == "1.0.0"

    def test_status_extraction_source_missing(self, tmp_path):
        """Status extraction correctly identifies MISSING when no timeseries found."""
        import json
        from scripts.generate_first_light_status import generate_status

        # CAL-EXP report without PRNG timeseries
        cal_exp_report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
        }

        evidence_pack_dir = tmp_path / "evidence_pack"
        evidence_pack_dir.mkdir()
        cal_exp_report_path = evidence_pack_dir / "cal_exp2_report.json"
        with open(cal_exp_report_path, "w", encoding="utf-8") as f:
            json.dump(cal_exp_report, f)
        
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        (p3_run / "synthetic_raw.jsonl").touch()
        (p4_run / "p4_summary.json").write_text(json.dumps({"mode": "SHADOW"}))

        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

        # Should not have PRNG timeseries signal (missing data is expected, not an error)
        signals = status.get("signals")
        if signals and "prng_regime" in signals:
            # If PRNG regime exists, timeseries should not be there
            assert "timeseries" not in signals["prng_regime"]

    def test_status_extraction_determinism_legacy_path_only(self, tmp_path):
        """
        Determinism test: schema_version + extraction_source are present even when only legacy path exists.
        
        PROVENANCE LAW v1: Ensures provenance fields are always present and deterministic.
        """
        import json
        from scripts.generate_first_light_status import generate_status

        # CAL-EXP report with legacy path only (no governance path)
        cal_exp_report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "prng_regime_timeseries": {  # Legacy path only
                "schema_version": "1.0.0",
                "window_size": 20,
                "windows": [
                    {"window_index": 0, "drift_status": "STABLE"},
                    {"window_index": 1, "drift_status": "DRIFTING"},
                ],
                "first_window_status": "STABLE",
                "last_window_status": "DRIFTING",
                "status_changed": True,
                "volatile_window_count": 0,
            },
        }

        evidence_pack_dir = tmp_path / "evidence_pack"
        evidence_pack_dir.mkdir()
        cal_exp_report_path = evidence_pack_dir / "cal_exp2_report.json"
        with open(cal_exp_report_path, "w", encoding="utf-8") as f:
            json.dump(cal_exp_report, f)
        
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        (p3_run / "synthetic_raw.jsonl").touch()
        (p4_run / "p4_summary.json").write_text(json.dumps({"mode": "SHADOW"}))

        # Run twice to verify determinism
        status1 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)

        signals1 = status1.get("signals")
        signals2 = status2.get("signals")
        
        assert signals1 is not None
        assert signals2 is not None
        assert "prng_regime" in signals1
        assert "prng_regime" in signals2
        assert "timeseries" in signals1["prng_regime"]
        assert "timeseries" in signals2["prng_regime"]
        
        timeseries1 = signals1["prng_regime"]["timeseries"]
        timeseries2 = signals2["prng_regime"]["timeseries"]
        
        # Verify provenance fields are present
        assert "extraction_source" in timeseries1
        assert "extraction_source" in timeseries2
        assert "schema_version" in timeseries1
        assert "schema_version" in timeseries2
        assert "source_paths_checked" in timeseries1
        assert "source_paths_checked" in timeseries2
        
        # Verify extraction_source is LEGACY_PATH (coerced to canonical enum)
        assert timeseries1["extraction_source"] == "LEGACY_PATH"
        assert timeseries2["extraction_source"] == "LEGACY_PATH"
        
        # Verify schema_version is present
        assert timeseries1["schema_version"] == "1.0.0"
        assert timeseries2["schema_version"] == "1.0.0"
        
        # Verify determinism: identical outputs
        assert timeseries1 == timeseries2
        
        # Verify source_paths_checked is deterministic (sorted)
        assert timeseries1["source_paths_checked"] == sorted(timeseries1["source_paths_checked"])
        assert timeseries2["source_paths_checked"] == sorted(timeseries2["source_paths_checked"])


class TestPRNGRegimeTimeseriesManifestMirroring:
    """Test PRNG regime timeseries manifest mirroring in build_first_light_evidence_pack."""

    def test_manifest_mirroring_governance_path(self, tmp_path):
        """Manifest mirroring works with governance path (canonical location)."""
        import json
        from scripts.build_first_light_evidence_pack import build_evidence_pack

        # Create minimal P3/P4 dirs
        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        # Create minimal required files for build_evidence_pack
        (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
        (p3_run / "stability_report.json").write_text(
            json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
        )
        (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
        (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
        (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
        (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
        
        (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
        (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
        (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
        (p4_run / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW"}), encoding="utf-8"
        )
        (p4_run / "twin_accuracy.json").write_text("{}", encoding="utf-8")
        (p4_run / "run_config.json").write_text("{}", encoding="utf-8")

        out_dir = tmp_path / "evidence_pack"
        out_dir.mkdir()

        # Create CAL-EXP report with governance path
        cal_exp_report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "governance": {
                "prng_regime_timeseries": {
                    "schema_version": "1.0.0",
                    "window_size": 20,
                    "windows": [
                        {"window_index": 0, "drift_status": "STABLE"},
                    ],
                    "first_window_status": "STABLE",
                    "last_window_status": "STABLE",
                    "status_changed": False,
                    "volatile_window_count": 0,
                }
            },
        }
        cal_exp_report_path = out_dir / "cal_exp2_report.json"
        with open(cal_exp_report_path, "w", encoding="utf-8") as f:
            json.dump(cal_exp_report, f)

        # Build evidence pack
        build_evidence_pack(p3_dir, p4_dir, out_dir)

        # Check manifest
        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Should have PRNG timeseries in governance path
        assert "governance" in manifest
        assert "prng_regime_timeseries" in manifest["governance"]
        ts_mirrored = manifest["governance"]["prng_regime_timeseries"]
        assert ts_mirrored["schema_version"] == "1.0.0"
        # Should NOT have mirrored_from field (governance path is canonical)
        assert "mirrored_from" not in ts_mirrored
        # Should have source_paths_checked integrity invariant (deterministic ordering)
        assert "source_paths_checked" in ts_mirrored
        assert ts_mirrored["source_paths_checked"] == sorted(ts_mirrored["source_paths_checked"])

    def test_manifest_mirroring_legacy_path(self, tmp_path):
        """Manifest mirroring works with legacy path fallback and marks mirrored_from."""
        import json
        from scripts.build_first_light_evidence_pack import build_evidence_pack

        # Create minimal P3/P4 dirs
        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        # Create minimal required files for build_evidence_pack
        (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
        (p3_run / "stability_report.json").write_text(
            json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
        )
        (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
        (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
        (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
        (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
        
        (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
        (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
        (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
        (p4_run / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW"}), encoding="utf-8"
        )
        (p4_run / "twin_accuracy.json").write_text("{}", encoding="utf-8")
        (p4_run / "run_config.json").write_text("{}", encoding="utf-8")

        out_dir = tmp_path / "evidence_pack"
        out_dir.mkdir()

        # Create CAL-EXP report with legacy path only
        cal_exp_report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "prng_regime_timeseries": {  # Legacy path
                "schema_version": "1.0.0",
                "window_size": 20,
                "windows": [
                    {"window_index": 0, "drift_status": "DRIFTING"},
                ],
                "first_window_status": "DRIFTING",
                "last_window_status": "DRIFTING",
                "status_changed": False,
                "volatile_window_count": 0,
            },
        }
        cal_exp_report_path = out_dir / "cal_exp2_report.json"
        with open(cal_exp_report_path, "w", encoding="utf-8") as f:
            json.dump(cal_exp_report, f)

        # Build evidence pack
        build_evidence_pack(p3_dir, p4_dir, out_dir)

        # Check manifest
        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Should have PRNG timeseries in governance path (mirrored)
        assert "governance" in manifest
        assert "prng_regime_timeseries" in manifest["governance"]
        assert manifest["governance"]["prng_regime_timeseries"]["schema_version"] == "1.0.0"
        # Should have mirrored_from field (legacy path was used)
        assert manifest["governance"]["prng_regime_timeseries"]["mirrored_from"] == "LEGACY_PATH"

    def test_manifest_mirroring_deterministic_ordering(self, tmp_path):
        """Manifest mirroring preserves deterministic ordering of timeseries fields."""
        import json
        from scripts.build_first_light_evidence_pack import build_evidence_pack

        # Create minimal P3/P4 dirs
        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        
        # Create minimal required files for build_evidence_pack
        (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
        (p3_run / "stability_report.json").write_text(
            json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
        )
        (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
        (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
        (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
        (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
        
        (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
        (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
        (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
        (p4_run / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW"}), encoding="utf-8"
        )
        (p4_run / "twin_accuracy.json").write_text("{}", encoding="utf-8")
        (p4_run / "run_config.json").write_text("{}", encoding="utf-8")

        out_dir = tmp_path / "evidence_pack"
        out_dir.mkdir()

        # Create CAL-EXP report with governance path
        cal_exp_report = {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-2",
            "governance": {
                "prng_regime_timeseries": {
                    "schema_version": "1.0.0",
                    "window_size": 20,
                    "windows": [
                        {"window_index": 0, "drift_status": "STABLE"},
                        {"window_index": 1, "drift_status": "DRIFTING"},
                    ],
                    "first_window_status": "STABLE",
                    "last_window_status": "DRIFTING",
                    "status_changed": True,
                    "volatile_window_count": 0,
                }
            },
        }
        cal_exp_report_path = out_dir / "cal_exp2_report.json"
        with open(cal_exp_report_path, "w", encoding="utf-8") as f:
            json.dump(cal_exp_report, f)

        # Build evidence pack twice
        build_evidence_pack(p3_dir, p4_dir, out_dir)
        with open(manifest_path := out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest1 = json.load(f)

        # Rebuild (should produce same ordering)
        build_evidence_pack(p3_dir, p4_dir, out_dir)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest2 = json.load(f)

        # Timeseries should be identical (deterministic)
        ts1 = manifest1["governance"]["prng_regime_timeseries"]
        ts2 = manifest2["governance"]["prng_regime_timeseries"]
        json1 = json.dumps(ts1, sort_keys=True)
        json2 = json.dumps(ts2, sort_keys=True)
        assert json1 == json2


class TestDriftLedger:
    """Test build_prng_drift_ledger."""

    def test_empty_tiles_returns_zero_counts(self):
        """Empty tiles list → all counts zero."""
        ledger = build_prng_drift_ledger([])

        assert ledger["schema_version"] == "1.0.0"
        assert ledger["total_runs"] == 0
        assert ledger["volatile_runs"] == 0
        assert ledger["drifting_runs"] == 0
        assert ledger["stable_runs"] == 0
        assert ledger["frequent_rules"] == {}

    def test_ledger_classifies_runs_by_drift_status(self):
        """Ledger correctly classifies runs by drift_status."""
        tiles = [
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
            {"drift_status": DriftStatus.DRIFTING.value, "blocking_rules": ["R1"]},
            {"drift_status": DriftStatus.DRIFTING.value, "blocking_rules": ["R2"]},
            {"drift_status": DriftStatus.VOLATILE.value, "blocking_rules": ["R1", "R3"]},
        ]

        ledger = build_prng_drift_ledger(tiles)

        assert ledger["total_runs"] == 5
        assert ledger["stable_runs"] == 2
        assert ledger["drifting_runs"] == 2
        assert ledger["volatile_runs"] == 1

    def test_ledger_aggregates_frequent_rules(self):
        """Ledger aggregates blocking_rules across all tiles."""
        tiles = [
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
            {"drift_status": DriftStatus.DRIFTING.value, "blocking_rules": ["R1"]},
            {"drift_status": DriftStatus.DRIFTING.value, "blocking_rules": ["R1", "R2"]},
            {"drift_status": DriftStatus.VOLATILE.value, "blocking_rules": ["R1", "R2", "R3"]},
            {"drift_status": DriftStatus.VOLATILE.value, "blocking_rules": ["R2"]},
        ]

        ledger = build_prng_drift_ledger(tiles)

        # R1 appears 3 times, R2 appears 3 times, R3 appears 1 time
        assert ledger["frequent_rules"]["R1"] == 3
        assert ledger["frequent_rules"]["R2"] == 3
        assert ledger["frequent_rules"]["R3"] == 1
        assert len(ledger["frequent_rules"]) == 3

    def test_ledger_frequent_rules_sorted(self):
        """frequent_rules are sorted for determinism."""
        tiles = [
            {"drift_status": DriftStatus.DRIFTING.value, "blocking_rules": ["R3", "R1", "R2"]},
        ]

        ledger = build_prng_drift_ledger(tiles)

        # Should be sorted by rule_id
        rule_ids = list(ledger["frequent_rules"].keys())
        assert rule_ids == sorted(rule_ids)

    def test_ledger_all_stable_runs(self):
        """All STABLE runs → correct counts."""
        tiles = [
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
        ]

        ledger = build_prng_drift_ledger(tiles)

        assert ledger["total_runs"] == 3
        assert ledger["stable_runs"] == 3
        assert ledger["drifting_runs"] == 0
        assert ledger["volatile_runs"] == 0
        assert ledger["frequent_rules"] == {}


class TestDriftLedgerEvidenceIntegration:
    """Test attach_prng_drift_ledger_to_evidence."""

    def test_attach_ledger_preserves_evidence_structure(self):
        """attach_prng_drift_ledger_to_evidence preserves existing evidence sections."""
        tiles = [
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
            {"drift_status": DriftStatus.DRIFTING.value, "blocking_rules": ["R1"]},
        ]
        ledger = build_prng_drift_ledger(tiles)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
            "artifacts": ["log1.jsonl", "log2.jsonl"],
            "metadata": {"key": "value"},
            "other_section": {"data": "preserved"},
        }

        enriched = attach_prng_drift_ledger_to_evidence(evidence, ledger)

        # Original sections preserved
        assert enriched["version"] == "1.0.0"
        assert enriched["experiment_id"] == "test-123"
        assert enriched["artifacts"] == ["log1.jsonl", "log2.jsonl"]
        assert enriched["metadata"] == {"key": "value"}
        assert enriched["other_section"] == {"data": "preserved"}

        # Ledger attached under governance.prng_drift_ledger
        assert "governance" in enriched
        assert "prng_drift_ledger" in enriched["governance"]
        assert enriched["governance"]["prng_drift_ledger"] == ledger

    def test_attach_ledger_read_only(self):
        """attach_prng_drift_ledger_to_evidence does not mutate input evidence."""
        tiles = [
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
        ]
        ledger = build_prng_drift_ledger(tiles)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
        }

        enriched = attach_prng_drift_ledger_to_evidence(evidence, ledger)

        # Original evidence unchanged
        assert "governance" not in evidence

        # Enriched has ledger
        assert "governance" in enriched
        assert "prng_drift_ledger" in enriched["governance"]

    def test_attach_ledger_json_serializable(self):
        """Result from attach_prng_drift_ledger_to_evidence is JSON-serializable."""
        tiles = [
            {"drift_status": DriftStatus.VOLATILE.value, "blocking_rules": ["R1", "R2"]},
            {"drift_status": DriftStatus.DRIFTING.value, "blocking_rules": ["R1"]},
        ]
        ledger = build_prng_drift_ledger(tiles)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
        }

        result = attach_prng_drift_ledger_to_evidence(evidence, ledger)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        # Should be able to round-trip
        result_roundtrip = json.loads(json_str)
        assert result_roundtrip["governance"]["prng_drift_ledger"]["total_runs"] == 2
        assert result_roundtrip["governance"]["prng_drift_ledger"]["volatile_runs"] == 1
