# PHASE II — NOT USED IN PHASE I
"""
Tests for PRNG Governance Layer.

Verifies:
- Governance snapshot building
- Policy rule evaluation
- Global health summarization
- Full pipeline execution
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rfl.prng.governance import (
    GovernanceStatus,
    ManifestStatus,
    PRNG_GOV_RULES,
    PolicyRule,
    NamespaceIssues,
    PolicyViolation,
    PRNGGovernanceSnapshot,
    PolicyEvaluation,
    GlobalHealthSummary,
    build_prng_governance_snapshot,
    evaluate_prng_policy,
    summarize_prng_for_global_health,
    run_full_prng_governance,
    _is_test_file,
    _compute_lineage_fingerprint,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

def create_manifest(
    master_seed: str = "a" * 64,
    merkle_root: str = "b" * 64,
    entry_count: int = 10,
    derivation_scheme: str = "PRNGKey(root, path) -> SHA256 -> seed % 2^32",
):
    """Create a test manifest with PRNG attestation."""
    return {
        "manifest_version": "1.1",
        "experiment_id": "test_exp",
        "prng_attestation": {
            "schema_version": "1.0",
            "master_seed_hex": master_seed,
            "derivation_scheme": derivation_scheme,
            "lineage_merkle_root": merkle_root,
            "lineage_entry_count": entry_count,
        },
    }


def create_namespace_report(
    duplicates: int = 0,
    duplicate_files: list = None,
    hardcoded: int = 0,
    hardcoded_files: list = None,
    dynamic: int = 0,
    suppressed: int = 0,
):
    """Create a test namespace linter report."""
    dup_list = []
    for i in range(duplicates):
        files = duplicate_files or [f"file{j}.py" for j in range(2)]
        dup_list.append({
            "namespace": f"namespace_{i}",
            "usages": [{"file": f} for f in files],
        })

    hc_list = []
    for i in range(hardcoded):
        files = hardcoded_files or [f"runtime_{i}.py"]
        for f in files:
            hc_list.append({"file": f, "seed_value": str(i)})

    return {
        "duplicates": dup_list,
        "hard_coded_seeds": hc_list,
        "dynamic_paths": [{"file": f"dyn_{i}.py"} for i in range(dynamic)],
        "suppressed_count": suppressed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: GOVERNANCE SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════

class TestGovernanceSnapshot:
    """Tests for build_prng_governance_snapshot."""

    def test_snapshot_from_manifest(self):
        """Builds snapshot from manifest."""
        manifest = create_manifest()
        snapshot = build_prng_governance_snapshot(manifest=manifest)

        assert snapshot.manifest_status == ManifestStatus.EQUIVALENT
        assert snapshot.master_seed_hex == "a" * 64
        assert snapshot.lineage_entry_count == 10
        assert snapshot.seed_lineage_fingerprint != ""
        assert snapshot.governance_status == GovernanceStatus.OK

    def test_snapshot_with_namespace_issues(self):
        """Captures namespace issues in snapshot."""
        manifest = create_manifest()
        namespace_report = create_namespace_report(duplicates=2, hardcoded=3)

        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            namespace_report=namespace_report,
        )

        assert snapshot.namespace_issues.duplicate_count == 2
        assert snapshot.namespace_issues.hardcoded_seed_count == 3
        assert snapshot.hardcoded_seeds_detected is True
        assert snapshot.hardcoded_seed_count == 3

    def test_snapshot_missing_manifest(self):
        """Handles missing manifest."""
        snapshot = build_prng_governance_snapshot(manifest=None)

        assert snapshot.manifest_status == ManifestStatus.MISSING
        assert snapshot.master_seed_hex is None

    def test_snapshot_drifted_detection(self):
        """Detects DRIFTED status when same seed, different merkle."""
        manifest = create_manifest(merkle_root="a" * 64)
        replay = create_manifest(merkle_root="b" * 64)

        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            replay_manifest=replay,
        )

        assert snapshot.manifest_status == ManifestStatus.DRIFTED

    def test_snapshot_incompatible_detection(self):
        """Detects INCOMPATIBLE status when different schemes."""
        manifest = create_manifest(derivation_scheme="scheme_v1")
        replay = create_manifest(derivation_scheme="scheme_v2")

        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            replay_manifest=replay,
        )

        assert snapshot.manifest_status == ManifestStatus.INCOMPATIBLE

    def test_snapshot_evidence_run_flag(self):
        """Preserves evidence run flag."""
        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            is_evidence_run=True,
        )

        assert snapshot.is_evidence_run is True

    def test_snapshot_test_context_flag(self):
        """Preserves test context flag."""
        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            is_test_context=True,
        )

        assert snapshot.is_test_context is True

    def test_snapshot_to_dict(self):
        """Snapshot serializes to dict."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        d = snapshot.to_dict()

        assert "schema_version" in d
        assert "manifest_status" in d
        assert "namespace_issues" in d
        assert "governance_status" in d


class TestLineageFingerprint:
    """Tests for lineage fingerprint computation."""

    def test_fingerprint_deterministic(self):
        """Same inputs produce same fingerprint."""
        fp1 = _compute_lineage_fingerprint("a" * 64, "b" * 64, 10)
        fp2 = _compute_lineage_fingerprint("a" * 64, "b" * 64, 10)

        assert fp1 == fp2

    def test_fingerprint_different_seed(self):
        """Different seed produces different fingerprint."""
        fp1 = _compute_lineage_fingerprint("a" * 64, "b" * 64, 10)
        fp2 = _compute_lineage_fingerprint("c" * 64, "b" * 64, 10)

        assert fp1 != fp2

    def test_fingerprint_different_merkle(self):
        """Different merkle root produces different fingerprint."""
        fp1 = _compute_lineage_fingerprint("a" * 64, "b" * 64, 10)
        fp2 = _compute_lineage_fingerprint("a" * 64, "c" * 64, 10)

        assert fp1 != fp2

    def test_fingerprint_no_seed(self):
        """Handles missing seed gracefully."""
        fp = _compute_lineage_fingerprint(None, None, 0)
        assert fp == "no-seed"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: POLICY RULES
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolicyRules:
    """Tests for PRNG_GOV_RULES definitions."""

    def test_all_rules_defined(self):
        """All expected rules are defined."""
        expected = ["R1", "R2", "R3", "R4", "R5"]
        for rule_id in expected:
            assert rule_id in PRNG_GOV_RULES

    def test_rules_have_required_fields(self):
        """Each rule has required fields."""
        for rule_id, rule in PRNG_GOV_RULES.items():
            assert isinstance(rule, PolicyRule)
            assert rule.rule_id == rule_id
            assert rule.name
            assert rule.description
            assert isinstance(rule.severity, GovernanceStatus)
            assert rule.applies_to in ("evidence", "runtime", "all")


class TestPolicyEvaluation:
    """Tests for evaluate_prng_policy."""

    def test_clean_snapshot_passes(self):
        """Clean snapshot passes all rules."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        result = evaluate_prng_policy(snapshot)

        assert result.policy_ok is True
        assert result.status == GovernanceStatus.OK
        assert len(result.violations) == 0

    def test_r1_drifted_evidence_blocks(self):
        """R1: DRIFTED + evidence run = BLOCK."""
        manifest = create_manifest(merkle_root="a" * 64)
        replay = create_manifest(merkle_root="b" * 64)

        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            replay_manifest=replay,
            is_evidence_run=True,
        )

        result = evaluate_prng_policy(snapshot)

        assert result.policy_ok is False
        assert result.status == GovernanceStatus.BLOCK
        assert any(v.rule_id == "R1" for v in result.violations)

    def test_r1_drifted_non_evidence_ok(self):
        """R1: DRIFTED without evidence run doesn't trigger R1."""
        manifest = create_manifest(merkle_root="a" * 64)
        replay = create_manifest(merkle_root="b" * 64)

        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            replay_manifest=replay,
            is_evidence_run=False,
        )

        result = evaluate_prng_policy(snapshot, rules_to_check=["R1"])

        r1_violations = [v for v in result.violations if v.rule_id == "R1"]
        assert len(r1_violations) == 0

    def test_r2_incompatible_blocks(self):
        """R2: INCOMPATIBLE = BLOCK."""
        manifest = create_manifest(derivation_scheme="v1")
        replay = create_manifest(derivation_scheme="v2")

        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            replay_manifest=replay,
        )

        result = evaluate_prng_policy(snapshot)

        assert result.policy_ok is False
        assert result.status == GovernanceStatus.BLOCK
        assert any(v.rule_id == "R2" for v in result.violations)

    def test_r3_hardcoded_runtime_blocks(self):
        """R3: Hard-coded seeds in runtime = BLOCK."""
        namespace_report = create_namespace_report(
            hardcoded=1,
            hardcoded_files=["rfl/runner.py"],  # Non-test file
        )

        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            namespace_report=namespace_report,
            is_test_context=False,
        )

        result = evaluate_prng_policy(snapshot)

        assert any(v.rule_id == "R3" for v in result.violations)

    def test_r3_hardcoded_test_allowed(self):
        """R3: Hard-coded seeds in test context allowed."""
        namespace_report = create_namespace_report(
            hardcoded=1,
            hardcoded_files=["tests/test_example.py"],
        )

        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            namespace_report=namespace_report,
            is_test_context=True,
        )

        result = evaluate_prng_policy(snapshot, rules_to_check=["R3"])

        r3_violations = [v for v in result.violations if v.rule_id == "R3"]
        assert len(r3_violations) == 0

    def test_r4_namespace_collision_warns(self):
        """R4: Namespace collisions = WARN."""
        namespace_report = create_namespace_report(duplicates=2)

        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            namespace_report=namespace_report,
        )

        result = evaluate_prng_policy(snapshot)

        r4_violations = [v for v in result.violations if v.rule_id == "R4"]
        assert len(r4_violations) == 1
        assert r4_violations[0].severity == GovernanceStatus.WARN

    def test_r5_missing_attestation_warns(self):
        """R5: Missing attestation = WARN."""
        snapshot = build_prng_governance_snapshot(manifest=None)

        result = evaluate_prng_policy(snapshot)

        r5_violations = [v for v in result.violations if v.rule_id == "R5"]
        assert len(r5_violations) == 1

    def test_specific_rules_only(self):
        """Can evaluate specific rules only."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        result = evaluate_prng_policy(snapshot, rules_to_check=["R1", "R2"])

        assert "R1" in result.rules_checked
        assert "R2" in result.rules_checked
        assert "R3" not in result.rules_checked

    def test_evaluation_to_dict(self):
        """Policy evaluation serializes to dict."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        result = evaluate_prng_policy(snapshot)
        d = result.to_dict()

        assert "policy_ok" in d
        assert "violations" in d
        assert "status" in d
        assert "rules_checked" in d


class TestIsTestFile:
    """Tests for _is_test_file helper."""

    def test_test_prefix(self):
        """Detects test_ prefix."""
        assert _is_test_file("tests/test_example.py") is True
        assert _is_test_file("/path/to/test_foo.py") is True

    def test_tests_directory(self):
        """Detects tests/ directory."""
        assert _is_test_file("project/tests/something.py") is True

    def test_test_suffix(self):
        """Detects _test.py suffix."""
        assert _is_test_file("module_test.py") is True

    def test_conftest(self):
        """Detects conftest.py."""
        assert _is_test_file("tests/conftest.py") is True

    def test_runtime_file(self):
        """Identifies runtime files as non-test."""
        assert _is_test_file("rfl/runner.py") is False
        assert _is_test_file("experiments/run.py") is False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: GLOBAL HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlobalHealth:
    """Tests for summarize_prng_for_global_health."""

    def test_healthy_summary(self):
        """Clean state produces healthy summary."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        policy_eval = evaluate_prng_policy(snapshot)

        health = summarize_prng_for_global_health(snapshot, policy_eval)

        assert health.prng_policy_ok is True
        assert health.has_namespace_collisions is False
        assert health.has_schedule_drift is False
        assert health.has_hardcoded_seeds is False
        assert health.status == GovernanceStatus.OK

    def test_drift_detected(self):
        """Schedule drift flagged in health."""
        manifest = create_manifest(merkle_root="a" * 64)
        replay = create_manifest(merkle_root="b" * 64)

        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            replay_manifest=replay,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        health = summarize_prng_for_global_health(snapshot, policy_eval)

        assert health.has_schedule_drift is True

    def test_namespace_collisions_detected(self):
        """Namespace collisions flagged in health."""
        namespace_report = create_namespace_report(duplicates=3)

        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            namespace_report=namespace_report,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        health = summarize_prng_for_global_health(snapshot, policy_eval)

        assert health.has_namespace_collisions is True

    def test_hardcoded_seeds_detected(self):
        """Hard-coded seeds flagged in health."""
        namespace_report = create_namespace_report(hardcoded=2)

        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            namespace_report=namespace_report,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        health = summarize_prng_for_global_health(snapshot, policy_eval)

        assert health.has_hardcoded_seeds is True

    def test_violation_count(self):
        """Counts violations in health."""
        namespace_report = create_namespace_report(
            duplicates=1,
            hardcoded=1,
            hardcoded_files=["runtime.py"],
        )

        snapshot = build_prng_governance_snapshot(
            manifest=create_manifest(),
            namespace_report=namespace_report,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        health = summarize_prng_for_global_health(snapshot, policy_eval)

        assert health.violation_count >= 1

    def test_summary_message_ok(self):
        """OK summary has appropriate message."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        policy_eval = evaluate_prng_policy(snapshot)

        health = summarize_prng_for_global_health(snapshot, policy_eval)

        assert "OK" in health.summary_message

    def test_summary_message_with_issues(self):
        """Summary with issues describes them."""
        manifest = create_manifest(merkle_root="a" * 64)
        replay = create_manifest(merkle_root="b" * 64)

        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            replay_manifest=replay,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        health = summarize_prng_for_global_health(snapshot, policy_eval)

        # Should mention drift somewhere in the message
        assert "drift" in health.summary_message.lower() or health.has_schedule_drift

    def test_health_to_dict(self):
        """Health summary serializes to dict."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())
        policy_eval = evaluate_prng_policy(snapshot)
        health = summarize_prng_for_global_health(snapshot, policy_eval)

        d = health.to_dict()

        assert "prng_policy_ok" in d
        assert "has_namespace_collisions" in d
        assert "has_schedule_drift" in d
        assert "status" in d


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Tests for run_full_prng_governance."""

    def test_full_pipeline_healthy(self):
        """Full pipeline with healthy inputs."""
        result = run_full_prng_governance(
            manifest=create_manifest(),
        )

        assert "snapshot" in result
        assert "policy_eval" in result
        assert "health" in result

        assert result["health"]["prng_policy_ok"] is True
        assert result["health"]["status"] == "OK"

    def test_full_pipeline_with_issues(self):
        """Full pipeline detects issues."""
        namespace_report = create_namespace_report(
            duplicates=1,
            hardcoded=1,
            hardcoded_files=["runtime.py"],
        )

        result = run_full_prng_governance(
            manifest=create_manifest(),
            namespace_report=namespace_report,
        )

        assert result["health"]["prng_policy_ok"] is False
        assert result["policy_eval"]["violations"]

    def test_full_pipeline_evidence_run(self):
        """Full pipeline with evidence run flag."""
        manifest = create_manifest(merkle_root="a" * 64)
        replay = create_manifest(merkle_root="b" * 64)

        result = run_full_prng_governance(
            manifest=manifest,
            replay_manifest=replay,
            is_evidence_run=True,
        )

        assert result["health"]["status"] == "BLOCK"

    def test_full_pipeline_test_context(self):
        """Full pipeline in test context."""
        namespace_report = create_namespace_report(
            hardcoded=1,
            hardcoded_files=["tests/test_example.py"],
        )

        result = run_full_prng_governance(
            manifest=create_manifest(),
            namespace_report=namespace_report,
            is_test_context=True,
        )

        # Hard-coded seeds in tests should be allowed
        r3_violations = [
            v for v in result["policy_eval"]["violations"]
            if v["rule_id"] == "R3"
        ]
        assert len(r3_violations) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_snapshot_deterministic(self):
        """Snapshot is deterministic for same inputs."""
        manifest = create_manifest()
        namespace_report = create_namespace_report(duplicates=1)

        snap1 = build_prng_governance_snapshot(
            manifest=manifest,
            namespace_report=namespace_report,
        )
        snap2 = build_prng_governance_snapshot(
            manifest=manifest,
            namespace_report=namespace_report,
        )

        # Key fields should match (timestamp may differ)
        assert snap1.manifest_status == snap2.manifest_status
        assert snap1.seed_lineage_fingerprint == snap2.seed_lineage_fingerprint
        assert snap1.governance_status == snap2.governance_status

    def test_policy_eval_deterministic(self):
        """Policy evaluation is deterministic."""
        snapshot = build_prng_governance_snapshot(manifest=create_manifest())

        eval1 = evaluate_prng_policy(snapshot)
        eval2 = evaluate_prng_policy(snapshot)

        assert eval1.policy_ok == eval2.policy_ok
        assert eval1.status == eval2.status
        assert len(eval1.violations) == len(eval2.violations)

