# PHASE II — NOT USED IN PHASE I
"""
PRNG Governance Layer — Policy Enforcement for Deterministic Randomness.

This module provides a governance layer that fuses namespace linting,
manifest comparison, and lineage tracking into a unified policy framework.

Components:
    1. Governance Snapshot: Unified view of PRNG state across tools
    2. Policy Rules: Codified rules for PRNG compliance
    3. Global Health: Summary for MAAS integration

Usage:
    from rfl.prng.governance import (
        build_prng_governance_snapshot,
        evaluate_prng_policy,
        summarize_prng_for_global_health,
    )

    # Build snapshot from tool outputs
    snapshot = build_prng_governance_snapshot(
        manifest=manifest_data,
        replay_manifest=replay_data,
        namespace_report=lint_result,
    )

    # Evaluate policy
    policy_eval = evaluate_prng_policy(snapshot)

    # Get global health summary
    health = summarize_prng_for_global_health(snapshot, policy_eval)

Contract Reference:
    Implements PRNG governance requirements from docs/DETERMINISM_CONTRACT.md.

Author: Agent A2 (runtime-ops-2)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class GovernanceStatus(str, Enum):
    """Overall governance status."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


class ManifestStatus(str, Enum):
    """Status from manifest comparison."""
    EQUIVALENT = "EQUIVALENT"
    DRIFTED = "DRIFTED"
    INCOMPATIBLE = "INCOMPATIBLE"
    MISSING = "MISSING"


# ═══════════════════════════════════════════════════════════════════════════════
# PRNG POLICY RULES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PolicyRule:
    """A single PRNG policy rule."""
    rule_id: str
    name: str
    description: str
    severity: GovernanceStatus  # WARN or BLOCK
    applies_to: str  # "evidence", "runtime", "all"


# Codified PRNG governance rules
PRNG_GOV_RULES: Dict[str, PolicyRule] = {
    "R1": PolicyRule(
        rule_id="R1",
        name="no_drifted_evidence",
        description="No DRIFTED schedules allowed for evidence runs",
        severity=GovernanceStatus.BLOCK,
        applies_to="evidence",
    ),
    "R2": PolicyRule(
        rule_id="R2",
        name="incompatible_block",
        description="INCOMPATIBLE manifests must be blocked",
        severity=GovernanceStatus.BLOCK,
        applies_to="all",
    ),
    "R3": PolicyRule(
        rule_id="R3",
        name="no_hardcoded_runtime",
        description="Hard-coded seeds only allowed in tests; BLOCK in runtime paths",
        severity=GovernanceStatus.BLOCK,
        applies_to="runtime",
    ),
    "R4": PolicyRule(
        rule_id="R4",
        name="namespace_collision_warn",
        description="Namespace collisions should be reviewed",
        severity=GovernanceStatus.WARN,
        applies_to="all",
    ),
    "R5": PolicyRule(
        rule_id="R5",
        name="missing_attestation_warn",
        description="Manifests should have PRNG attestation blocks",
        severity=GovernanceStatus.WARN,
        applies_to="all",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NamespaceIssues:
    """Summary of namespace linting issues."""
    duplicate_count: int = 0
    duplicate_files: List[str] = field(default_factory=list)
    hardcoded_seed_count: int = 0
    hardcoded_seed_files: List[str] = field(default_factory=list)
    dynamic_path_count: int = 0
    suppressed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duplicate_count": self.duplicate_count,
            "duplicate_files": self.duplicate_files,
            "hardcoded_seed_count": self.hardcoded_seed_count,
            "hardcoded_seed_files": self.hardcoded_seed_files,
            "dynamic_path_count": self.dynamic_path_count,
            "suppressed_count": self.suppressed_count,
        }


@dataclass
class PolicyViolation:
    """A single policy violation."""
    rule_id: str
    rule_name: str
    severity: GovernanceStatus
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class PRNGGovernanceSnapshot:
    """
    Unified snapshot of PRNG governance state.

    Fuses outputs from:
    - seed_namespace_linter.py
    - prng_manifest_diff.py
    - seed_lineage_tree.py
    """
    schema_version: str = "1.0"
    timestamp: str = ""
    manifest_status: ManifestStatus = ManifestStatus.MISSING
    namespace_issues: NamespaceIssues = field(default_factory=NamespaceIssues)
    hardcoded_seeds_detected: bool = False
    hardcoded_seed_count: int = 0
    seed_lineage_fingerprint: str = ""
    master_seed_hex: Optional[str] = None
    lineage_entry_count: int = 0
    governance_status: GovernanceStatus = GovernanceStatus.OK
    is_evidence_run: bool = False
    is_test_context: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "manifest_status": self.manifest_status.value,
            "namespace_issues": self.namespace_issues.to_dict(),
            "hardcoded_seeds_detected": self.hardcoded_seeds_detected,
            "hardcoded_seed_count": self.hardcoded_seed_count,
            "seed_lineage_fingerprint": self.seed_lineage_fingerprint,
            "master_seed_hex": self.master_seed_hex,
            "lineage_entry_count": self.lineage_entry_count,
            "governance_status": self.governance_status.value,
            "is_evidence_run": self.is_evidence_run,
            "is_test_context": self.is_test_context,
        }


@dataclass
class PolicyEvaluation:
    """Result of policy evaluation."""
    policy_ok: bool = True
    violations: List[PolicyViolation] = field(default_factory=list)
    status: GovernanceStatus = GovernanceStatus.OK
    rules_checked: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_ok": self.policy_ok,
            "violations": [v.to_dict() for v in self.violations],
            "status": self.status.value,
            "rules_checked": self.rules_checked,
            "timestamp": self.timestamp,
        }


@dataclass
class GlobalHealthSummary:
    """Summary for global health / MAAS integration."""
    prng_policy_ok: bool = True
    has_namespace_collisions: bool = False
    has_schedule_drift: bool = False
    has_hardcoded_seeds: bool = False
    status: GovernanceStatus = GovernanceStatus.OK
    violation_count: int = 0
    summary_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prng_policy_ok": self.prng_policy_ok,
            "has_namespace_collisions": self.has_namespace_collisions,
            "has_schedule_drift": self.has_schedule_drift,
            "has_hardcoded_seeds": self.has_hardcoded_seeds,
            "status": self.status.value,
            "violation_count": self.violation_count,
            "summary_message": self.summary_message,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: PRNG GOVERNANCE SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_lineage_fingerprint(
    master_seed_hex: Optional[str],
    merkle_root: Optional[str],
    entry_count: int,
) -> str:
    """
    Compute a short fingerprint for seed lineage.

    This fingerprint uniquely identifies the lineage state for quick comparison.
    """
    if not master_seed_hex:
        return "no-seed"

    material = f"{master_seed_hex}:{merkle_root or 'none'}:{entry_count}"
    digest = hashlib.sha256(material.encode('utf-8')).hexdigest()
    return digest[:16]  # Short fingerprint


def _extract_namespace_issues(namespace_report: Optional[Dict[str, Any]]) -> NamespaceIssues:
    """Extract namespace issues from linter report."""
    if not namespace_report:
        return NamespaceIssues()

    issues = NamespaceIssues()

    # Extract duplicates
    duplicates = namespace_report.get("duplicates", [])
    issues.duplicate_count = len(duplicates)
    for dup in duplicates:
        usages = dup.get("usages", [])
        for usage in usages:
            file_path = usage.get("file", "")
            if file_path and file_path not in issues.duplicate_files:
                issues.duplicate_files.append(file_path)

    # Extract hard-coded seeds
    hard_coded = namespace_report.get("hard_coded_seeds", [])
    issues.hardcoded_seed_count = len(hard_coded)
    for seed in hard_coded:
        file_path = seed.get("file", "")
        if file_path and file_path not in issues.hardcoded_seed_files:
            issues.hardcoded_seed_files.append(file_path)

    # Extract dynamic paths
    dynamic = namespace_report.get("dynamic_paths", [])
    issues.dynamic_path_count = len(dynamic)

    # Suppressed count
    issues.suppressed_count = namespace_report.get("suppressed_count", 0)

    return issues


def _determine_manifest_status(
    manifest: Optional[Dict[str, Any]],
    replay_manifest: Optional[Dict[str, Any]],
    diff_result: Optional[Dict[str, Any]] = None,
) -> ManifestStatus:
    """Determine manifest comparison status."""
    if diff_result:
        status_str = diff_result.get("status", "MISSING")
        try:
            return ManifestStatus(status_str)
        except ValueError:
            return ManifestStatus.MISSING

    if manifest is None:
        return ManifestStatus.MISSING

    if replay_manifest is None:
        # No replay to compare against
        return ManifestStatus.EQUIVALENT

    # Extract attestations
    attest1 = manifest.get("prng_attestation") or manifest.get("prng", {})
    attest2 = replay_manifest.get("prng_attestation") or replay_manifest.get("prng", {})

    # Compare key fields
    seed1 = attest1.get("master_seed_hex")
    seed2 = attest2.get("master_seed_hex")
    scheme1 = attest1.get("derivation_scheme")
    scheme2 = attest2.get("derivation_scheme")
    merkle1 = attest1.get("lineage_merkle_root")
    merkle2 = attest2.get("lineage_merkle_root")

    if scheme1 != scheme2:
        return ManifestStatus.INCOMPATIBLE

    if seed1 == seed2 and merkle1 != merkle2:
        return ManifestStatus.DRIFTED

    return ManifestStatus.EQUIVALENT


def _compute_initial_governance_status(
    manifest_status: ManifestStatus,
    namespace_issues: NamespaceIssues,
    is_evidence_run: bool,
) -> GovernanceStatus:
    """Compute initial governance status before policy evaluation."""
    if manifest_status == ManifestStatus.INCOMPATIBLE:
        return GovernanceStatus.BLOCK

    if manifest_status == ManifestStatus.DRIFTED and is_evidence_run:
        return GovernanceStatus.BLOCK

    if namespace_issues.duplicate_count > 0:
        return GovernanceStatus.WARN

    if namespace_issues.hardcoded_seed_count > 0:
        return GovernanceStatus.WARN

    return GovernanceStatus.OK


def build_prng_governance_snapshot(
    manifest: Optional[Dict[str, Any]] = None,
    replay_manifest: Optional[Dict[str, Any]] = None,
    namespace_report: Optional[Dict[str, Any]] = None,
    diff_result: Optional[Dict[str, Any]] = None,
    is_evidence_run: bool = False,
    is_test_context: bool = False,
) -> PRNGGovernanceSnapshot:
    """
    Build a unified PRNG governance snapshot.

    Fuses outputs from:
    - Manifest data (prng_attestation block)
    - Manifest diff result
    - Namespace linter report

    Args:
        manifest: Primary manifest data.
        replay_manifest: Replay/comparison manifest data.
        namespace_report: Output from seed_namespace_linter.
        diff_result: Output from prng_manifest_diff (optional, will compute if not provided).
        is_evidence_run: Whether this is an evidence-gathering run.
        is_test_context: Whether this is running in a test context.

    Returns:
        PRNGGovernanceSnapshot with unified view.
    """
    # Extract attestation from manifest
    attestation = {}
    if manifest:
        attestation = manifest.get("prng_attestation") or manifest.get("prng", {})

    master_seed_hex = attestation.get("master_seed_hex")
    merkle_root = attestation.get("lineage_merkle_root")
    entry_count = attestation.get("lineage_entry_count", 0)

    # Determine manifest status
    manifest_status = _determine_manifest_status(manifest, replay_manifest, diff_result)

    # Extract namespace issues
    namespace_issues = _extract_namespace_issues(namespace_report)

    # Compute lineage fingerprint
    fingerprint = _compute_lineage_fingerprint(master_seed_hex, merkle_root, entry_count)

    # Determine initial governance status
    governance_status = _compute_initial_governance_status(
        manifest_status, namespace_issues, is_evidence_run
    )

    return PRNGGovernanceSnapshot(
        manifest_status=manifest_status,
        namespace_issues=namespace_issues,
        hardcoded_seeds_detected=namespace_issues.hardcoded_seed_count > 0,
        hardcoded_seed_count=namespace_issues.hardcoded_seed_count,
        seed_lineage_fingerprint=fingerprint,
        master_seed_hex=master_seed_hex,
        lineage_entry_count=entry_count,
        governance_status=governance_status,
        is_evidence_run=is_evidence_run,
        is_test_context=is_test_context,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: PRNG POLICY RULES EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def _check_rule_r1(snapshot: PRNGGovernanceSnapshot) -> Optional[PolicyViolation]:
    """R1: No DRIFTED schedules allowed for evidence runs."""
    rule = PRNG_GOV_RULES["R1"]

    if not snapshot.is_evidence_run:
        return None  # Rule doesn't apply

    if snapshot.manifest_status == ManifestStatus.DRIFTED:
        return PolicyViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            message="Evidence run has DRIFTED schedule - determinism compromised",
            context={
                "manifest_status": snapshot.manifest_status.value,
                "fingerprint": snapshot.seed_lineage_fingerprint,
            },
        )

    return None


def _check_rule_r2(snapshot: PRNGGovernanceSnapshot) -> Optional[PolicyViolation]:
    """R2: INCOMPATIBLE manifests must be blocked."""
    rule = PRNG_GOV_RULES["R2"]

    if snapshot.manifest_status == ManifestStatus.INCOMPATIBLE:
        return PolicyViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            message="Manifest is INCOMPATIBLE - cannot verify determinism",
            context={
                "manifest_status": snapshot.manifest_status.value,
            },
        )

    return None


def _check_rule_r3(snapshot: PRNGGovernanceSnapshot) -> Optional[PolicyViolation]:
    """R3: Hard-coded seeds only allowed in tests; BLOCK in runtime paths."""
    rule = PRNG_GOV_RULES["R3"]

    if snapshot.is_test_context:
        return None  # Hard-coded seeds allowed in tests

    if snapshot.hardcoded_seeds_detected:
        # Check if any hard-coded seeds are in non-test files
        non_test_files = [
            f for f in snapshot.namespace_issues.hardcoded_seed_files
            if not _is_test_file(f)
        ]

        if non_test_files:
            return PolicyViolation(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                message=f"Hard-coded seeds found in {len(non_test_files)} runtime file(s)",
                context={
                    "files": non_test_files[:5],  # Limit to first 5
                    "total_count": len(non_test_files),
                },
            )

    return None


def _check_rule_r4(snapshot: PRNGGovernanceSnapshot) -> Optional[PolicyViolation]:
    """R4: Namespace collisions should be reviewed."""
    rule = PRNG_GOV_RULES["R4"]

    if snapshot.namespace_issues.duplicate_count > 0:
        return PolicyViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=f"{snapshot.namespace_issues.duplicate_count} namespace collision(s) detected",
            context={
                "duplicate_count": snapshot.namespace_issues.duplicate_count,
                "files": snapshot.namespace_issues.duplicate_files[:5],
            },
        )

    return None


def _check_rule_r5(snapshot: PRNGGovernanceSnapshot) -> Optional[PolicyViolation]:
    """R5: Manifests should have PRNG attestation blocks."""
    rule = PRNG_GOV_RULES["R5"]

    if snapshot.manifest_status == ManifestStatus.MISSING:
        return PolicyViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            message="Manifest missing or lacks PRNG attestation",
            context={},
        )

    if not snapshot.master_seed_hex:
        return PolicyViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            message="PRNG attestation block incomplete (no master_seed_hex)",
            context={},
        )

    return None


def _is_test_file(file_path: str) -> bool:
    """Check if a file path is a test file."""
    path_lower = file_path.lower().replace("\\", "/")
    return (
        "/test_" in path_lower or
        "/tests/" in path_lower or
        "_test.py" in path_lower or
        "conftest.py" in path_lower
    )


def evaluate_prng_policy(
    snapshot: PRNGGovernanceSnapshot,
    rules_to_check: Optional[List[str]] = None,
) -> PolicyEvaluation:
    """
    Evaluate PRNG policy rules against a governance snapshot.

    Args:
        snapshot: PRNGGovernanceSnapshot to evaluate.
        rules_to_check: Optional list of rule IDs to check. If None, checks all.

    Returns:
        PolicyEvaluation with results.
    """
    if rules_to_check is None:
        rules_to_check = list(PRNG_GOV_RULES.keys())

    violations: List[PolicyViolation] = []
    rules_checked: List[str] = []

    # Rule checkers
    rule_checkers = {
        "R1": _check_rule_r1,
        "R2": _check_rule_r2,
        "R3": _check_rule_r3,
        "R4": _check_rule_r4,
        "R5": _check_rule_r5,
    }

    for rule_id in rules_to_check:
        if rule_id not in rule_checkers:
            continue

        rules_checked.append(rule_id)
        checker = rule_checkers[rule_id]
        violation = checker(snapshot)

        if violation:
            violations.append(violation)

    # Determine overall status
    has_block = any(v.severity == GovernanceStatus.BLOCK for v in violations)
    has_warn = any(v.severity == GovernanceStatus.WARN for v in violations)

    if has_block:
        status = GovernanceStatus.BLOCK
    elif has_warn:
        status = GovernanceStatus.WARN
    else:
        status = GovernanceStatus.OK

    policy_ok = status == GovernanceStatus.OK

    return PolicyEvaluation(
        policy_ok=policy_ok,
        violations=violations,
        status=status,
        rules_checked=rules_checked,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: GLOBAL HEALTH & MAAS HOOK
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_prng_for_global_health(
    snapshot: PRNGGovernanceSnapshot,
    policy_eval: PolicyEvaluation,
) -> GlobalHealthSummary:
    """
    Summarize PRNG state for global health / MAAS integration.

    This provides a simplified view suitable for dashboards and monitoring.

    Args:
        snapshot: PRNGGovernanceSnapshot.
        policy_eval: PolicyEvaluation result.

    Returns:
        GlobalHealthSummary with key indicators.
    """
    # Determine key flags
    has_namespace_collisions = snapshot.namespace_issues.duplicate_count > 0
    has_schedule_drift = snapshot.manifest_status == ManifestStatus.DRIFTED
    has_hardcoded_seeds = snapshot.hardcoded_seeds_detected

    # Build summary message
    issues = []
    if has_schedule_drift:
        issues.append("schedule drift")
    if has_namespace_collisions:
        issues.append(f"{snapshot.namespace_issues.duplicate_count} namespace collision(s)")
    if has_hardcoded_seeds:
        issues.append(f"{snapshot.hardcoded_seed_count} hard-coded seed(s)")

    if policy_eval.policy_ok and not issues:
        summary_message = "PRNG governance: OK"
    elif policy_eval.policy_ok and issues:
        summary_message = f"PRNG governance: OK (noted: {', '.join(issues)})"
    else:
        summary_message = f"PRNG governance: {', '.join(issues) if issues else 'policy violations'}"

    return GlobalHealthSummary(
        prng_policy_ok=policy_eval.policy_ok,
        has_namespace_collisions=has_namespace_collisions,
        has_schedule_drift=has_schedule_drift,
        has_hardcoded_seeds=has_hardcoded_seeds,
        status=policy_eval.status,
        violation_count=len(policy_eval.violations),
        summary_message=summary_message,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_prng_governance(
    manifest: Optional[Dict[str, Any]] = None,
    replay_manifest: Optional[Dict[str, Any]] = None,
    namespace_report: Optional[Dict[str, Any]] = None,
    is_evidence_run: bool = False,
    is_test_context: bool = False,
) -> Dict[str, Any]:
    """
    Run the full PRNG governance pipeline.

    Convenience function that builds snapshot, evaluates policy, and
    generates global health summary in one call.

    Args:
        manifest: Primary manifest data.
        replay_manifest: Replay/comparison manifest data.
        namespace_report: Output from seed_namespace_linter.
        is_evidence_run: Whether this is an evidence-gathering run.
        is_test_context: Whether this is running in a test context.

    Returns:
        Dict with 'snapshot', 'policy_eval', and 'health' keys.
    """
    snapshot = build_prng_governance_snapshot(
        manifest=manifest,
        replay_manifest=replay_manifest,
        namespace_report=namespace_report,
        is_evidence_run=is_evidence_run,
        is_test_context=is_test_context,
    )

    policy_eval = evaluate_prng_policy(snapshot)
    health = summarize_prng_for_global_health(snapshot, policy_eval)

    return {
        "snapshot": snapshot.to_dict(),
        "policy_eval": policy_eval.to_dict(),
        "health": health.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: CI-FACING GATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_prng_for_ci(health: GlobalHealthSummary) -> int:
    """
    Evaluate PRNG governance for CI gate decision.

    Maps governance status to exit codes:
        - 0: OK (governance passed)
        - 1: WARN (governance issues but non-blocking)
        - 2: BLOCK (governance violations must be fixed)

    Args:
        health: GlobalHealthSummary from governance pipeline.

    Returns:
        Exit code: 0 (OK), 1 (WARN), or 2 (BLOCK).
    """
    if health.status == GovernanceStatus.OK:
        return 0
    elif health.status == GovernanceStatus.WARN:
        return 1
    elif health.status == GovernanceStatus.BLOCK:
        return 2
    else:
        # Unknown status defaults to BLOCK
        return 2


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: AUTO-REMEDIATION SUGGESTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_prng_remediation_suggestions(
    snapshot: PRNGGovernanceSnapshot,
    policy_eval: PolicyEvaluation,
) -> List[Dict[str, Any]]:
    """
    Build auto-remediation suggestions for PRNG governance violations.

    Generates descriptive, actionable suggestions for fixing governance issues.
    This is read-only; it never modifies code.

    Args:
        snapshot: PRNGGovernanceSnapshot with current state.
        policy_eval: PolicyEvaluation with violations.

    Returns:
        List of suggestion dictionaries, each with:
        - rule_id: Rule that triggered the suggestion
        - impact: Descriptive impact statement
        - suggested_action: Actionable remediation step
        - files_involved: List of file paths affected
    """
    suggestions: List[Dict[str, Any]] = []

    for violation in policy_eval.violations:
        rule_id = violation.rule_id
        context = violation.context

        if rule_id == "R1":
            # DRIFTED schedule in evidence run
            suggestions.append({
                "rule_id": rule_id,
                "impact": "Evidence run has non-deterministic seed schedule, compromising reproducibility",
                "suggested_action": (
                    "Investigate why seed derivation differs from expected. "
                    "Verify that master_seed_hex matches between runs and that "
                    "no code changes affected PRNG derivation logic."
                ),
                "files_involved": [],
            })

        elif rule_id == "R2":
            # INCOMPATIBLE manifest
            suggestions.append({
                "rule_id": rule_id,
                "impact": "Manifests use incompatible derivation schemes, preventing comparison",
                "suggested_action": (
                    "Ensure both manifests use the same derivation_scheme. "
                    "If schemes differ, regenerate one manifest using the canonical scheme: "
                    "'PRNGKey(root, path) -> SHA256 -> seed % 2^32'"
                ),
                "files_involved": [],
            })

        elif rule_id == "R3":
            # Hard-coded seeds in runtime
            files = context.get("files", [])
            suggestions.append({
                "rule_id": rule_id,
                "impact": f"Hard-coded seeds found in {len(files)} runtime file(s), breaking determinism",
                "suggested_action": (
                    "Replace hard-coded seeds with DeterministicPRNG.for_path() calls. "
                    "Example: Replace `random.seed(42)` with "
                    "`prng = DeterministicPRNG(master_seed_hex); rng = prng.for_path('component', 'subcomponent')`"
                ),
                "files_involved": files[:10],  # Limit to first 10
            })

        elif rule_id == "R4":
            # Namespace collisions
            files = context.get("files", [])
            duplicate_count = context.get("duplicate_count", 0)
            suggestions.append({
                "rule_id": rule_id,
                "impact": f"{duplicate_count} namespace collision(s) detected, risking seed reuse",
                "suggested_action": (
                    "Review namespace paths to ensure uniqueness. "
                    "If intentional reuse, add `# prng: namespace-ok` comment. "
                    "Otherwise, add distinguishing labels to paths (e.g., component name, cycle index)."
                ),
                "files_involved": files[:10],  # Limit to first 10
            })

        elif rule_id == "R5":
            # Missing attestation
            suggestions.append({
                "rule_id": rule_id,
                "impact": "Manifest missing or incomplete PRNG attestation block",
                "suggested_action": (
                    "Add prng_attestation block to manifest with: "
                    "master_seed_hex, derivation_scheme, lineage_merkle_root, and lineage_entry_count. "
                    "Use ManifestBuilder.build() which automatically includes attestation."
                ),
                "files_involved": [],
            })

    # Add suggestions for issues not covered by violations
    if snapshot.namespace_issues.dynamic_path_count > 0:
        suggestions.append({
            "rule_id": "INFO",
            "impact": f"{snapshot.namespace_issues.dynamic_path_count} dynamic namespace path(s) detected",
            "suggested_action": (
                "Review dynamic paths to ensure they produce deterministic namespaces. "
                "Consider using constant labels where possible for better traceability."
            ),
            "files_involved": [],
        })

    return suggestions


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: PRNG GOVERNANCE HISTORY LEDGER
# ═══════════════════════════════════════════════════════════════════════════════

def build_prng_governance_history(
    snapshots: List[PRNGGovernanceSnapshot],
    run_ids: Optional[List[str]] = None,
    policy_evaluations: Optional[List[PolicyEvaluation]] = None,
) -> Dict[str, Any]:
    """
    Build a PRNG governance history ledger from a sequence of snapshots.

    Creates a canonical history record suitable for long-horizon analysis,
    audit trails, and integration with global health systems.

    Args:
        snapshots: Sequence of PRNGGovernanceSnapshot objects.
        run_ids: Optional list of run identifiers (one per snapshot).
                If None, generates sequential IDs.
        policy_evaluations: Optional list of PolicyEvaluation objects (one per snapshot).
                          If provided, violations are included in run records.

    Returns:
        Dict with:
        - schema_version: "1.0"
        - runs: List of run records (with violations if policy_evaluations provided)
        - status_counts: Aggregated counts by status
        - history_hash: SHA-256 hash of canonical history body
    """
    if run_ids is None:
        run_ids = [f"run_{i:04d}" for i in range(len(snapshots))]
    elif len(run_ids) != len(snapshots):
        raise ValueError(f"run_ids length ({len(run_ids)}) must match snapshots length ({len(snapshots)})")

    if policy_evaluations is not None and len(policy_evaluations) != len(snapshots):
        raise ValueError(
            f"policy_evaluations length ({len(policy_evaluations)}) must match snapshots length ({len(snapshots)})"
        )

    # Build run records
    runs = []
    for i, (snapshot, run_id) in enumerate(zip(snapshots, run_ids)):
        run_record = {
            "run_id": run_id,
            "governance_status": snapshot.governance_status.value,
            "manifest_status": snapshot.manifest_status.value,
            "has_hardcoded_seeds": snapshot.hardcoded_seeds_detected,
            "hardcoded_seed_count": snapshot.hardcoded_seed_count,
            "namespace_duplicate_count": snapshot.namespace_issues.duplicate_count,
            "lineage_fingerprint": snapshot.seed_lineage_fingerprint,
            "timestamp": snapshot.timestamp,
        }

        # Add violations if policy evaluations provided
        if policy_evaluations:
            policy_eval = policy_evaluations[i]
            run_record["violations"] = [
                {
                    "rule_id": v.rule_id,
                    "kind": v.rule_name,
                    "severity": v.severity.value,
                }
                for v in policy_eval.violations
            ]
        else:
            run_record["violations"] = []

        runs.append(run_record)

    # Compute status counts
    status_counts = {
        "OK": sum(1 for r in runs if r["governance_status"] == "OK"),
        "WARN": sum(1 for r in runs if r["governance_status"] == "WARN"),
        "BLOCK": sum(1 for r in runs if r["governance_status"] == "BLOCK"),
    }

    manifest_status_counts = {}
    for run in runs:
        status = run["manifest_status"]
        manifest_status_counts[status] = manifest_status_counts.get(status, 0) + 1

    # Build history body (deterministic ordering)
    history_body = {
        "schema_version": "1.0",
        "total_runs": len(runs),
        "runs": sorted(runs, key=lambda r: (r["run_id"], r["timestamp"])),  # Deterministic sort
        "status_counts": status_counts,
        "manifest_status_counts": dict(sorted(manifest_status_counts.items())),
    }

    # Compute canonical hash
    canonical_json = json.dumps(history_body, sort_keys=True, separators=(",", ":"))
    history_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return {
        "schema_version": "1.0",
        "total_runs": len(runs),
        "runs": history_body["runs"],
        "status_counts": status_counts,
        "manifest_status_counts": manifest_status_counts,
        "history_hash": history_hash,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK: PRNG DRIFT RADAR & EVIDENCE PACK TILE
# ═══════════════════════════════════════════════════════════════════════════════

class DriftStatus(str, Enum):
    """Status of PRNG governance drift."""
    STABLE = "STABLE"
    DRIFTING = "DRIFTING"
    VOLATILE = "VOLATILE"


# Drift radar thresholds (configurable constants)
DRIFT_RADAR_FREQUENCY_THRESHOLD = 3  # Rule must appear in >= 3 runs to be "frequent"
DRIFT_RADAR_STABLE_BLOCK_THRESHOLD = 0.10  # <10% BLOCK runs for STABLE
DRIFT_RADAR_VOLATILE_BLOCK_THRESHOLD = 0.30  # >=30% BLOCK runs for VOLATILE
DRIFT_RADAR_VOLATILE_COMBINED_THRESHOLD = 0.50  # >=50% combined WARN+BLOCK for VOLATILE


def build_prng_drift_radar(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build PRNG drift radar from governance history.

    Analyzes governance history to detect patterns and classify drift status:
    - STABLE: No frequent violations
    - DRIFTING: 1-2 frequent violations
    - VOLATILE: ≥3 frequent violations

    A frequent violation is a rule_id that appears as a violation in ≥3 runs.

    Args:
        history: Governance history from build_prng_governance_history().

    Returns:
        Dict with:
        - schema_version: "1.0.0"
        - drift_status: STABLE | DRIFTING | VOLATILE
        - frequent_violations: Dict mapping rule_id to count (only rules with count >= 3)
        - total_runs: Total number of runs analyzed
    """
    if not history or history.get("total_runs", 0) == 0:
        return {
            "schema_version": "1.0.0",
            "drift_status": DriftStatus.STABLE.value,
            "frequent_violations": {},
            "total_runs": 0,
        }

    runs = history.get("runs", [])
    total_runs = history.get("total_runs", len(runs))

    # Count violation occurrences per rule_id across all runs
    rule_counts: Dict[str, int] = {}
    for run in runs:
        violations = run.get("violations", [])
        for violation in violations:
            rule_id = violation.get("rule_id")
            if rule_id:
                rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

    # Apply frequency threshold: rule must appear in ≥3 runs to be "frequent"
    frequent_violations = {
        rule_id: count
        for rule_id, count in rule_counts.items()
        if count >= DRIFT_RADAR_FREQUENCY_THRESHOLD
    }

    # Classify drift status based on frequent violations count
    frequent_count = len(frequent_violations)
    
    if frequent_count == 0:
        drift_status = DriftStatus.STABLE
    elif frequent_count <= 2:
        drift_status = DriftStatus.DRIFTING
    else:  # frequent_count >= 3
        drift_status = DriftStatus.VOLATILE

    return {
        "schema_version": "1.0.0",
        "drift_status": drift_status.value,
        "frequent_violations": dict(sorted(frequent_violations.items())),  # Deterministic ordering
        "total_runs": total_runs,
    }


def build_prng_governance_tile(
    history: Dict[str, Any],
    radar: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build PRNG governance tile for evidence pack.

    Creates a PRNG section suitable for external auditors and evidence chains.
    This tile provides a compact, schema-compliant view of PRNG governance state.

    Args:
        history: Governance history from build_prng_governance_history().
        radar: Optional drift radar from build_prng_drift_radar().
                If None, will be computed from history.

    Returns:
        Dict with PRNG tile structure:
        - schema_version: "1.0.0"
        - status: OK | WARN | BLOCK
        - drift_status: STABLE | DRIFTING | VOLATILE
        - blocking_rules: List of rule IDs that caused BLOCK status (from BLOCK runs)
        - headline: Neutral summary string
    """
    if radar is None:
        radar = build_prng_drift_radar(history)

    runs = history.get("runs", [])

    # Extract all BLOCK-status runs and their violations
    block_runs = [
        run for run in runs
        if run.get("governance_status") == "BLOCK"
    ]

    # Extract blocking rules: all rule_ids with severity="BLOCK" from BLOCK runs
    blocking_rules_set: set[str] = set()
    for run in block_runs:
        violations = run.get("violations", [])
        for violation in violations:
            if violation.get("severity") == "BLOCK":
                rule_id = violation.get("rule_id")
                if rule_id:
                    blocking_rules_set.add(rule_id)

    blocking_rules = sorted(blocking_rules_set)  # Deterministic ordering

    # Get drift status from radar
    drift_status_str = radar.get("drift_status", DriftStatus.STABLE.value)
    frequent_violations = radar.get("frequent_violations", {})
    has_frequent_violations = len(frequent_violations) > 0
    has_block_runs = len(block_runs) > 0

    # Determine status:
    # BLOCK: Any BLOCK runs with violations
    if has_block_runs:
        status = GovernanceStatus.BLOCK.value
    # WARN: Any frequent violations in drift radar (but no BLOCK runs)
    elif has_frequent_violations:
        status = GovernanceStatus.WARN.value
    # OK: Otherwise
    else:
        status = GovernanceStatus.OK.value

    # Build neutral headline
    if status == GovernanceStatus.BLOCK.value:
        headline = f"PRNG governance: {len(block_runs)} blocking violation(s) detected"
    elif status == GovernanceStatus.WARN.value:
        headline = f"PRNG governance: drift detected ({drift_status_str.lower()})"
    else:
        headline = "PRNG governance: compliant"

    return {
        "schema_version": "1.0.0",
        "status": status,
        "drift_status": drift_status_str,
        "blocking_rules": blocking_rules,
        "headline": headline,
    }


def attach_prng_tile_to_evidence(
    evidence_payload: Dict[str, Any],
    history: Dict[str, Any],
    radar: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach PRNG governance tile to an evidence payload.

    This is an advisory-only integration; the tile provides observability
    but does not enforce decisions.

    Args:
        evidence_payload: Evidence payload dictionary (will be modified in-place).
        history: Governance history from build_prng_governance_history().
        radar: Optional drift radar from build_prng_drift_radar().
                If None, will be computed from history.

    Returns:
        Modified evidence_payload with prng_governance tile attached.
    """
    tile = build_prng_governance_tile(history, radar=radar)
    
    # Attach under prng_governance key
    if "governance" not in evidence_payload:
        evidence_payload["governance"] = {}
    evidence_payload["governance"]["prng_governance"] = tile
    
    return evidence_payload


def build_first_light_prng_summary(
    radar: Optional[Dict[str, Any]] = None,
    tile: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build compact PRNG summary for First Light evidence pack.

    This provides a footnote-level stability signal suitable for First Light
    evidence packs. It is purely observational and does not gate any decisions.

    Args:
        radar: PRNG drift radar from build_prng_drift_radar().
        tile: PRNG governance tile from build_prng_governance_tile().

    Returns:
        Compact summary dict with:
        - schema_version: "1.0.0"
        - drift_status: "STABLE" | "DRIFTING" | "VOLATILE"
        - frequent_violations: Dict[str, int]
        - status: "OK" | "WARN" | "BLOCK"
        - blocking_rules: List[str]
        - total_runs: int

    Example:
        >>> radar = build_prng_drift_radar(history)
        >>> tile = build_prng_governance_tile(history, radar=radar)
        >>> summary = build_first_light_prng_summary(radar, tile)
        >>> summary["drift_status"]
        'STABLE'
    """
    radar = radar or {}
    tile = tile or {}
    return {
        "schema_version": "1.0.0",
        "drift_status": radar.get("drift_status", DriftStatus.STABLE.value),
        "frequent_violations": radar.get("frequent_violations", {}),
        "status": tile.get("status", GovernanceStatus.OK.value),
        "blocking_rules": tile.get("blocking_rules", []),
        "total_runs": radar.get("total_runs", 0),
        "seed": seed if seed is not None else "unknown",
        "notes": "First Light PRNG summary (demo)",
    }


def attach_prng_governance_tile(
    evidence: Dict[str, Any],
    prng_tile: Dict[str, Any],
    radar: Optional[Dict[str, Any]] = None,
    include_first_light_summary: bool = False,
) -> Dict[str, Any]:
    """
    Attach PRNG governance tile to an evidence pack (read-only, additive).

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the tile attached.

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        prng_tile: PRNG governance tile from build_prng_governance_tile().
        radar: Optional PRNG drift radar from build_prng_drift_radar().
               Required if include_first_light_summary=True.
        include_first_light_summary: If True, attach first_light_summary sub-object.

    Returns:
        New dict with evidence contents plus prng_governance tile attached.
        If include_first_light_summary=True and radar provided, includes
        first_light_summary sub-object.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_prng_governance_tile(history)
        >>> enriched = attach_prng_governance_tile(evidence, tile)
        >>> "prng_governance" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Attach PRNG governance tile under governance.prng_governance key
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    # Create a copy of the tile to avoid mutating the original
    tile_copy = prng_tile.copy()
    
    # Optionally attach first_light_summary
    if include_first_light_summary:
        if radar is None:
            raise ValueError("radar is required when include_first_light_summary=True")
        tile_copy["first_light_summary"] = build_first_light_prng_summary(radar, prng_tile)
    
    enriched["governance"]["prng_governance"] = tile_copy
    
    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
# PRNG DRIFT LEDGER — LONG-HORIZON AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_prng_drift_ledger(tiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build PRNG drift ledger from multiple per-run governance tiles.

    Aggregates multiple PRNG governance tiles (one per run or time period) into
    a long-horizon ledger view suitable for Phase X evidence and dashboards.

    This is advisory-only; it provides observability but does not enforce decisions.

    Args:
        tiles: List of PRNG governance tiles from build_prng_governance_tile().
               Each tile should contain:
               - drift_status: "STABLE" | "DRIFTING" | "VOLATILE"
               - blocking_rules: List[str] (rule IDs)

    Returns:
        Dict with:
        - schema_version: "1.0.0"
        - total_runs: Total number of tiles/runs analyzed
        - volatile_runs: Number of tiles with drift_status="VOLATILE"
        - drifting_runs: Number of tiles with drift_status="DRIFTING"
        - stable_runs: Number of tiles with drift_status="STABLE"
        - frequent_rules: Dict[str, int] mapping rule_id to count across all tiles

    Example:
        >>> tiles = [
        ...     {"drift_status": "STABLE", "blocking_rules": []},
        ...     {"drift_status": "DRIFTING", "blocking_rules": ["R1"]},
        ...     {"drift_status": "VOLATILE", "blocking_rules": ["R1", "R2"]},
        ... ]
        >>> ledger = build_prng_drift_ledger(tiles)
        >>> ledger["total_runs"]
        3
        >>> ledger["volatile_runs"]
        1

    P5 Usage Note:
        The ledger is designed to be compared across multiple First Light runs
        (mock vs real). A single run's PRNG tile is not sufficient to judge
        anything about P5 behavior. The ledger aggregates drift patterns over
        time, providing long-horizon context for deterministic randomness
        governance. For early P5, this is context only, not a gate.
    """
    if not tiles:
        return {
            "schema_version": "1.0.0",
            "total_runs": 0,
            "volatile_runs": 0,
            "drifting_runs": 0,
            "stable_runs": 0,
            "frequent_rules": {},
        }

    # Classify runs by drift status
    volatile_runs = 0
    drifting_runs = 0
    stable_runs = 0

    # Aggregate blocking rules across all tiles
    rule_counts: Dict[str, int] = {}

    for tile in tiles:
        drift_status = tile.get("drift_status", DriftStatus.STABLE.value)
        
        # Count by drift status
        if drift_status == DriftStatus.VOLATILE.value:
            volatile_runs += 1
        elif drift_status == DriftStatus.DRIFTING.value:
            drifting_runs += 1
        else:  # STABLE
            stable_runs += 1

        # Aggregate blocking rules
        blocking_rules = tile.get("blocking_rules", [])
        for rule_id in blocking_rules:
            if rule_id:
                rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

    # Sort frequent_rules for determinism
    frequent_rules = dict(sorted(rule_counts.items()))

    return {
        "schema_version": "1.0.0",
        "total_runs": len(tiles),
        "volatile_runs": volatile_runs,
        "drifting_runs": drifting_runs,
        "stable_runs": stable_runs,
        "frequent_rules": frequent_rules,
    }


def build_prng_regime_comparison(
    mock_ledger: Dict[str, Any],
    real_ledger: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build PRNG regime comparison between mock and real calibration experiment ledgers.

    Compares two PRNG drift ledgers (typically from CAL-EXP-* mock vs real runs) to
    identify differences in drift patterns, rule frequencies, and regime stability.
    This is purely observational and provides context for P5 calibration analysis.

    Args:
        mock_ledger: PRNG drift ledger from mock calibration experiment runs.
        real_ledger: PRNG drift ledger from real calibration experiment runs.

    Returns:
        Dict with:
        - schema_version: "1.0.0"
        - mock_drift_status: Most common drift_status from mock ledger (or "STABLE" if empty)
        - real_drift_status: Most common drift_status from real ledger (or "STABLE" if empty)
        - delta_volatile_runs: real_volatile_runs - mock_volatile_runs
        - delta_stable_runs: real_stable_runs - mock_stable_runs
        - rules_more_frequent_in_real: List of rule IDs that appear more often in real
        - rules_more_frequent_in_mock: List of rule IDs that appear more often in mock

    Example:
        >>> mock_ledger = {"volatile_runs": 2, "stable_runs": 8, "frequent_rules": {"R1": 3}}
        >>> real_ledger = {"volatile_runs": 1, "stable_runs": 9, "frequent_rules": {"R1": 5, "R2": 2}}
        >>> comparison = build_prng_regime_comparison(mock_ledger, real_ledger)
        >>> comparison["delta_volatile_runs"]
        -1
        >>> "R2" in comparison["rules_more_frequent_in_real"]
        True
    """
    # Extract counts from ledgers (handle empty ledgers gracefully)
    mock_volatile = mock_ledger.get("volatile_runs", 0)
    mock_drifting = mock_ledger.get("drifting_runs", 0)
    mock_stable = mock_ledger.get("stable_runs", 0)
    mock_total = mock_ledger.get("total_runs", 0)
    
    real_volatile = real_ledger.get("volatile_runs", 0)
    real_drifting = real_ledger.get("drifting_runs", 0)
    real_stable = real_ledger.get("stable_runs", 0)
    real_total = real_ledger.get("total_runs", 0)
    
    # Determine most common drift status (for single-run ledgers, use that run's status)
    # For multi-run ledgers, use the status with the highest count
    if mock_total == 0:
        mock_drift_status = DriftStatus.STABLE.value
    elif mock_volatile >= mock_drifting and mock_volatile >= mock_stable:
        mock_drift_status = DriftStatus.VOLATILE.value
    elif mock_drifting >= mock_stable:
        mock_drift_status = DriftStatus.DRIFTING.value
    else:
        mock_drift_status = DriftStatus.STABLE.value
    
    if real_total == 0:
        real_drift_status = DriftStatus.STABLE.value
    elif real_volatile >= real_drifting and real_volatile >= real_stable:
        real_drift_status = DriftStatus.VOLATILE.value
    elif real_drifting >= real_stable:
        real_drift_status = DriftStatus.DRIFTING.value
    else:
        real_drift_status = DriftStatus.STABLE.value
    
    # Compute deltas
    delta_volatile_runs = real_volatile - mock_volatile
    delta_stable_runs = real_stable - mock_stable
    
    # Compare frequent rules
    mock_rules = mock_ledger.get("frequent_rules", {})
    real_rules = real_ledger.get("frequent_rules", {})
    
    # Find all unique rule IDs
    all_rule_ids = set(mock_rules.keys()) | set(real_rules.keys())
    
    rules_more_frequent_in_real = []
    rules_more_frequent_in_mock = []
    
    for rule_id in sorted(all_rule_ids):  # Deterministic sort
        mock_count = mock_rules.get(rule_id, 0)
        real_count = real_rules.get(rule_id, 0)
        
        if real_count > mock_count:
            rules_more_frequent_in_real.append(rule_id)
        elif mock_count > real_count:
            rules_more_frequent_in_mock.append(rule_id)
    
    return {
        "schema_version": "1.0.0",
        "mock_drift_status": mock_drift_status,
        "real_drift_status": real_drift_status,
        "delta_volatile_runs": delta_volatile_runs,
        "delta_stable_runs": delta_stable_runs,
        "rules_more_frequent_in_real": rules_more_frequent_in_real,
        "rules_more_frequent_in_mock": rules_more_frequent_in_mock,
    }


def build_prng_regime_timeseries(
    tiles_by_window: List[List[Dict[str, Any]]],
    window_size: int,
) -> Dict[str, Any]:
    """
    Build PRNG regime timeseries from windowed governance tiles.

    Aggregates PRNG governance tiles grouped by time windows to produce a
    timeseries showing how drift patterns evolve over the course of a
    calibration experiment.

    Args:
        tiles_by_window: List of window lists, where each inner list contains
                         PRNG governance tiles for that window.
        window_size: Number of cycles per window (for metadata).

    Returns:
        Dict with:
        - schema_version: "1.0.0"
        - window_size: Number of cycles per window
        - windows: List of window dicts, each containing:
          - window_index: 0-based window index
          - drift_status: Most common drift_status in this window
          - frequent_rules_top5: Top 5 most frequent rule IDs (sorted by count, then by ID)
          - volatile_count: Number of tiles with drift_status="VOLATILE"
          - drifting_count: Number of tiles with drift_status="DRIFTING"
          - stable_count: Number of tiles with drift_status="STABLE"

    Example:
        >>> tiles_window0 = [
        ...     {"drift_status": "STABLE", "blocking_rules": []},
        ...     {"drift_status": "DRIFTING", "blocking_rules": ["R1"]},
        ... ]
        >>> tiles_window1 = [
        ...     {"drift_status": "VOLATILE", "blocking_rules": ["R1", "R2"]},
        ... ]
        >>> ts = build_prng_regime_timeseries([tiles_window0, tiles_window1], window_size=20)
        >>> ts["window_size"]
        20
        >>> len(ts["windows"])
        2
    """
    windows = []
    
    for window_index, window_tiles in enumerate(tiles_by_window):
        if not window_tiles:
            # Empty window - use STABLE defaults
            windows.append({
                "window_index": window_index,
                "drift_status": DriftStatus.STABLE.value,
                "frequent_rules_top5": [],
                "volatile_count": 0,
                "drifting_count": 0,
                "stable_count": 0,
            })
            continue
        
        # Count drift statuses in this window
        volatile_count = 0
        drifting_count = 0
        stable_count = 0
        
        # Aggregate blocking rules across tiles in this window
        rule_counts: Dict[str, int] = {}
        
        for tile in window_tiles:
            drift_status = tile.get("drift_status", DriftStatus.STABLE.value)
            
            if drift_status == DriftStatus.VOLATILE.value:
                volatile_count += 1
            elif drift_status == DriftStatus.DRIFTING.value:
                drifting_count += 1
            else:  # STABLE
                stable_count += 1
            
            # Aggregate blocking rules
            blocking_rules = tile.get("blocking_rules", [])
            for rule_id in blocking_rules:
                if rule_id:
                    rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
        
        # Determine most common drift status
        if volatile_count >= drifting_count and volatile_count >= stable_count:
            window_drift_status = DriftStatus.VOLATILE.value
        elif drifting_count >= stable_count:
            window_drift_status = DriftStatus.DRIFTING.value
        else:
            window_drift_status = DriftStatus.STABLE.value
        
        # Get top 5 frequent rules (sorted by count descending, then by rule_id ascending)
        sorted_rules = sorted(
            rule_counts.items(),
            key=lambda x: (-x[1], x[0])  # Negative count for descending, then rule_id ascending
        )
        frequent_rules_top5 = [rule_id for rule_id, _ in sorted_rules[:5]]
        
        windows.append({
            "window_index": window_index,
            "drift_status": window_drift_status,
            "frequent_rules_top5": frequent_rules_top5,
            "volatile_count": volatile_count,
            "drifting_count": drifting_count,
            "stable_count": stable_count,
        })
    
    # Compute summary fields
    first_window_status = DriftStatus.STABLE.value
    last_window_status = DriftStatus.STABLE.value
    status_changed = False
    volatile_window_count = 0
    
    if windows:
        first_window_status = windows[0]["drift_status"]
        last_window_status = windows[-1]["drift_status"]
        status_changed = first_window_status != last_window_status
        
        # Count windows with VOLATILE status
        volatile_window_count = sum(
            1 for w in windows if w["drift_status"] == DriftStatus.VOLATILE.value
        )
    
    return {
        "schema_version": "1.0.0",
        "window_size": window_size,
        "windows": windows,
        # Summary fields (deterministic, derived only)
        "first_window_status": first_window_status,
        "last_window_status": last_window_status,
        "status_changed": status_changed,
        "volatile_window_count": volatile_window_count,
    }


def attach_prng_regime_timeseries_to_cal_exp_report(
    report: Dict[str, Any],
    prng_timeseries: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach PRNG regime timeseries to a calibration experiment report (read-only, additive).

    This is a lightweight helper for CAL-EXP report builders. It does not
    modify the input report dict, but returns a new dict with the timeseries attached.

    Canonical attachment location: report["governance"]["prng_regime_timeseries"]
    Legacy compatibility: If report["prng_regime_timeseries"] exists, it is mirrored
    into governance on write (no mutation of input; return enriched copy).

    Args:
        report: Existing calibration experiment report (read-only, not modified).
        prng_timeseries: PRNG regime timeseries from build_prng_regime_timeseries().

    Returns:
        New dict with report contents plus prng_regime_timeseries attached under
        report["governance"]["prng_regime_timeseries"] (canonical path).
        If report["prng_regime_timeseries"] exists in input, it is also preserved
        for backward compatibility.

    Example:
        >>> report = {"schema_version": "1.0.0", "summary": {...}}
        >>> ts = build_prng_regime_timeseries([tiles_window0, tiles_window1], window_size=20)
        >>> enriched = attach_prng_regime_timeseries_to_cal_exp_report(report, ts)
        >>> "governance" in enriched
        True
        >>> "prng_regime_timeseries" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = report.copy()
    
    # Ensure governance dict exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    # Attach to canonical location: governance.prng_regime_timeseries
    enriched["governance"]["prng_regime_timeseries"] = prng_timeseries
    
    # Backward compatibility: if legacy path exists in input, preserve it
    # (but don't overwrite if it's different - canonical path takes precedence)
    if "prng_regime_timeseries" in report:
        # Legacy path exists - keep it for backward compatibility
        # (canonical path is already set above)
        pass
    else:
        # No legacy path - optionally mirror to legacy location for compatibility
        # (but canonical path is preferred)
        pass
    
    return enriched


def attach_prng_drift_ledger_to_evidence(
    evidence: Dict[str, Any],
    ledger: Dict[str, Any],
    mock_ledger: Optional[Dict[str, Any]] = None,
    real_ledger: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach PRNG drift ledger to an evidence pack (read-only, additive).

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the ledger attached.

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        ledger: PRNG drift ledger from build_prng_drift_ledger().
        mock_ledger: Optional mock calibration experiment ledger for regime comparison.
        real_ledger: Optional real calibration experiment ledger for regime comparison.

    Returns:
        New dict with evidence contents plus prng_drift_ledger attached under
        evidence["governance"]["prng_drift_ledger"].
        If both mock_ledger and real_ledger are provided, also attaches
        first_light_prng_regime_comparison under the same key.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> ledger = build_prng_drift_ledger(tiles)
        >>> enriched = attach_prng_drift_ledger_to_evidence(evidence, ledger)
        >>> "prng_drift_ledger" in enriched["governance"]
        True

    P5 Usage Note:
        The ledger is meant to be compared across multiple First Light runs
        (mock vs real). A single run's PRNG tile is not sufficient to judge
        anything about P5 behavior. The ledger provides long-horizon context
        for deterministic randomness governance trends. For early P5, this is
        context only, not a gate.
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Attach PRNG drift ledger under governance.prng_drift_ledger key
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    # Create a copy of the ledger to avoid mutating the original
    ledger_copy = ledger.copy()
    
    # Optionally attach regime comparison if both mock and real ledgers provided
    if mock_ledger is not None and real_ledger is not None:
        ledger_copy["first_light_prng_regime_comparison"] = build_prng_regime_comparison(
            mock_ledger, real_ledger
        )
    
    enriched["governance"]["prng_drift_ledger"] = ledger_copy
    
    return enriched


def summarize_prng_for_evidence(
    radar: Dict[str, Any],
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize PRNG governance for Phase X Evidence Package.

    Provides a neutral forensic narrative suitable for evidence packs and
    long-horizon integrity tracking.

    Args:
        radar: PRNG drift radar from build_prng_drift_radar().
        tile: PRNG governance tile from build_prng_governance_tile().

    Returns:
        Dict with:
        - schema_version: "1.0.0"
        - drift_status: "STABLE" | "DRIFTING" | "VOLATILE"
        - rule_frequencies: Dict[str, int] (frequent violations)
        - blocking_rules: List[str]
        - forensic_narrative: str (neutral descriptive text)

    Example:
        >>> radar = build_prng_drift_radar(history)
        >>> tile = build_prng_governance_tile(history, radar=radar)
        >>> summary = summarize_prng_for_evidence(radar, tile)
        >>> summary["drift_status"]
        'STABLE'
    """
    drift_status = radar.get("drift_status", DriftStatus.STABLE.value)
    frequent_violations = radar.get("frequent_violations", {})
    blocking_rules = tile.get("blocking_rules", [])
    total_runs = radar.get("total_runs", 0)

    # Build neutral forensic narrative
    narrative_parts = []
    
    if total_runs == 0:
        narrative_parts.append("No PRNG governance history available.")
    else:
        narrative_parts.append(f"Analyzed {total_runs} PRNG governance run(s).")
        
        if drift_status == DriftStatus.STABLE.value:
            narrative_parts.append("Drift status: STABLE. No frequent violations detected.")
        elif drift_status == DriftStatus.DRIFTING.value:
            narrative_parts.append(
                f"Drift status: DRIFTING. {len(frequent_violations)} rule(s) "
                f"appeared frequently across runs."
            )
        else:  # VOLATILE
            narrative_parts.append(
                f"Drift status: VOLATILE. {len(frequent_violations)} rule(s) "
                f"appeared frequently across runs."
            )
        
        if frequent_violations:
            rule_list = ", ".join(
                f"{rule_id} ({count}x)" for rule_id, count in sorted(frequent_violations.items())
            )
            narrative_parts.append(f"Frequent violations: {rule_list}.")
        
        if blocking_rules:
            narrative_parts.append(
                f"Blocking rules detected: {', '.join(sorted(blocking_rules))}."
            )
        else:
            narrative_parts.append("No blocking rules detected.")

    forensic_narrative = " ".join(narrative_parts)

    return {
        "schema_version": "1.0.0",
        "drift_status": drift_status,
        "rule_frequencies": dict(sorted(frequent_violations.items())),  # Deterministic ordering
        "blocking_rules": sorted(blocking_rules),  # Deterministic ordering
        "forensic_narrative": forensic_narrative,
        "total_runs": total_runs,
    }




# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "GovernanceStatus",
    "ManifestStatus",
    "DriftStatus",
    # Data structures
    "PolicyRule",
    "NamespaceIssues",
    "PolicyViolation",
    "PRNGGovernanceSnapshot",
    "PolicyEvaluation",
    "GlobalHealthSummary",
    # Rules
    "PRNG_GOV_RULES",
    # Functions
    "build_prng_governance_snapshot",
    "evaluate_prng_policy",
    "summarize_prng_for_global_health",
    "run_full_prng_governance",
    # Phase IV
    "evaluate_prng_for_ci",
    "build_prng_remediation_suggestions",
    "build_prng_governance_history",
    # Drift Radar & Tile
    "build_prng_drift_radar",
    "build_prng_governance_tile",
    "attach_prng_tile_to_evidence",
    "attach_prng_governance_tile",
    "summarize_prng_for_evidence",
    "build_first_light_prng_summary",
    # Drift Ledger
    "build_prng_drift_ledger",
    "attach_prng_drift_ledger_to_evidence",
    "build_prng_regime_comparison",
    # Regime Timeseries
    "build_prng_regime_timeseries",
    "attach_prng_regime_timeseries_to_cal_exp_report",
]
