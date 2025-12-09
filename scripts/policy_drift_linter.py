#!/usr/bin/env python3
"""
Policy Drift Linter & Commit Guard.

This tool enforces that policy updates remain equation-consistent with the
RFL contract: every change must be attested, policy surfaces must remain
stable, and learning schedules must evolve smoothly.  It provides:

1. Linting of manifests, weights, surfaces, and delta logs.
2. Deterministic snapshots written to a JSONL ledger.
3. Drift comparisons between consecutive snapshots for commit/CI guards.

Exit codes:
    0 - Lint passed and no blocking drift detected.
    1 - Drift detected with BLOCK status (policy hash/surface change).
    2 - Lint errors (hash mismatch, malformed metadata, etc.).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

DEFAULT_POLICY_DIR = Path("artifacts/policy")
DEFAULT_LEDGER_PATH = DEFAULT_POLICY_DIR / "policy_hash_ledger.jsonl"
WEIGHT_CANDIDATES = (
    "policy.weights.bin",
    "policy.bin",
    "policy.pkl",
    "policy_weights.bin",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _detect_git_sha() -> Optional[str]:
    for env_var in ("GITHUB_SHA", "CI_COMMIT_SHA", "GIT_SHA", "GIT_COMMIT"):
        sha = os.environ.get(env_var)
        if sha:
            return sha[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode == 0:
        sha = result.stdout.strip()
        return sha[:12] if sha else None
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_delta_log(path: Path) -> Tuple[int, Optional[str], Optional[float], Optional[float]]:
    count = 0
    last_hash = None
    learning_rates: List[float] = []
    last_eta = None
    if not path.exists():
        return count, last_hash, last_eta, None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise ValueError(f"delta_log.jsonl contains invalid JSON: {exc}") from exc
            count += 1
            last_hash = entry.get("policy_hash_after")
            eta = entry.get("learning_rate")
            if isinstance(eta, (int, float)):
                last_eta = float(eta)
                learning_rates.append(last_eta)
    lr_spread = None
    if learning_rates:
        lr_spread = max(learning_rates) - min(learning_rates)
    return count, last_hash, last_eta, lr_spread


def _load_surface_signature(path: Path) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8") as handle:
        surface_data = yaml.safe_load(handle) or {}
    signature = hashlib.sha256(_canonical_json(surface_data).encode("utf-8")).hexdigest()
    return signature, surface_data


def _resolve_manifest_path(policy_dir: Path) -> Path:
    manifest = policy_dir / "policy.manifest.json"
    if manifest.exists():
        return manifest
    legacy = policy_dir / "policy.json"
    if legacy.exists():
        data = _load_json(legacy)
        policy_hash = data.get("hash")
        if policy_hash:
            candidate = policy_dir / policy_hash / "policy.manifest.json"
            if candidate.exists():
                return candidate
        return legacy
    raise FileNotFoundError(f"No policy manifest found in {policy_dir}")


def _resolve_weight_path(policy_dir: Path) -> Optional[Path]:
    for candidate in WEIGHT_CANDIDATES:
        path = policy_dir / candidate
        if path.exists():
            return path
    return None


@dataclass
class PolicyMetadata:
    policy_hash: str
    version: str
    model_type: str
    created_at: Optional[str]
    manifest_path: Path
    manifest_sha256: str
    weights_path: Optional[Path]
    weights_sha256: Optional[str]
    surface_path: Optional[Path]
    surface_signature: Optional[str]
    delta_log_path: Optional[Path]
    delta_entry_count: int
    delta_last_hash: Optional[str]
    learning_rate_spread: Optional[float]
    learning_rate_tail: Optional[float]
    notes: Optional[str] = None


def load_policy_metadata(policy_dir: Path) -> PolicyMetadata:
    manifest_path = _resolve_manifest_path(policy_dir)
    artifact_dir = manifest_path.parent if manifest_path.name != "policy.json" else policy_dir
    manifest_data = _load_json(manifest_path)
    if "policy" in manifest_data:
        policy_section = manifest_data["policy"]
    else:
        policy_section = manifest_data
    policy_hash = policy_section.get("hash") or policy_section.get("policy_hash")
    if not policy_hash:
        raise ValueError(f"policy hash missing from manifest {manifest_path}")
    manifest_sha = _sha256_path(manifest_path)
    weights_path = _resolve_weight_path(artifact_dir)
    weights_sha = _sha256_path(weights_path) if weights_path else None
    surface_path = artifact_dir / "policy.surface.yaml"
    surface_signature, _ = _load_surface_signature(surface_path) if surface_path.exists() else (None, None)
    delta_log_path = artifact_dir / "delta_log.jsonl"
    delta_entry_count, delta_last_hash, lr_tail, lr_spread = _read_delta_log(delta_log_path)
    return PolicyMetadata(
        policy_hash=policy_hash,
        version=policy_section.get("version", "unknown"),
        model_type=policy_section.get("model_type", "unknown"),
        created_at=policy_section.get("created_at"),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        weights_path=weights_path,
        weights_sha256=weights_sha,
        surface_path=surface_path if surface_path.exists() else None,
        surface_signature=surface_signature,
        delta_log_path=delta_log_path if delta_log_path.exists() else None,
        delta_entry_count=delta_entry_count,
        delta_last_hash=delta_last_hash,
        learning_rate_spread=lr_spread,
        learning_rate_tail=lr_tail,
    )


@dataclass
class PolicyLintResult:
    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.issues:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"


def lint_policy(metadata: PolicyMetadata) -> PolicyLintResult:
    issues: List[str] = []
    warnings: List[str] = []
    if not metadata.policy_hash:
        issues.append("policy hash missing from manifest metadata")
    if metadata.weights_path is None:
        warnings.append("weights artifact not found; cannot verify hash")
    elif metadata.weights_sha256 != metadata.policy_hash:
        issues.append(
            f"weights hash {metadata.weights_sha256 or 'unknown'} "
            f"does not match manifest hash {metadata.policy_hash}"
        )
    if metadata.surface_path and not metadata.surface_signature:
        warnings.append("policy.surface.yaml is empty; cannot compute signature")
    if metadata.delta_entry_count == 0:
        warnings.append("delta_log.jsonl missing or empty (no policy delta attestation)")
    if metadata.delta_last_hash and metadata.delta_last_hash != metadata.policy_hash:
        issues.append(
            f"latest delta_log hash ({metadata.delta_last_hash}) "
            f"does not match manifest hash ({metadata.policy_hash})"
        )
    if metadata.learning_rate_spread is not None and metadata.learning_rate_spread > 0.2:
        warnings.append(
            f"learning rate schedule spread {metadata.learning_rate_spread:.4f} exceeds smoothness bound 0.2"
        )
    return PolicyLintResult(passed=not issues, issues=issues, warnings=warnings)


def _snapshot_id(metadata: PolicyMetadata, recorded_at: str) -> str:
    base = f"{metadata.policy_hash}-{recorded_at}-{metadata.manifest_sha256}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


class PolicyLedger:
    def __init__(self, path: Path):
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        try:
                            self.entries.append(json.loads(stripped))
                        except json.JSONDecodeError:
                            continue

    def record_snapshot(self, metadata: PolicyMetadata, lint: PolicyLintResult, notes: Optional[str]) -> Dict[str, Any]:
        recorded_at = _utc_now()
        entry = {
            "snapshot_id": _snapshot_id(metadata, recorded_at),
            "recorded_at": recorded_at,
            "git_sha": _detect_git_sha(),
            "policy_hash": metadata.policy_hash,
            "policy_version": metadata.version,
            "model_type": metadata.model_type,
            "manifest_path": str(metadata.manifest_path),
            "manifest_sha256": metadata.manifest_sha256,
            "weights_path": str(metadata.weights_path) if metadata.weights_path else None,
            "weights_sha256": metadata.weights_sha256,
            "surface_signature": metadata.surface_signature,
            "delta_entry_count": metadata.delta_entry_count,
            "delta_last_hash": metadata.delta_last_hash,
            "learning_rate_spread": metadata.learning_rate_spread,
            "learning_rate_tail": metadata.learning_rate_tail,
            "lint_issues": list(lint.issues),
            "lint_warnings": list(lint.warnings),
            "notes": notes or metadata.notes,
        }
        _ensure_dir(self.path)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(entry))
            handle.write("\n")
        self.entries.append(entry)
        return entry

    def _resolve_ref(self, ref: str) -> Optional[Dict[str, Any]]:
        if not self.entries:
            return None
        try:
            index = int(ref)
        except ValueError:
            ref_lower = ref.lower()
            for entry in reversed(self.entries):
                if entry["policy_hash"].lower().startswith(ref_lower) or entry["snapshot_id"].lower().startswith(ref_lower):
                    return entry
            return None
        if index < 0:
            index = len(self.entries) + index
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def compare(self, ref_old: str, ref_new: str) -> Optional["PolicyDriftReport"]:
        old_entry = self._resolve_ref(ref_old)
        new_entry = self._resolve_ref(ref_new)
        if old_entry is None or new_entry is None:
            return None
        return PolicyDriftReport.from_entries(old_entry, new_entry)


@dataclass
class PolicyDriftReport:
    status: str
    reasons: List[str]
    changed_fields: List[str]
    drift_score: float

    @classmethod
    def from_entries(cls, old: Dict[str, Any], new: Dict[str, Any]) -> "PolicyDriftReport":
        reasons: List[str] = []
        changed: List[str] = []
        status = "STABLE"
        score = 0.0

        if old["policy_hash"] != new["policy_hash"]:
            changed.append("policy_hash")
            reasons.append(
                f"policy hash changed {old['policy_hash'][:12]} -> {new['policy_hash'][:12]}"
            )
            status = "BLOCK"
            score += 1.0
            if new.get("delta_last_hash") != new["policy_hash"]:
                reasons.append("delta_log does not attest the new hash")
            else:
                reasons.append("delta_log attests new hash; manual review still required")

        if old.get("surface_signature") != new.get("surface_signature"):
            changed.append("policy_surface")
            reasons.append("policy surface signature changed")
            status = "BLOCK"
            score += 0.75

        if old.get("model_type") != new.get("model_type"):
            changed.append("model_type")
            reasons.append(f"model_type changed {old.get('model_type')} -> {new.get('model_type')}")
            status = "BLOCK"
            score += 0.5

        old_spread = old.get("learning_rate_spread")
        new_spread = new.get("learning_rate_spread")
        if (
            new_spread is not None
            and old_spread is not None
            and new_spread - old_spread > 0.1
        ):
            changed.append("learning_rate_spread")
            reasons.append(
                f"learning rate spread increased ({old_spread:.4f} -> {new_spread:.4f})"
            )
            if status != "BLOCK":
                status = "WARN"
            score += 0.2

        if not changed:
            reasons.append("no policy drift detected")
            status = "STABLE"

        return cls(status=status, reasons=reasons, changed_fields=changed, drift_score=round(score, 3))


def run_lint(policy_dir: Path, quiet: bool = False) -> Tuple[PolicyMetadata, PolicyLintResult]:
    metadata = load_policy_metadata(policy_dir)
    lint_result = lint_policy(metadata)
    if not quiet:
        print(f"[policy-lint] Manifest: {metadata.manifest_path}")
        print(f"[policy-lint] Policy hash: {metadata.policy_hash}")
        if lint_result.issues:
            print("[policy-lint] Issues:")
            for issue in lint_result.issues:
                print(f"  - {issue}")
        if lint_result.warnings:
            print("[policy-lint] Warnings:")
            for warning in lint_result.warnings:
                print(f"  - {warning}")
        if not lint_result.issues and not lint_result.warnings:
            print("[policy-lint] PASS: metadata, weights, delta log, and surface consistent")
    return metadata, lint_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Policy Drift Linter & Commit Guard")
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR, help="Directory containing policy artifacts")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH, help="Path to policy drift ledger JSONL")
    parser.add_argument("--snapshot", action="store_true", help="Record a new snapshot in the policy ledger")
    parser.add_argument("--lint", action="store_true", help="Run linter checks against the policy directory")
    parser.add_argument("--drift-check", action="store_true", help="Compare the last two snapshots for drift")
    parser.add_argument("--from", dest="from_ref", default="-2", help="Reference for older snapshot when running --drift-check")
    parser.add_argument("--to", dest="to_ref", default="-1", help="Reference for new snapshot when running --drift-check")
    parser.add_argument("--notes", type=str, default=None, help="Optional note appended to ledger snapshot entries")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--quiet", action="store_true", help="Suppress informational output")
    args = parser.parse_args()

    if not any([args.snapshot, args.lint, args.drift_check]):
        args.lint = True

    exit_code = 0
    metadata: Optional[PolicyMetadata] = None
    lint_result: Optional[PolicyLintResult] = None

    try:
        if args.lint or args.snapshot:
            metadata, lint_result = run_lint(args.policy_dir, quiet=args.quiet)
            if lint_result.issues or (args.strict and lint_result.warnings):
                return 2

        ledger: Optional[PolicyLedger] = None
        if args.snapshot:
            ledger = PolicyLedger(args.ledger)
            entry = ledger.record_snapshot(metadata, lint_result or PolicyLintResult(passed=True), args.notes)
            if not args.quiet:
                print(f"[policy-ledger] Snapshot recorded: {entry['snapshot_id']} ({entry['policy_hash'][:12]})")

        if args.drift_check:
            ledger = ledger or PolicyLedger(args.ledger)
            report = ledger.compare(args.from_ref, args.to_ref)
            if report is None:
                if not args.quiet:
                    print("[policy-drift] Insufficient snapshots to compare (need at least two entries).")
                return exit_code
            if not args.quiet:
                print(f"[policy-drift] Status: {report.status}")
                for reason in report.reasons:
                    print(f"  - {reason}")
                print(f"[policy-drift] Drift score: {report.drift_score}")
            if report.status == "BLOCK":
                exit_code = max(exit_code, 1)
            elif report.status == "WARN" and args.strict:
                exit_code = max(exit_code, 1)

    except Exception as exc:
        if not args.quiet:
            print(f"[policy-drift] ERROR: {exc}", file=sys.stderr)
        return 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
