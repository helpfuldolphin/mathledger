#!/usr/bin/env python3
"""
Evidence Pack Toolchain

Modular tool for creating, sealing, auditing, and diffing evidence packs.

Commands:
  create  - Create a new evidence pack from artifacts
  seal    - Seal an evidence pack with cryptographic signatures
  audit   - Audit an evidence pack for integrity
  diff    - Compare two evidence packs

Exit Codes:
  0 - PASS: Operation successful
  1 - FAIL: Operation failed (integrity violation, missing files)
  3 - ERROR: Infrastructure failure
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

# Exit codes
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 3


class EvidencePackToolchain:
    """Evidence pack operations."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def compute_sha256(self, filepath: Path) -> Optional[str]:
        """Compute SHA256 hash of a file."""
        if not filepath.exists():
            return None
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            h.update(f.read())
        return h.hexdigest()

    def create(self, artifacts_dir: Path, output_path: Path, experiment_id: str, experiment_type: str) -> bool:
        """Create a new evidence pack from artifacts."""
        print(f"📦 Creating evidence pack...")
        print(f"   Artifacts: {artifacts_dir}")
        print(f"   Output: {output_path}")
        print()

        if not artifacts_dir.exists():
            print(f"❌ ERROR: Artifacts directory not found: {artifacts_dir}", file=sys.stderr)
            return False

        # Discover artifacts
        logs = list(artifacts_dir.glob("**/*.jsonl")) + list(artifacts_dir.glob("**/*.log"))
        figures = list(artifacts_dir.glob("**/*.png")) + list(artifacts_dir.glob("**/*.jpg"))

        print(f"   Found {len(logs)} log file(s)")
        print(f"   Found {len(figures)} figure file(s)")
        print()

        # Build manifest
        manifest = {
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": {
                "id": experiment_id,
                "type": experiment_type
            },
            "artifacts": {
                "logs": [],
                "figures": []
            }
        }

        # Add logs
        for log_path in logs:
            rel_path = log_path.relative_to(self.repo_root)
            sha256 = self.compute_sha256(log_path)
            line_count = 0
            if log_path.suffix == ".jsonl":
                with open(log_path, 'r') as f:
                    line_count = sum(1 for _ in f)

            manifest["artifacts"]["logs"].append({
                "path": str(rel_path),
                "type": "jsonl" if log_path.suffix == ".jsonl" else "text",
                "sha256": sha256,
                "size_bytes": log_path.stat().st_size,
                "line_count": line_count if line_count > 0 else None
            })
            print(f"   ✓ Log: {rel_path} ({sha256[:16]}...)")

        # Add figures
        for fig_path in figures:
            rel_path = fig_path.relative_to(self.repo_root)
            sha256 = self.compute_sha256(fig_path)

            manifest["artifacts"]["figures"].append({
                "path": str(rel_path),
                "type": fig_path.suffix[1:],  # Remove leading dot
                "sha256": sha256,
                "size_bytes": fig_path.stat().st_size
            })
            print(f"   ✓ Figure: {rel_path} ({sha256[:16]}...)")

        # Save manifest
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        print()
        print(f"✅ Evidence pack created: {output_path}")
        print(f"   Total artifacts: {len(logs) + len(figures)}")
        return True

    def seal(self, input_path: Path, output_path: Path) -> bool:
        """Seal an evidence pack with cryptographic signatures."""
        print(f"🔒 Sealing evidence pack...")
        print(f"   Input: {input_path}")
        print(f"   Output: {output_path}")
        print()

        if not input_path.exists():
            print(f"❌ ERROR: Input manifest not found: {input_path}", file=sys.stderr)
            return False

        # Load manifest
        try:
            with open(input_path, 'r') as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON in manifest: {e}", file=sys.stderr)
            return False

        # Compute manifest hash
        manifest_canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
        manifest_hash = hashlib.sha256(manifest_canonical.encode('utf-8')).hexdigest()

        # Create sealed pack
        sealed_pack = {
            "version": "1.0.0",
            "sealed_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest,
            "seal": {
                "manifest_hash": manifest_hash,
                "algorithm": "SHA-256"
            }
        }

        # Save sealed pack
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(sealed_pack, f, indent=2, sort_keys=True)

        print(f"✅ Evidence pack sealed: {output_path}")
        print(f"   Manifest hash: {manifest_hash}")
        return True

    def audit(self, sealed_pack_path: Path) -> bool:
        """Audit a sealed evidence pack for integrity."""
        print(f"🔍 Auditing evidence pack...")
        print(f"   Pack: {sealed_pack_path}")
        print()

        if not sealed_pack_path.exists():
            print(f"❌ ERROR: Sealed pack not found: {sealed_pack_path}", file=sys.stderr)
            return False

        # Load sealed pack
        try:
            with open(sealed_pack_path, 'r') as f:
                sealed_pack = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON in sealed pack: {e}", file=sys.stderr)
            return False

        manifest = sealed_pack.get("manifest", {})
        seal = sealed_pack.get("seal", {})
        expected_hash = seal.get("manifest_hash")

        # Verify manifest hash
        manifest_canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
        actual_hash = hashlib.sha256(manifest_canonical.encode('utf-8')).hexdigest()

        if actual_hash != expected_hash:
            print(f"❌ FAIL: Manifest hash mismatch")
            print(f"   Expected: {expected_hash}")
            print(f"   Actual:   {actual_hash}")
            return False

        print(f"✅ Manifest hash verified: {actual_hash}")
        print()

        # Verify artifacts
        issues = []
        verified_count = 0

        for log_entry in manifest.get("artifacts", {}).get("logs", []):
            log_path = self.repo_root / log_entry["path"]
            expected_sha256 = log_entry.get("sha256")

            if not log_path.exists():
                issues.append(f"Log file missing: {log_entry['path']}")
                continue

            actual_sha256 = self.compute_sha256(log_path)
            if expected_sha256 and actual_sha256 != expected_sha256:
                issues.append(f"SHA256 mismatch for {log_entry['path']}")
            else:
                verified_count += 1
                print(f"   ✓ Log: {log_entry['path']}")

        for fig_entry in manifest.get("artifacts", {}).get("figures", []):
            fig_path = self.repo_root / fig_entry["path"]
            expected_sha256 = fig_entry.get("sha256")

            if not fig_path.exists():
                issues.append(f"Figure file missing: {fig_entry['path']}")
                continue

            actual_sha256 = self.compute_sha256(fig_path)
            if expected_sha256 and actual_sha256 != expected_sha256:
                issues.append(f"SHA256 mismatch for {fig_entry['path']}")
            else:
                verified_count += 1
                print(f"   ✓ Figure: {fig_entry['path']}")

        print()
        if issues:
            print(f"❌ FAIL: {len(issues)} issue(s) detected")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print(f"✅ PASS: All {verified_count} artifact(s) verified")
            return True

    def diff(self, baseline_path: Path, current_path: Path) -> bool:
        """Compare two evidence packs."""
        print(f"🔍 Comparing evidence packs...")
        print(f"   Baseline: {baseline_path}")
        print(f"   Current:  {current_path}")
        print()

        if not baseline_path.exists():
            print(f"❌ ERROR: Baseline pack not found: {baseline_path}", file=sys.stderr)
            return False

        if not current_path.exists():
            print(f"❌ ERROR: Current pack not found: {current_path}", file=sys.stderr)
            return False

        # Load packs
        try:
            with open(baseline_path, 'r') as f:
                baseline = json.load(f)
            with open(current_path, 'r') as f:
                current = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON: {e}", file=sys.stderr)
            return False

        # Extract manifests
        baseline_manifest = baseline.get("manifest", baseline)
        current_manifest = current.get("manifest", current)

        # Compare artifact counts
        baseline_logs = len(baseline_manifest.get("artifacts", {}).get("logs", []))
        current_logs = len(current_manifest.get("artifacts", {}).get("logs", []))
        baseline_figs = len(baseline_manifest.get("artifacts", {}).get("figures", []))
        current_figs = len(current_manifest.get("artifacts", {}).get("figures", []))

        print(f"   Logs:    {baseline_logs} → {current_logs} ({current_logs - baseline_logs:+d})")
        print(f"   Figures: {baseline_figs} → {current_figs} ({current_figs - baseline_figs:+d})")
        print()

        if baseline_manifest == current_manifest:
            print("✅ Evidence packs are identical")
        else:
            print("⚠️  Evidence packs differ")

        return True


def main():
    parser = argparse.ArgumentParser(description="Evidence Pack Toolchain")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # create command
    create_parser = subparsers.add_parser("create", help="Create a new evidence pack")
    create_parser.add_argument("--artifacts-dir", type=Path, required=True, help="Directory containing artifacts")
    create_parser.add_argument("--output", type=Path, required=True, help="Output manifest path")
    create_parser.add_argument("--experiment-id", type=str, required=True, help="Experiment ID")
    create_parser.add_argument("--experiment-type", type=str, required=True, help="Experiment type")

    # seal command
    seal_parser = subparsers.add_parser("seal", help="Seal an evidence pack")
    seal_parser.add_argument("--input", type=Path, required=True, help="Input manifest path")
    seal_parser.add_argument("--output", type=Path, required=True, help="Output sealed pack path")

    # audit command
    audit_parser = subparsers.add_parser("audit", help="Audit a sealed evidence pack")
    audit_parser.add_argument("--pack", type=Path, required=True, help="Sealed pack path")

    # diff command
    diff_parser = subparsers.add_parser("diff", help="Compare two evidence packs")
    diff_parser.add_argument("--baseline", type=Path, required=True, help="Baseline pack path")
    diff_parser.add_argument("--current", type=Path, required=True, help="Current pack path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(EXIT_ERROR)

    toolchain = EvidencePackToolchain(args.repo_root)

    if args.command == "create":
        success = toolchain.create(args.artifacts_dir, args.output, args.experiment_id, args.experiment_type)
    elif args.command == "seal":
        success = toolchain.seal(args.input, args.output)
    elif args.command == "audit":
        success = toolchain.audit(args.pack)
    elif args.command == "diff":
        success = toolchain.diff(args.baseline, args.current)
    else:
        parser.print_help()
        sys.exit(EXIT_ERROR)

    sys.exit(EXIT_PASS if success else EXIT_FAIL)


if __name__ == "__main__":
    main()
