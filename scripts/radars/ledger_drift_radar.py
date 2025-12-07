#!/usr/bin/env python3
"""
Ledger Drift Radar

Detects unintended changes to ledger state, block hashes, and Merkle roots
to ensure cryptographic integrity and deterministic state.

Exit Codes:
  0 - PASS: No drift detected
  1 - FAIL: Critical drift detected (broken chain, state hash mismatch, Merkle root changed)
  2 - WARN: Non-critical drift detected
  3 - ERROR: Infrastructure failure (missing files, invalid JSON)
  4 - SKIP: No baseline snapshot available
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Exit codes
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_WARN = 2
EXIT_ERROR = 3
EXIT_SKIP = 4


class LedgerDriftRadar:
    """Ledger state drift detection engine."""

    def __init__(self, baseline_path: Path, current_path: Path, output_dir: Path):
        self.baseline_path = baseline_path
        self.current_path = current_path
        self.output_dir = output_dir
        self.drift_report = {
            "version": "1.0.0",
            "radar": "ledger",
            "status": "PASS",
            "drifts": [],
            "summary": {
                "critical": 0,
                "warning": 0,
                "info": 0
            }
        }

    def load_snapshot(self, path: Path) -> Dict[str, Any]:
        """Load and parse a ledger snapshot."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: Snapshot not found: {path}", file=sys.stderr)
            sys.exit(EXIT_ERROR)
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON in {path}: {e}", file=sys.stderr)
            sys.exit(EXIT_ERROR)

    def detect_chain_drift(self, baseline: Dict, current: Dict) -> List[Dict]:
        """Detect changes to blockchain state."""
        drifts = []

        # Check chain ID
        if baseline.get("chain_id") != current.get("chain_id"):
            drifts.append({
                "type": "chain_id_changed",
                "severity": "CRITICAL",
                "baseline_chain_id": baseline.get("chain_id"),
                "current_chain_id": current.get("chain_id"),
                "message": "Chain ID changed - this indicates a different ledger"
            })

        # Check height
        baseline_height = baseline.get("height", 0)
        current_height = current.get("height", 0)
        if baseline_height != current_height:
            drifts.append({
                "type": "height_mismatch",
                "severity": "CRITICAL",
                "baseline_height": baseline_height,
                "current_height": current_height,
                "message": f"Ledger height mismatch: {baseline_height} → {current_height}"
            })

        # Check last block hash
        if baseline.get("last_block_hash") != current.get("last_block_hash"):
            drifts.append({
                "type": "last_block_hash_mismatch",
                "severity": "CRITICAL",
                "baseline_hash": baseline.get("last_block_hash"),
                "current_hash": current.get("last_block_hash"),
                "message": "Last block hash mismatch - state divergence detected"
            })

        return drifts

    def detect_block_drift(self, baseline: Dict, current: Dict) -> List[Dict]:
        """Detect changes to individual blocks."""
        drifts = []
        baseline_blocks = {b["height"]: b for b in baseline.get("blocks", [])}
        current_blocks = {b["height"]: b for b in current.get("blocks", [])}

        for height, baseline_block in baseline_blocks.items():
            if height not in current_blocks:
                drifts.append({
                    "type": "block_missing",
                    "severity": "CRITICAL",
                    "height": height,
                    "message": f"Block at height {height} is missing"
                })
                continue

            current_block = current_blocks[height]

            # Check block hash
            if baseline_block.get("hash") != current_block.get("hash"):
                drifts.append({
                    "type": "block_hash_changed",
                    "severity": "CRITICAL",
                    "height": height,
                    "baseline_hash": baseline_block.get("hash"),
                    "current_hash": current_block.get("hash"),
                    "message": f"Block hash changed at height {height}"
                })

            # Check Merkle root
            if baseline_block.get("merkle_root") != current_block.get("merkle_root"):
                drifts.append({
                    "type": "merkle_root_changed",
                    "severity": "CRITICAL",
                    "height": height,
                    "baseline_root": baseline_block.get("merkle_root"),
                    "current_root": current_block.get("merkle_root"),
                    "message": f"Merkle root changed at height {height}"
                })

            # Check prev_hash
            if baseline_block.get("prev_hash") != current_block.get("prev_hash"):
                drifts.append({
                    "type": "prev_hash_changed",
                    "severity": "CRITICAL",
                    "height": height,
                    "baseline_prev": baseline_block.get("prev_hash"),
                    "current_prev": current_block.get("prev_hash"),
                    "message": f"Previous hash changed at height {height}"
                })

        # Verify chain integrity (prev_hash linkage)
        sorted_blocks = sorted(current_blocks.values(), key=lambda b: b["height"])
        for i in range(1, len(sorted_blocks)):
            prev_block = sorted_blocks[i - 1]
            curr_block = sorted_blocks[i]
            if curr_block.get("prev_hash") != prev_block.get("hash"):
                drifts.append({
                    "type": "broken_chain",
                    "severity": "CRITICAL",
                    "height": curr_block["height"],
                    "expected_prev": prev_block.get("hash"),
                    "actual_prev": curr_block.get("prev_hash"),
                    "message": f"Broken chain at height {curr_block['height']}: prev_hash does not match previous block"
                })

        return drifts

    def run(self) -> int:
        """Execute drift detection."""
        print("🔍 Ledger Drift Radar - Scanning for state changes...")
        print(f"   Baseline: {self.baseline_path}")
        print(f"   Current:  {self.current_path}")
        print()

        # Check if baseline exists
        if not self.baseline_path.exists():
            print("⏭️  SKIP: No baseline snapshot available")
            print("   This is expected for the first run.")
            self.drift_report["status"] = "SKIP"
            self._save_report()
            return EXIT_SKIP

        # Load snapshots
        baseline = self.load_snapshot(self.baseline_path)
        current = self.load_snapshot(self.current_path)

        # Detect drifts
        chain_drifts = self.detect_chain_drift(baseline, current)
        block_drifts = self.detect_block_drift(baseline, current)

        all_drifts = chain_drifts + block_drifts
        self.drift_report["drifts"] = all_drifts

        # Classify severity
        for drift in all_drifts:
            severity = drift.get("severity", "INFO")
            if severity == "CRITICAL":
                self.drift_report["summary"]["critical"] += 1
            elif severity == "WARNING":
                self.drift_report["summary"]["warning"] += 1
            else:
                self.drift_report["summary"]["info"] += 1

        # Determine exit code
        if self.drift_report["summary"]["critical"] > 0:
            self.drift_report["status"] = "FAIL"
            exit_code = EXIT_FAIL
        elif self.drift_report["summary"]["warning"] > 0:
            self.drift_report["status"] = "WARN"
            exit_code = EXIT_WARN
        else:
            self.drift_report["status"] = "PASS"
            exit_code = EXIT_PASS

        # Print results
        self._print_results()

        # Save artifacts
        self._save_report()
        self._save_summary()

        return exit_code

    def _print_results(self):
        """Print drift detection results."""
        status = self.drift_report["status"]
        summary = self.drift_report["summary"]

        if status == "PASS":
            print("✅ PASS: No ledger drift detected")
        elif status == "WARN":
            print(f"⚠️  WARN: {summary['warning']} non-critical drift(s) detected")
        elif status == "FAIL":
            print(f"❌ FAIL: {summary['critical']} critical drift(s) detected")

        print()
        print(f"   Critical: {summary['critical']}")
        print(f"   Warning:  {summary['warning']}")
        print(f"   Info:     {summary['info']}")
        print()

        # Print details
        for drift in self.drift_report["drifts"]:
            severity_icon = {
                "CRITICAL": "❌",
                "WARNING": "⚠️",
                "INFO": "ℹ️"
            }.get(drift["severity"], "•")
            print(f"{severity_icon} [{drift['severity']}] {drift['message']}")

    def _save_report(self):
        """Save machine-readable drift report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "ledger_drift_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.drift_report, f, indent=2, sort_keys=True)
        print()
        print(f"📄 Report saved: {report_path}")

    def _save_summary(self):
        """Save human-readable drift summary."""
        summary_path = self.output_dir / "ledger_drift_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Ledger Drift Report\n\n")
            f.write(f"**Status**: {self.drift_report['status']}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- **Critical**: {self.drift_report['summary']['critical']}\n")
            f.write(f"- **Warning**: {self.drift_report['summary']['warning']}\n")
            f.write(f"- **Info**: {self.drift_report['summary']['info']}\n\n")
            f.write("## Detected Drifts\n\n")
            for drift in self.drift_report["drifts"]:
                f.write(f"### [{drift['severity']}] {drift['type']}\n\n")
                f.write(f"{drift['message']}\n\n")
                for key, value in drift.items():
                    if key not in ["type", "severity", "message"]:
                        f.write(f"- **{key}**: `{value}`\n")
                f.write("\n")
        print(f"📄 Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Ledger Drift Radar")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline snapshot")
    parser.add_argument("--current", type=Path, required=True, help="Path to current snapshot")
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift"), help="Output directory")
    args = parser.parse_args()

    radar = LedgerDriftRadar(args.baseline, args.current, args.output)
    exit_code = radar.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
