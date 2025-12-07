#!/usr/bin/env python3
"""
HT Triangle Drift Radar

Verifies the H_t = SHA256(R_t || U_t) invariant for dual-attestation seals.

Exit Codes:
  0 - PASS: No drift detected, all H_t values are valid
  1 - FAIL: Critical drift detected (H_t mismatch, invalid hash format, missing root)
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


class HTTriangleDriftRadar:
    """HT triangle invariant verification engine."""

    def __init__(self, baseline_path: Path, current_path: Path, output_dir: Path):
        self.baseline_path = baseline_path
        self.current_path = current_path
        self.output_dir = output_dir
        self.drift_report = {
            "version": "1.0.0",
            "radar": "ht_triangle",
            "status": "PASS",
            "drifts": [],
            "summary": {
                "critical": 0,
                "warning": 0,
                "info": 0
            }
        }

    def load_snapshot(self, path: Path) -> Dict[str, Any]:
        """Load and parse an attestation snapshot."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: Snapshot not found: {path}", file=sys.stderr)
            sys.exit(EXIT_ERROR)
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON in {path}: {e}", file=sys.stderr)
            sys.exit(EXIT_ERROR)

    def verify_ht_invariant(self, attestation: Dict) -> Dict[str, Any]:
        """Verify H_t = SHA256(R_t || U_t) for a single attestation."""
        att_id = attestation.get("id", "unknown")
        H_t = attestation.get("H_t", "")
        R_t = attestation.get("R_t", "")
        U_t = attestation.get("U_t", "")

        # Check for missing roots
        if not R_t:
            return {
                "type": "missing_root",
                "severity": "CRITICAL",
                "attestation_id": att_id,
                "missing_field": "R_t",
                "message": f"Attestation '{att_id}' is missing R_t"
            }

        if not U_t:
            return {
                "type": "missing_root",
                "severity": "CRITICAL",
                "attestation_id": att_id,
                "missing_field": "U_t",
                "message": f"Attestation '{att_id}' is missing U_t"
            }

        if not H_t:
            return {
                "type": "missing_root",
                "severity": "CRITICAL",
                "attestation_id": att_id,
                "missing_field": "H_t",
                "message": f"Attestation '{att_id}' is missing H_t"
            }

        # Validate hash format (64-character hex)
        if not self._is_valid_hash(H_t):
            return {
                "type": "invalid_hash_format",
                "severity": "CRITICAL",
                "attestation_id": att_id,
                "field": "H_t",
                "value": H_t,
                "message": f"Attestation '{att_id}' has invalid H_t format (expected 64-char hex)"
            }

        if not self._is_valid_hash(R_t):
            return {
                "type": "invalid_hash_format",
                "severity": "CRITICAL",
                "attestation_id": att_id,
                "field": "R_t",
                "value": R_t,
                "message": f"Attestation '{att_id}' has invalid R_t format (expected 64-char hex)"
            }

        if not self._is_valid_hash(U_t):
            return {
                "type": "invalid_hash_format",
                "severity": "CRITICAL",
                "attestation_id": att_id,
                "field": "U_t",
                "value": U_t,
                "message": f"Attestation '{att_id}' has invalid U_t format (expected 64-char hex)"
            }

        # Compute expected H_t
        composite_data = f"{R_t}{U_t}".encode('ascii')
        expected_H_t = hashlib.sha256(composite_data).hexdigest()

        # Verify invariant
        if H_t != expected_H_t:
            return {
                "type": "ht_mismatch",
                "severity": "CRITICAL",
                "attestation_id": att_id,
                "declared_H_t": H_t,
                "expected_H_t": expected_H_t,
                "R_t": R_t,
                "U_t": U_t,
                "message": f"Attestation '{att_id}' violates H_t = SHA256(R_t || U_t) invariant"
            }

        # All checks passed
        return None

    def _is_valid_hash(self, value: str) -> bool:
        """Check if a value is a valid SHA-256 hash (64-character hex)."""
        if not isinstance(value, str):
            return False
        if len(value) != 64:
            return False
        try:
            int(value, 16)
            return True
        except ValueError:
            return False

    def detect_attestation_drift(self, baseline: Dict, current: Dict) -> List[Dict]:
        """Detect changes to attestations."""
        drifts = []
        baseline_attestations = {a["id"]: a for a in baseline.get("attestations", [])}
        current_attestations = {a["id"]: a for a in current.get("attestations", [])}

        # Check for removed attestations
        for att_id in baseline_attestations:
            if att_id not in current_attestations:
                drifts.append({
                    "type": "attestation_removed",
                    "severity": "CRITICAL",
                    "attestation_id": att_id,
                    "message": f"Attestation '{att_id}' was removed"
                })

        # Verify H_t invariant for all current attestations
        for att_id, attestation in current_attestations.items():
            drift = self.verify_ht_invariant(attestation)
            if drift:
                drifts.append(drift)

            # Check if attestation changed from baseline
            if att_id in baseline_attestations:
                baseline_att = baseline_attestations[att_id]
                if baseline_att.get("H_t") != attestation.get("H_t"):
                    drifts.append({
                        "type": "ht_changed",
                        "severity": "CRITICAL",
                        "attestation_id": att_id,
                        "baseline_H_t": baseline_att.get("H_t"),
                        "current_H_t": attestation.get("H_t"),
                        "message": f"Attestation '{att_id}' H_t changed"
                    })

        # Check for new attestations (INFO only)
        for att_id in current_attestations:
            if att_id not in baseline_attestations:
                drifts.append({
                    "type": "attestation_added",
                    "severity": "INFO",
                    "attestation_id": att_id,
                    "message": f"New attestation '{att_id}' added"
                })

        return drifts

    def run(self) -> int:
        """Execute drift detection."""
        print("🔍 HT Triangle Drift Radar - Verifying H_t = SHA256(R_t || U_t) invariant...")
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
        attestation_drifts = self.detect_attestation_drift(baseline, current)
        self.drift_report["drifts"] = attestation_drifts

        # Classify severity
        for drift in attestation_drifts:
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
            print("✅ PASS: All H_t invariants verified")
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
        report_path = self.output_dir / "ht_triangle_drift_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.drift_report, f, indent=2, sort_keys=True)
        print()
        print(f"📄 Report saved: {report_path}")

    def _save_summary(self):
        """Save human-readable drift summary."""
        summary_path = self.output_dir / "ht_triangle_drift_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# HT Triangle Drift Report\n\n")
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
    parser = argparse.ArgumentParser(description="HT Triangle Drift Radar")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline snapshot")
    parser.add_argument("--current", type=Path, required=True, help="Path to current snapshot")
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift"), help="Output directory")
    args = parser.parse_args()

    radar = HTTriangleDriftRadar(args.baseline, args.current, args.output)
    exit_code = radar.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
