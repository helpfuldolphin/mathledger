#!/usr/bin/env python3
"""
Curriculum Drift Radar

Detects unintended changes to curriculum structure, problem definitions,
difficulty scores, and topic taxonomies.

Exit Codes:
  0 - PASS: No drift detected
  1 - FAIL: Critical drift detected (schema violation, taxonomy change, content hash change)
  2 - WARN: Non-critical drift detected (difficulty score shift >10%)
  3 - ERROR: Infrastructure failure (missing files, invalid JSON)
  4 - SKIP: No baseline snapshot available
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Exit codes
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_WARN = 2
EXIT_ERROR = 3
EXIT_SKIP = 4


class CurriculumDriftRadar:
    """Curriculum drift detection engine."""

    def __init__(self, baseline_path: Path, current_path: Path, output_dir: Path):
        self.baseline_path = baseline_path
        self.current_path = current_path
        self.output_dir = output_dir
        self.drift_report = {
            "version": "1.0.0",
            "radar": "curriculum",
            "status": "PASS",
            "drifts": [],
            "summary": {
                "critical": 0,
                "warning": 0,
                "info": 0
            }
        }

    def load_snapshot(self, path: Path) -> Dict[str, Any]:
        """Load and parse a curriculum snapshot."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: Snapshot not found: {path}", file=sys.stderr)
            sys.exit(EXIT_ERROR)
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON in {path}: {e}", file=sys.stderr)
            sys.exit(EXIT_ERROR)

    def detect_taxonomy_drift(self, baseline: Dict, current: Dict) -> List[Dict]:
        """Detect changes to topic taxonomy."""
        drifts = []
        baseline_tax = baseline.get("topic_taxonomy", {})
        current_tax = current.get("topic_taxonomy", {})

        # Detect removed topics
        for topic in baseline_tax:
            if topic not in current_tax:
                drifts.append({
                    "type": "taxonomy_removed",
                    "severity": "CRITICAL",
                    "topic": topic,
                    "message": f"Topic '{topic}' was removed from taxonomy"
                })

        # Detect renamed topics
        for topic in current_tax:
            if topic not in baseline_tax:
                drifts.append({
                    "type": "taxonomy_added",
                    "severity": "CRITICAL",
                    "topic": topic,
                    "message": f"Topic '{topic}' was added to taxonomy (possible rename)"
                })

        return drifts

    def detect_problem_drift(self, baseline: Dict, current: Dict) -> List[Dict]:
        """Detect changes to problem definitions."""
        drifts = []
        baseline_problems = {p["id"]: p for p in baseline.get("problems", [])}
        current_problems = {p["id"]: p for p in current.get("problems", [])}

        for problem_id, baseline_prob in baseline_problems.items():
            if problem_id not in current_problems:
                drifts.append({
                    "type": "problem_removed",
                    "severity": "CRITICAL",
                    "problem_id": problem_id,
                    "message": f"Problem '{problem_id}' was removed"
                })
                continue

            current_prob = current_problems[problem_id]

            # Check content hash
            if baseline_prob.get("content_hash") != current_prob.get("content_hash"):
                drifts.append({
                    "type": "content_changed",
                    "severity": "CRITICAL",
                    "problem_id": problem_id,
                    "baseline_hash": baseline_prob.get("content_hash"),
                    "current_hash": current_prob.get("content_hash"),
                    "message": f"Problem '{problem_id}' content changed"
                })

            # Check difficulty score shift
            baseline_diff = baseline_prob.get("difficulty_score", 0)
            current_diff = current_prob.get("difficulty_score", 0)
            if baseline_diff > 0:
                pct_change = abs(current_diff - baseline_diff) / baseline_diff
                if pct_change > 0.10:  # >10% change
                    drifts.append({
                        "type": "difficulty_shift",
                        "severity": "WARNING",
                        "problem_id": problem_id,
                        "baseline_difficulty": baseline_diff,
                        "current_difficulty": current_diff,
                        "percent_change": round(pct_change * 100, 2),
                        "message": f"Problem '{problem_id}' difficulty shifted by {round(pct_change * 100, 1)}%"
                    })

            # Check topic change
            if baseline_prob.get("topic") != current_prob.get("topic"):
                drifts.append({
                    "type": "topic_changed",
                    "severity": "CRITICAL",
                    "problem_id": problem_id,
                    "baseline_topic": baseline_prob.get("topic"),
                    "current_topic": current_prob.get("topic"),
                    "message": f"Problem '{problem_id}' topic changed"
                })

        # Detect new problems (INFO only)
        for problem_id in current_problems:
            if problem_id not in baseline_problems:
                drifts.append({
                    "type": "problem_added",
                    "severity": "INFO",
                    "problem_id": problem_id,
                    "message": f"New problem '{problem_id}' added"
                })

        return drifts

    def run(self) -> int:
        """Execute drift detection."""
        print("🔍 Curriculum Drift Radar - Scanning for changes...")
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
        taxonomy_drifts = self.detect_taxonomy_drift(baseline, current)
        problem_drifts = self.detect_problem_drift(baseline, current)

        all_drifts = taxonomy_drifts + problem_drifts
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
            print("✅ PASS: No curriculum drift detected")
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
        report_path = self.output_dir / "curriculum_drift_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.drift_report, f, indent=2, sort_keys=True)
        print()
        print(f"📄 Report saved: {report_path}")

    def _save_summary(self):
        """Save human-readable drift summary."""
        summary_path = self.output_dir / "curriculum_drift_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Curriculum Drift Report\n\n")
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
    parser = argparse.ArgumentParser(description="Curriculum Drift Radar")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline snapshot")
    parser.add_argument("--current", type=Path, required=True, help="Path to current snapshot")
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift"), help="Output directory")
    args = parser.parse_args()

    radar = CurriculumDriftRadar(args.baseline, args.current, args.output)
    exit_code = radar.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
