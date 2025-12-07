#!/usr/bin/env python3
"""
Telemetry Drift Radar

Detects unintended changes to telemetry event schemas, field types,
and required fields.

Exit Codes:
  0 - PASS: No drift detected
  1 - FAIL: Critical drift detected (field type change, required field removed)
  2 - WARN: Non-critical drift detected (description change)
  3 - ERROR: Infrastructure failure (missing files, invalid JSON)
  4 - SKIP: No baseline snapshot available
"""

import argparse
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


class TelemetryDriftRadar:
    """Telemetry schema drift detection engine."""

    def __init__(self, baseline_path: Path, current_path: Path, output_dir: Path):
        self.baseline_path = baseline_path
        self.current_path = current_path
        self.output_dir = output_dir
        self.drift_report = {
            "version": "1.0.0",
            "radar": "telemetry",
            "status": "PASS",
            "drifts": [],
            "summary": {
                "critical": 0,
                "warning": 0,
                "info": 0
            }
        }

    def load_snapshot(self, path: Path) -> Dict[str, Any]:
        """Load and parse a telemetry schema snapshot."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: Snapshot not found: {path}", file=sys.stderr)
            sys.exit(EXIT_ERROR)
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON in {path}: {e}", file=sys.stderr)
            sys.exit(EXIT_ERROR)

    def detect_event_drift(self, baseline: Dict, current: Dict) -> List[Dict]:
        """Detect changes to event schemas."""
        drifts = []
        baseline_events = baseline.get("events", {})
        current_events = current.get("events", {})

        # Detect removed events
        for event_name in baseline_events:
            if event_name not in current_events:
                drifts.append({
                    "type": "event_removed",
                    "severity": "CRITICAL",
                    "event": event_name,
                    "message": f"Event '{event_name}' was removed"
                })

        # Detect schema changes
        for event_name, baseline_event in baseline_events.items():
            if event_name not in current_events:
                continue

            current_event = current_events[event_name]

            # Check description change
            if baseline_event.get("description") != current_event.get("description"):
                drifts.append({
                    "type": "description_changed",
                    "severity": "WARNING",
                    "event": event_name,
                    "baseline_description": baseline_event.get("description"),
                    "current_description": current_event.get("description"),
                    "message": f"Event '{event_name}' description changed"
                })

            # Check schema changes
            baseline_schema = baseline_event.get("schema", {})
            current_schema = current_event.get("schema", {})

            schema_drifts = self._detect_schema_drift(event_name, baseline_schema, current_schema)
            drifts.extend(schema_drifts)

        # Detect new events (INFO only)
        for event_name in current_events:
            if event_name not in baseline_events:
                drifts.append({
                    "type": "event_added",
                    "severity": "INFO",
                    "event": event_name,
                    "message": f"New event '{event_name}' added"
                })

        return drifts

    def _detect_schema_drift(self, event_name: str, baseline_schema: Dict, current_schema: Dict) -> List[Dict]:
        """Detect changes to a single event schema."""
        drifts = []
        baseline_props = baseline_schema.get("properties", {})
        current_props = current_schema.get("properties", {})
        baseline_required = set(baseline_schema.get("required", []))
        current_required = set(current_schema.get("required", []))

        # Detect removed required fields
        for field in baseline_required:
            if field not in current_required:
                drifts.append({
                    "type": "required_field_removed",
                    "severity": "CRITICAL",
                    "event": event_name,
                    "field": field,
                    "message": f"Required field '{field}' removed from event '{event_name}'"
                })

        # Detect field type changes
        for field, baseline_def in baseline_props.items():
            if field not in current_props:
                if field in baseline_required:
                    drifts.append({
                        "type": "field_removed",
                        "severity": "CRITICAL",
                        "event": event_name,
                        "field": field,
                        "message": f"Field '{field}' removed from event '{event_name}'"
                    })
                continue

            current_def = current_props[field]
            baseline_type = baseline_def.get("type")
            current_type = current_def.get("type")

            if baseline_type != current_type:
                drifts.append({
                    "type": "field_type_changed",
                    "severity": "CRITICAL",
                    "event": event_name,
                    "field": field,
                    "baseline_type": baseline_type,
                    "current_type": current_type,
                    "message": f"Field '{field}' type changed in event '{event_name}': {baseline_type} → {current_type}"
                })

        # Detect new optional fields (INFO only)
        for field in current_props:
            if field not in baseline_props:
                severity = "INFO" if field not in current_required else "WARNING"
                drifts.append({
                    "type": "field_added",
                    "severity": severity,
                    "event": event_name,
                    "field": field,
                    "required": field in current_required,
                    "message": f"Field '{field}' added to event '{event_name}' (required={field in current_required})"
                })

        return drifts

    def run(self) -> int:
        """Execute drift detection."""
        print("🔍 Telemetry Drift Radar - Scanning for schema changes...")
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
        event_drifts = self.detect_event_drift(baseline, current)
        self.drift_report["drifts"] = event_drifts

        # Classify severity
        for drift in event_drifts:
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
            print("✅ PASS: No telemetry drift detected")
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
        report_path = self.output_dir / "telemetry_drift_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.drift_report, f, indent=2, sort_keys=True)
        print()
        print(f"📄 Report saved: {report_path}")

    def _save_summary(self):
        """Save human-readable drift summary."""
        summary_path = self.output_dir / "telemetry_drift_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Telemetry Drift Report\n\n")
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
    parser = argparse.ArgumentParser(description="Telemetry Drift Radar")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline snapshot")
    parser.add_argument("--current", type=Path, required=True, help="Path to current snapshot")
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift"), help="Output directory")
    args = parser.parse_args()

    radar = TelemetryDriftRadar(args.baseline, args.current, args.output)
    exit_code = radar.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
