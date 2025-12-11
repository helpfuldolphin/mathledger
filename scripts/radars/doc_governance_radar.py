#!/usr/bin/env python3
"""
Doc Governance Radar

Detects premature uplift claims, TDA enforcement claims, and Phase X language violations
in documentation. Guards the "First Light" narrative and ensures Phase boundaries are
properly maintained.

Exit Codes:
  0 - PASS: No governance violations detected
  1 - FAIL: Critical violations detected (premature uplift claims, incorrect phase language)
  2 - WARN: Non-critical issues detected
  3 - ERROR: Infrastructure failure (missing files, read errors)
  4 - SKIP: No documents to check
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Exit codes
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_WARN = 2
EXIT_ERROR = 3
EXIT_SKIP = 4


class DocGovernanceRadar:
    """Documentation governance enforcement engine."""

    def __init__(self, repo_root: Path, output_dir: Path):
        self.repo_root = repo_root
        self.output_dir = output_dir
        self.drift_report = {
            "version": "1.0.0",
            "radar": "doc_governance",
            "status": "PASS",
            "violations": [],
            "summary": {
                "critical": 0,
                "warning": 0,
                "info": 0
            }
        }
        
        # Phase X watchlist - documents that require special scrutiny
        self.phase_x_watchlist = [
            "docs/system_law/Phase_X_Prelaunch_Review.md",
            "docs/system_law/Phase_X_Divergence_Metric.md",
            "docs/system_law/Phase_X_P3P4_TODO.md",
            "docs/CORTEX_INTEGRATION.md",
            "docs/TDA_MODES.md",
        ]
        
        # Core watchlist - always checked
        self.core_watchlist = [
            "README.md",
            "docs/PHASE2_RFL_UPLIFT_PLAN.md",
            "docs/RFL_PHASE_I_TRUTH_SOURCE.md",
        ]

    def load_document(self, path: Path) -> str:
        """Load a document's content."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None  # Document may not exist yet, not an error
        except Exception as e:
            print(f"⚠️  WARNING: Could not read {path}: {e}", file=sys.stderr)
            return None

    def detect_premature_uplift_claims(self, content: str, filepath: str) -> List[Dict]:
        """Detect uplift claims without proper evidence citations."""
        violations = []
        
        # Patterns that claim uplift
        uplift_claim_patterns = [
            (r"proved uplift", "Claimed 'proved uplift'"),
            (r"uplift achieved", "Claimed 'uplift achieved'"),
            (r"uplift guaranteed", "Claimed 'uplift guaranteed'"),
            (r"demonstrated uplift", "Claimed 'demonstrated uplift'"),
            (r"uplift confirmed", "Claimed 'uplift confirmed'"),
        ]
        
        # Acceptable evidence phrases that validate uplift claims
        evidence_patterns = [
            r"P3/P4 evidence package",
            r"First Light complete",
            r"integrated-run pending",
            r"G1-G5 gates",
            r"gate evidence:",
        ]
        
        for pattern, description in uplift_claim_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Check if there's evidence nearby (within 500 chars)
                start = max(0, match.start() - 250)
                end = min(len(content), match.end() + 250)
                context = content[start:end]
                
                has_evidence = any(re.search(ep, context, re.IGNORECASE) for ep in evidence_patterns)
                
                if not has_evidence:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "type": "premature_uplift_claim",
                        "severity": "CRITICAL",
                        "filepath": filepath,
                        "line": line_num,
                        "matched_text": match.group(),
                        "message": f"{description} without evidence citation at line {line_num}"
                    })
        
        return violations

    def detect_tda_enforcement_claims(self, content: str, filepath: str) -> List[Dict]:
        """Detect TDA enforcement claims without actual wiring."""
        violations = []
        
        # Patterns that claim TDA is live/enforcing
        tda_claim_patterns = [
            (r"TDA enforcement live", "Claimed 'TDA enforcement live'"),
            (r"TDA is now the final arbiter", "Claimed 'TDA is now the final arbiter'"),
            (r"TDA hooks.*wired", "Claimed TDA hooks are wired"),
            (r"TDA.*operational", "Claimed TDA is operational"),
        ]
        
        # Acceptable qualifiers that make TDA claims valid
        qualifier_patterns = [
            r"TDA.*\(not yet wired\)",
            r"TDA.*pending",
            r"TDA.*planned",
            r"TDA.*design",
        ]
        
        for pattern, description in tda_claim_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Check if there's a qualifier nearby
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                
                has_qualifier = any(re.search(qp, context, re.IGNORECASE) for qp in qualifier_patterns)
                
                if not has_qualifier:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "type": "premature_tda_claim",
                        "severity": "CRITICAL",
                        "filepath": filepath,
                        "line": line_num,
                        "matched_text": match.group(),
                        "message": f"{description} without proper qualification at line {line_num}"
                    })
        
        return violations

    def detect_phase_x_language_violations(self, content: str, filepath: str) -> List[Dict]:
        """Detect incorrect Phase X language (P3/P4 descriptions)."""
        violations = []
        
        # P3 must be described as synthetic/wind tunnel
        p3_real_world_patterns = [
            (r"P3.*real world", "P3 described as 'real world'"),
            (r"P3.*production", "P3 described as 'production'"),
            (r"P3.*live environment", "P3 described as 'live environment'"),
        ]
        
        for pattern, description in p3_real_world_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                violations.append({
                    "type": "p3_language_violation",
                    "severity": "CRITICAL",
                    "filepath": filepath,
                    "line": line_num,
                    "matched_text": match.group(),
                    "message": f"{description} - P3 must be 'synthetic wind tunnel' at line {line_num}"
                })
        
        # P4 should be described as shadow/no control authority
        p4_control_patterns = [
            (r"P4.*has control", "P4 described as having control"),
            (r"P4.*controls", "P4 described as controlling"),
            (r"P4.*authority", "P4 described as having authority"),
        ]
        
        # Check if it's properly qualified as "no control authority"
        for pattern, description in p4_control_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Look for "no" or "shadow" qualifier
                start = max(0, match.start() - 50)
                context = content[start:match.end()]
                
                if not re.search(r"\bno\b|\bshadow\b", context, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "type": "p4_language_violation",
                        "severity": "CRITICAL",
                        "filepath": filepath,
                        "line": line_num,
                        "matched_text": match.group(),
                        "message": f"{description} - P4 must be 'shadow/no control authority' at line {line_num}"
                    })
        
        return violations

    def detect_substrate_alignment_claims(self, content: str, filepath: str) -> List[Dict]:
        """Detect claims that Substrate solves alignment."""
        violations = []
        
        alignment_claim_patterns = [
            (r"Substrate solves alignment", "Claimed Substrate solves alignment"),
            (r"Substrate.*alignment.*solved", "Claimed alignment is solved"),
            (r"alignment.*guaranteed", "Claimed alignment is guaranteed"),
        ]
        
        for pattern, description in alignment_claim_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                violations.append({
                    "type": "substrate_alignment_claim",
                    "severity": "CRITICAL",
                    "filepath": filepath,
                    "line": line_num,
                    "matched_text": match.group(),
                    "message": f"{description} at line {line_num} - Substrate does not solve alignment"
                })
        
        return violations

    def check_document(self, filepath: str) -> List[Dict]:
        """Check a single document for all violations."""
        full_path = self.repo_root / filepath
        content = self.load_document(full_path)
        
        if content is None:
            return []  # Skip if document doesn't exist
        
        violations = []
        violations.extend(self.detect_premature_uplift_claims(content, filepath))
        violations.extend(self.detect_tda_enforcement_claims(content, filepath))
        violations.extend(self.detect_phase_x_language_violations(content, filepath))
        violations.extend(self.detect_substrate_alignment_claims(content, filepath))
        
        return violations

    def run(self) -> int:
        """Execute governance radar scan."""
        print("🔍 Doc Governance Radar - Scanning documentation...")
        print()
        
        # Combine watchlists
        all_docs = self.core_watchlist + self.phase_x_watchlist
        
        # Check each document
        docs_checked = 0
        for doc_path in all_docs:
            full_path = self.repo_root / doc_path
            if full_path.exists():
                docs_checked += 1
                print(f"   Checking: {doc_path}")
                violations = self.check_document(doc_path)
                self.drift_report["violations"].extend(violations)
        
        if docs_checked == 0:
            print("⏭️  SKIP: No documents found to check")
            self.drift_report["status"] = "SKIP"
            self._save_report()
            return EXIT_SKIP
        
        print(f"\n   Checked {docs_checked} document(s)")
        print()
        
        # Classify severity
        for violation in self.drift_report["violations"]:
            severity = violation.get("severity", "INFO")
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
        """Print governance scan results."""
        status = self.drift_report["status"]
        summary = self.drift_report["summary"]
        
        if status == "PASS":
            print("✅ PASS: No documentation governance violations detected")
        elif status == "WARN":
            print(f"⚠️  WARN: {summary['warning']} non-critical issue(s) detected")
        elif status == "FAIL":
            print(f"❌ FAIL: {summary['critical']} critical violation(s) detected")
        
        print()
        print(f"   Critical: {summary['critical']}")
        print(f"   Warning:  {summary['warning']}")
        print(f"   Info:     {summary['info']}")
        print()
        
        # Print details
        for violation in self.drift_report["violations"]:
            severity_icon = {
                "CRITICAL": "❌",
                "WARNING": "⚠️",
                "INFO": "ℹ️"
            }.get(violation["severity"], "•")
            filepath = violation.get("filepath", "unknown")
            line = violation.get("line", "?")
            message = violation.get("message", "Unknown violation")
            print(f"{severity_icon} [{violation['severity']}] {filepath}:{line}")
            print(f"   {message}")
            if "matched_text" in violation:
                print(f"   Text: '{violation['matched_text']}'")
            print()

    def _save_report(self):
        """Save machine-readable governance report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "doc_governance_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.drift_report, f, indent=2, sort_keys=True)
        print(f"📄 Report saved: {report_path}")

    def _save_summary(self):
        """Save human-readable governance summary."""
        summary_path = self.output_dir / "doc_governance_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Doc Governance Radar Report\n\n")
            f.write(f"**Status**: {self.drift_report['status']}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- **Critical**: {self.drift_report['summary']['critical']}\n")
            f.write(f"- **Warning**: {self.drift_report['summary']['warning']}\n")
            f.write(f"- **Info**: {self.drift_report['summary']['info']}\n\n")
            f.write("## Detected Violations\n\n")
            
            if not self.drift_report["violations"]:
                f.write("No violations detected.\n")
            else:
                for violation in self.drift_report["violations"]:
                    f.write(f"### [{violation['severity']}] {violation['type']}\n\n")
                    f.write(f"**File**: `{violation.get('filepath', 'unknown')}`\n\n")
                    f.write(f"**Line**: {violation.get('line', '?')}\n\n")
                    f.write(f"**Message**: {violation['message']}\n\n")
                    if "matched_text" in violation:
                        f.write(f"**Matched Text**: `{violation['matched_text']}`\n\n")
                    f.write("---\n\n")
        
        print(f"📄 Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Doc Governance Radar")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), 
                       help="Repository root directory")
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift"), 
                       help="Output directory")
    args = parser.parse_args()
    
    radar = DocGovernanceRadar(args.repo_root, args.output)
    exit_code = radar.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
