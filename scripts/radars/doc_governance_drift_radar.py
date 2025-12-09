#!/usr/bin/env python3
"""
Documentation Governance Drift Radar

Detects narrative violations during First Light integration sprint:
- Uplift claims without "integrated-run pending" disclaimer
- TDA enforcement claims before runner wiring is complete
- Contradictions to Phase I-II disclaimers

Exit Codes:
  0 - PASS: No governance violations detected
  1 - FAIL: Critical governance violation (uplift claim, TDA enforcement claim, Phase disclaimer contradiction)
  2 - WARN: Non-critical drift detected (missing disclaimer, ambiguous phrasing)
  3 - ERROR: Infrastructure failure (missing files, invalid paths)
  4 - SKIP: No files to scan
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

# Exit codes
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_WARN = 2
EXIT_ERROR = 3
EXIT_SKIP = 4


class DocGovernanceDriftRadar:
    """Documentation governance violation detection engine."""

    def __init__(self, docs_dir: Path, output_dir: Path, mode: str = "full-scan"):
        self.docs_dir = docs_dir
        self.output_dir = output_dir
        self.mode = mode
        self.drift_report = {
            "version": "1.0.0",
            "radar": "doc_governance",
            "mode": mode,
            "status": "PASS",
            "violations": [],
            "summary": {
                "critical": 0,
                "warning": 0,
                "info": 0
            }
        }
        
        # Governance patterns
        self.uplift_patterns = [
            r'\buplift\b',
            r'\bimprove(?:d|ment|s)?\b.*\b(?:performance|success rate|accuracy)\b',
            r'\b(?:demonstrate|show|prove)[ds]?\b.*\b(?:uplift|improvement|gains?)\b',
            r'\breduced?\b.*\b(?:abstention|risk)\b',
            r'\bincreased?\b.*\b(?:success|coverage|velocity)\b',
            r'\bΔp\b.*\b(?:positive|significant|>|greater)\b',
        ]
        
        self.tda_enforcement_patterns = [
            r'\bTDA\b.*\benforce[ds]?\b',
            r'\btriple.?descendant.?attestation\b.*\benforce[ds]?\b',
            r'\bTDA\b.*\b(?:blocks?|prevents?|stops?|guards?)\b',
            r'\bevaluate_hard_gate_decision\(\).*\b(?:active|live|enforcing|enforced)\b',
        ]
        
        # Required disclaimers for Phase I/II
        self.phase_i_markers = [
            r'Phase\s*I(?:\s+\(Current\))?',
            r'negative\s+control',
            r'100%\s+abstention',
            r'infrastructure\s+validation\s+only',
        ]
        
        self.phase_ii_markers = [
            r'Phase\s*II.*NOT\s+(?:YET\s+)?RUN',
            r'PHASE\s*II.*NOT\s+RUN\s+IN\s+PHASE\s*I',
            r'integrated-run\s+pending',
        ]
        
        self.required_disclaimers = [
            "integrated-run pending",
            "PHASE II — NOT YET RUN",
            "NOT RUN IN PHASE I",
        ]

    def scan_file(self, file_path: Path) -> List[Dict]:
        """Scan a single markdown file for governance violations."""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            violations.append({
                "type": "file_read_error",
                "severity": "ERROR",
                "file": str(file_path),
                "message": f"Failed to read file: {e}"
            })
            return violations
        
        # Check for uplift claims
        uplift_violations = self._detect_uplift_violations(file_path, content, lines)
        violations.extend(uplift_violations)
        
        # Check for TDA enforcement claims
        tda_violations = self._detect_tda_enforcement_violations(file_path, content, lines)
        violations.extend(tda_violations)
        
        # Check Phase I/II disclaimer consistency
        phase_violations = self._detect_phase_disclaimer_violations(file_path, content, lines)
        violations.extend(phase_violations)
        
        return violations

    def _detect_uplift_violations(self, file_path: Path, content: str, lines: List[str]) -> List[Dict]:
        """Detect uplift claims without proper disclaimers."""
        violations = []
        
        # Patterns that indicate positive uplift claims (actual claims, not negations or gate definitions)
        positive_claim_patterns = [
            r'\b(?:demonstrates?|shows?|proves?|achieved?)\b.*\buplift\b',
            r'\buplift\b.*\b(?:observed|measured|detected|confirmed)\b',
            r'\bsignificant\b.*\buplift\b',
            r'\b(?:successful|effective)\b.*\buplift\b',
            r'\bΔp\b.*\b(?:>|greater|positive)\b.*\b(?:0\.\d+|\d+%)',
        ]
        
        # Negation patterns (these are OK - they're saying NO uplift)
        negation_patterns = [
            r'\bno\b.*\buplift\b',
            r'\bzero\b.*\buplift\b',
            r'\buplift\b.*\bnone\b',
            r'\buplift\b.*\bzero\b',
            r'\bwithout\b.*\buplift\b',
            r'\bdoes\s+not.*\buplift\b',
            r'\bdo\s+not.*\buplift\b',
            r'\bcannot.*\buplift\b',
            r'\bforbid.*\buplift\b',
        ]
        
        # Gate/criteria patterns (these are OK - defining rules, not claiming)
        gate_patterns = [
            r'\buplift\s+evidence\s+gate\b',
            r'\bgate.*uplift\b',
            r'\bfor.*uplift.*to\s+(?:be|qualify)\b',
            r'\bif.*uplift\b',
            r'\bwhen.*uplift\b',
            r'\bcriteria.*uplift\b',
            r'\brequires?.*uplift\b',
            r'\bmust.*\buplift\b',
            r'\bshould.*\buplift\b',
            r'\bwould.*\buplift\b',
            r'\bfuture.*\buplift\b',
            r'\bpotential.*\buplift\b',
            r'\bexpected.*\buplift\b',
            r'\btarget.*\buplift\b',
            r'\bgoal.*\buplift\b',
        ]
        
        # Check for positive uplift claims
        positive_claims = []
        for i, line in enumerate(lines, 1):
            # Skip if line is a quote/example of what NOT to say (contains quotes or "DON'T" markers)
            is_quote_example = (
                re.search(r'^[\s>]*["\'].*["\']', line) or  # Quoted line
                re.search(r'❌.*DON\'?T', line, re.IGNORECASE) or  # "DON'T" example
                re.search(r'^\s*-\s*["\']', line) or  # List item with quote
                re.search(r'example.*don\'?t', line, re.IGNORECASE) or  # Example of what not to do
                re.search(r'avoid.*saying', line, re.IGNORECASE)  # Avoid saying X
            )
            if is_quote_example:
                continue
            
            # Skip if line contains negation
            is_negation = any(re.search(pat, line, re.IGNORECASE) for pat in negation_patterns)
            if is_negation:
                continue
            
            # Skip if line is about gate/criteria
            is_gate = any(re.search(pat, line, re.IGNORECASE) for pat in gate_patterns)
            if is_gate:
                continue
            
            # Check for positive claim
            for pattern in positive_claim_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    positive_claims.append((i, line))
                    break
        
        if not positive_claims:
            return violations
        
        # Check if proper disclaimers are present in document
        has_integrated_run_disclaimer = re.search(r'integrated-run\s+pending', content, re.IGNORECASE)
        has_phase_ii_disclaimer = re.search(r'PHASE\s*II.*NOT\s+(?:YET\s+)?RUN', content, re.IGNORECASE)
        has_not_run_disclaimer = re.search(r'NOT\s+RUN\s+IN\s+PHASE\s*I', content, re.IGNORECASE)
        
        # If document has top-level disclaimer, it's OK
        if has_integrated_run_disclaimer or has_phase_ii_disclaimer or has_not_run_disclaimer:
            return violations
        
        # Check each positive claim for local disclaimers
        for line_num, line in positive_claims:
            # Get context window (7 lines before and after)
            start_idx = max(0, line_num - 8)
            end_idx = min(len(lines), line_num + 7)
            context_lines = lines[start_idx:end_idx]
            context = '\n'.join(context_lines)
            
            # Check if context has disclaimers or qualifiers
            context_has_disclaimer = (
                re.search(r'integrated-run\s+pending', context, re.IGNORECASE) or
                re.search(r'PHASE\s*II.*NOT\s+(?:YET\s+)?RUN', context, re.IGNORECASE) or
                re.search(r'NOT\s+RUN\s+IN\s+PHASE\s*I', context, re.IGNORECASE) or
                re.search(r'to\s+be\s+(?:implemented|run|completed)', context, re.IGNORECASE) or
                re.search(r'will\s+(?:demonstrate|show|test)', context, re.IGNORECASE) or
                re.search(r'pending\s+(?:verification|validation|completion)', context, re.IGNORECASE) or
                re.search(r'once\s+(?:complete|wired|integrated)', context, re.IGNORECASE)
            )
            
            if not context_has_disclaimer:
                violations.append({
                    "type": "uplift_claim_without_disclaimer",
                    "severity": "CRITICAL",
                    "file": str(file_path.relative_to(self.docs_dir.parent)),
                    "line": line_num,
                    "snippet": line.strip(),
                    "message": f"Positive uplift claim without 'integrated-run pending' disclaimer at line {line_num}"
                })
        
        return violations

    def _detect_tda_enforcement_violations(self, file_path: Path, content: str, lines: List[str]) -> List[Dict]:
        """Detect TDA enforcement claims before runner wiring is complete."""
        violations = []
        
        for i, line in enumerate(lines, 1):
            for pattern in self.tda_enforcement_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check context for clarifications
                    start_idx = max(0, i - 6)
                    end_idx = min(len(lines), i + 5)
                    context = '\n'.join(lines[start_idx:end_idx])
                    
                    # Allow if context clarifies this is future/planned
                    has_future_qualifier = re.search(
                        r'\b(?:will|future|planned|to be|once|after|when).*\b(?:wired|integrated|connected)\b',
                        context,
                        re.IGNORECASE
                    )
                    
                    if not has_future_qualifier:
                        violations.append({
                            "type": "tda_enforcement_claim",
                            "severity": "CRITICAL",
                            "file": str(file_path.relative_to(self.docs_dir.parent)),
                            "line": i,
                            "snippet": line.strip(),
                            "message": f"TDA enforcement claim before runner wiring complete at line {i}"
                        })
        
        return violations

    def _detect_phase_disclaimer_violations(self, file_path: Path, content: str, lines: List[str]) -> List[Dict]:
        """Detect contradictions to Phase I-II disclaimers."""
        violations = []
        
        # Check if file discusses Phase I
        has_phase_i_content = any(re.search(pattern, content, re.IGNORECASE) for pattern in self.phase_i_markers)
        
        if has_phase_i_content:
            # Check for required Phase I disclaimers
            required_phrases = [
                (r'negative\s+control', 'Phase I negative control'),
                (r'100%\s+abstention', 'Phase I 100% abstention'),
                (r'infrastructure\s+validation', 'Phase I infrastructure validation only'),
            ]
            
            for pattern, description in required_phrases:
                if re.search(r'Phase\s*I', content, re.IGNORECASE) and \
                   not re.search(pattern, content, re.IGNORECASE):
                    # This is a warning, not critical, as some docs may reference Phase I briefly
                    violations.append({
                        "type": "incomplete_phase_i_disclaimer",
                        "severity": "WARNING",
                        "file": str(file_path.relative_to(self.docs_dir.parent)),
                        "message": f"Phase I discussion without '{description}' context"
                    })
        
        # Check for Phase I uplift claims (critical violation)
        phase_i_sections = re.finditer(
            r'(Phase\s*I[^#\n]*?(?:(?=Phase\s*II)|(?=##)|$))',
            content,
            re.IGNORECASE | re.DOTALL
        )
        
        for match in phase_i_sections:
            section = match.group(1)
            # Check for uplift claims in Phase I section
            for pattern in self.uplift_patterns:
                if re.search(pattern, section, re.IGNORECASE):
                    # Find line number
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "type": "phase_i_uplift_claim",
                        "severity": "CRITICAL",
                        "file": str(file_path.relative_to(self.docs_dir.parent)),
                        "line": line_num,
                        "message": f"Uplift claim in Phase I section (Phase I has no uplift)"
                    })
                    break
        
        return violations

    def scan_pr_diff(self, diff_path: Path) -> List[Dict]:
        """Scan a git diff for governance violations in added lines."""
        violations = []
        
        try:
            with open(diff_path, 'r', encoding='utf-8') as f:
                diff_content = f.read()
        except Exception as e:
            return [{
                "type": "diff_read_error",
                "severity": "ERROR",
                "file": str(diff_path),
                "message": f"Failed to read diff: {e}"
            }]
        
        current_file = None
        added_lines = []
        
        # Parse diff
        for line in diff_content.split('\n'):
            # Track which file we're in
            if line.startswith('+++'):
                current_file = line[4:].strip()
                # Only scan markdown files
                if not current_file.endswith('.md'):
                    current_file = None
                added_lines = []
            elif current_file and line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:])  # Remove '+'
        
        # Check added lines for violations
        if current_file:
            added_content = '\n'.join(added_lines)
            
            # Check for uplift claims
            for pattern in self.uplift_patterns:
                matches = re.finditer(pattern, added_content, re.IGNORECASE)
                for match in matches:
                    # Check if proper disclaimer is in the added content
                    has_disclaimer = any(
                        disclaimer in added_content
                        for disclaimer in self.required_disclaimers
                    )
                    
                    if not has_disclaimer:
                        violations.append({
                            "type": "pr_uplift_without_disclaimer",
                            "severity": "CRITICAL",
                            "file": current_file,
                            "snippet": match.group(0),
                            "message": f"PR adds uplift claim without 'integrated-run pending' disclaimer"
                        })
            
            # Check for TDA enforcement claims
            for pattern in self.tda_enforcement_patterns:
                if re.search(pattern, added_content, re.IGNORECASE):
                    violations.append({
                        "type": "pr_tda_enforcement_claim",
                        "severity": "CRITICAL",
                        "file": current_file,
                        "message": f"PR adds TDA enforcement claim before runner wiring complete"
                    })
        
        return violations

    def run(self) -> int:
        """Execute drift detection based on mode."""
        print(f"🔍 Documentation Governance Radar - {self.mode} mode")
        print(f"   Directory: {self.docs_dir}")
        print()
        
        if self.mode == "full-scan":
            return self._run_full_scan()
        elif self.mode == "watchdog":
            return self._run_watchdog()
        elif self.mode == "pr-diff":
            return self._run_pr_diff()
        else:
            print(f"❌ ERROR: Unknown mode '{self.mode}'", file=sys.stderr)
            return EXIT_ERROR

    def _run_full_scan(self) -> int:
        """Scan all markdown files in docs directory."""
        md_files = list(self.docs_dir.rglob("*.md"))
        
        if not md_files:
            print("⏭️  SKIP: No markdown files found")
            self.drift_report["status"] = "SKIP"
            self._save_report()
            return EXIT_SKIP
        
        print(f"Scanning {len(md_files)} markdown files...")
        print()
        
        all_violations = []
        for md_file in md_files:
            violations = self.scan_file(md_file)
            all_violations.extend(violations)
        
        self.drift_report["violations"] = all_violations
        return self._finalize_report()

    def _run_watchdog(self) -> int:
        """Watchdog mode: continuous monitoring of key governance docs."""
        key_docs = [
            "PHASE2_RFL_UPLIFT_PLAN.md",
            "RFL_PHASE_I_TRUTH_SOURCE.md",
            "VSD_PHASE_2.md",
            "README.md",
            "CLAUDE.md",
        ]
        
        print("Watchdog mode: Scanning key governance documents...")
        print()
        
        all_violations = []
        scanned_files = []
        
        for doc_name in key_docs:
            # Search for the doc in docs_dir
            matches = list(self.docs_dir.rglob(doc_name))
            if not matches:
                # Try parent directory
                matches = list(self.docs_dir.parent.rglob(doc_name))
            
            for doc_path in matches:
                scanned_files.append(doc_path)
                violations = self.scan_file(doc_path)
                all_violations.extend(violations)
        
        if not scanned_files:
            print("⏭️  SKIP: No key governance documents found")
            self.drift_report["status"] = "SKIP"
            self._save_report()
            return EXIT_SKIP
        
        print(f"Scanned {len(scanned_files)} key documents")
        print()
        
        self.drift_report["violations"] = all_violations
        return self._finalize_report()

    def _run_pr_diff(self) -> int:
        """PR diff mode: scan git diff for violations."""
        # Look for diff file in output directory
        diff_path = self.output_dir / "pr_diff.patch"
        
        if not diff_path.exists():
            print(f"⏭️  SKIP: No PR diff found at {diff_path}")
            self.drift_report["status"] = "SKIP"
            self._save_report()
            return EXIT_SKIP
        
        print(f"Scanning PR diff: {diff_path}")
        print()
        
        violations = self.scan_pr_diff(diff_path)
        self.drift_report["violations"] = violations
        return self._finalize_report()

    def _finalize_report(self) -> int:
        """Classify violations and determine exit code."""
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
        """Print governance violation results."""
        status = self.drift_report["status"]
        summary = self.drift_report["summary"]
        
        if status == "PASS":
            print("✅ PASS: No governance violations detected")
            print()
            print("   The organism does not move unless the Cortex approves.")
        elif status == "WARN":
            print(f"⚠️  WARN: {summary['warning']} non-critical violation(s) detected")
        elif status == "FAIL":
            print(f"❌ FAIL: {summary['critical']} critical governance violation(s) detected")
            print()
            print("   ⛔ ORGANISM NOT ALIVE: Narrative integrity compromised")
        
        print()
        print(f"   Critical: {summary['critical']}")
        print(f"   Warning:  {summary['warning']}")
        print(f"   Info:     {summary['info']}")
        print()
        
        # Print violation details
        for violation in self.drift_report["violations"]:
            severity_icon = {
                "CRITICAL": "❌",
                "WARNING": "⚠️",
                "INFO": "ℹ️",
                "ERROR": "💥"
            }.get(violation["severity"], "•")
            
            file_info = violation.get("file", "")
            line_info = f" (line {violation['line']})" if "line" in violation else ""
            
            print(f"{severity_icon} [{violation['severity']}] {violation['message']}")
            if file_info:
                print(f"   File: {file_info}{line_info}")
            if "snippet" in violation:
                print(f"   Snippet: {violation['snippet'][:80]}...")
            print()

    def _save_report(self):
        """Save machine-readable governance report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "doc_governance_drift_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.drift_report, f, indent=2, sort_keys=True)
        print(f"📄 Report saved: {report_path}")

    def _save_summary(self):
        """Save human-readable governance summary."""
        summary_path = self.output_dir / "doc_governance_drift_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Documentation Governance Drift Report\n\n")
            f.write(f"**Status**: {self.drift_report['status']}\n\n")
            f.write(f"**Mode**: {self.drift_report['mode']}\n\n")
            
            if self.drift_report['status'] == 'FAIL':
                f.write("## ⛔ ORGANISM NOT ALIVE\n\n")
                f.write("Narrative integrity compromised. No document may imply 'organism alive' until First Light run completes.\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Critical**: {self.drift_report['summary']['critical']}\n")
            f.write(f"- **Warning**: {self.drift_report['summary']['warning']}\n")
            f.write(f"- **Info**: {self.drift_report['summary']['info']}\n\n")
            
            f.write("## Governance Rules\n\n")
            f.write("1. ❌ Any mention of uplift MUST include 'integrated-run pending' disclaimer\n")
            f.write("2. ❌ No TDA enforcement claims before runner wiring complete\n")
            f.write("3. ❌ No contradiction to Phase I-II disclaimers\n\n")
            
            f.write("## Detected Violations\n\n")
            
            if not self.drift_report["violations"]:
                f.write("*No violations detected*\n\n")
            else:
                for violation in self.drift_report["violations"]:
                    f.write(f"### [{violation['severity']}] {violation['type']}\n\n")
                    f.write(f"{violation['message']}\n\n")
                    for key, value in violation.items():
                        if key not in ["type", "severity", "message"]:
                            f.write(f"- **{key}**: `{value}`\n")
                    f.write("\n")
        
        print(f"📄 Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Documentation Governance Drift Radar - First Light Watchdog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full-scan  - Scan all markdown files in docs directory
  watchdog   - Monitor key governance documents only
  pr-diff    - Scan PR diff for violations (requires pr_diff.patch in output dir)

Examples:
  # Full scan
  python doc_governance_drift_radar.py --mode=full-scan --docs=docs/ --output=artifacts/drift/

  # Watchdog mode
  python doc_governance_drift_radar.py --mode=watchdog --docs=docs/ --output=artifacts/drift/

  # PR diff mode (requires git diff > artifacts/drift/pr_diff.patch first)
  git diff origin/main...HEAD -- '*.md' > artifacts/drift/pr_diff.patch
  python doc_governance_drift_radar.py --mode=pr-diff --docs=docs/ --output=artifacts/drift/
        """
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path("docs"),
        help="Path to documentation directory (default: docs/)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/drift"),
        help="Output directory for reports (default: artifacts/drift/)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full-scan", "watchdog", "pr-diff"],
        default="full-scan",
        help="Scan mode (default: full-scan)"
    )
    
    args = parser.parse_args()
    
    radar = DocGovernanceDriftRadar(args.docs, args.output, args.mode)
    exit_code = radar.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
