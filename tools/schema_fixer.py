#!/usr/bin/env python3
"""
CLAUDE I — Structural Grammarian
Automated Schema Conformance Fixer
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


class SchemaFixer:
    def __init__(self, root_path: Path, dry_run: bool = False):
        self.root = root_path
        self.dry_run = dry_run
        self.fixes_applied: List[Dict[str, Any]] = []

    def remove_bom(self, file_path: Path) -> bool:
        """Remove UTF-8 BOM from file"""
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            if content.startswith(b'\xef\xbb\xbf'):
                print(f"  Removing BOM from {file_path.relative_to(self.root)}")
                if not self.dry_run:
                    with open(file_path, "wb") as f:
                        f.write(content[3:])  # Skip BOM
                return True
        except Exception as e:
            print(f"  Error removing BOM: {e}")
        return False

    def convert_crlf_to_lf(self, file_path: Path) -> bool:
        """Convert CRLF line endings to LF"""
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            if b'\r\n' in content:
                print(f"  Converting CRLF to LF in {file_path.relative_to(self.root)}")
                if not self.dry_run:
                    fixed_content = content.replace(b'\r\n', b'\n')
                    with open(file_path, "wb") as f:
                        f.write(fixed_content)
                return True
        except Exception as e:
            print(f"  Error converting newlines: {e}")
        return False

    def format_json(self, file_path: Path) -> bool:
        """Format minified JSON with proper indentation"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"  Formatting JSON: {file_path.relative_to(self.root)}")
            if not self.dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.write("\n")  # Ensure trailing newline
            return True
        except Exception as e:
            print(f"  Error formatting JSON: {e}")
        return False

    def fix_file(self, file_path: Path, violations: List[str]) -> Dict[str, Any]:
        """Apply fixes to a file based on its violations"""
        fixes = {
            "file": str(file_path.relative_to(self.root)),
            "fixes_applied": [],
            "success": True,
        }

        print(f"\nFixing: {file_path.relative_to(self.root)}")

        for violation in violations:
            if "UTF-8 BOM" in violation:
                if self.remove_bom(file_path):
                    fixes["fixes_applied"].append("Removed UTF-8 BOM")

            if "CRLF" in violation or "Mixed" in violation:
                if self.convert_crlf_to_lf(file_path):
                    fixes["fixes_applied"].append("Converted CRLF to LF")

            if "minified" in violation:
                if self.format_json(file_path):
                    fixes["fixes_applied"].append("Formatted JSON")

        return fixes

    def run_fixes(self, violations_report: Dict[str, Any]) -> Dict[str, Any]:
        """Run fixes on all files with violations"""
        print("=" * 80)
        print("CLAUDE I — Structural Grammarian")
        print("Automated Schema Conformance Fixer")
        print("=" * 80)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY FIXES'}")
        print("")

        violations = violations_report.get("violations", [])

        for violation_entry in violations:
            file_path = self.root / violation_entry["file"]

            # Skip files we can't or shouldn't fix
            skip_files = [
                "tmp/openapi.json",  # Too corrupted, needs regeneration
                "schema_actual.json",  # Empty file, likely intentional
            ]

            if any(skip in str(file_path) for skip in skip_files):
                print(f"\nSkipping (requires manual intervention): {violation_entry['file']}")
                continue

            # Skip YAML files with syntax errors (need manual fix)
            if violation_entry.get("type") == "yaml":
                yaml_violations = violation_entry.get("violations", [])
                if any("Invalid YAML" in v for v in yaml_violations):
                    print(f"\nSkipping (YAML syntax error): {violation_entry['file']}")
                    continue

            fixes = self.fix_file(file_path, violation_entry["violations"])
            self.fixes_applied.append(fixes)

        return {
            "total_fixed": len(self.fixes_applied),
            "fixes": self.fixes_applied,
        }

    def format_report(self, result: Dict[str, Any]) -> str:
        """Format fix report"""
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append("FIX SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Total Files Fixed: {result['total_fixed']}")
        lines.append("")

        for fix in result["fixes"]:
            if fix["fixes_applied"]:
                lines.append(f"{fix['file']}:")
                for applied in fix["fixes_applied"]:
                    lines.append(f"  ✓ {applied}")

        return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix schema conformance violations")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without applying changes")
    parser.add_argument("--report", type=str, default="artifacts/schema_audit_report.json", help="Path to audit report")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    fixer = SchemaFixer(root, dry_run=args.dry_run)

    # Load violations report
    report_path = root / args.report
    if not report_path.exists():
        print(f"ERROR: Report not found: {report_path}")
        print("Run schema_audit.py first to generate the report")
        sys.exit(1)

    with open(report_path, "r") as f:
        violations_report = json.load(f)

    # Run fixes
    result = fixer.run_fixes(violations_report)

    # Print summary
    print(fixer.format_report(result))

    if args.dry_run:
        print("\n[DRY RUN] No changes were made. Run without --dry-run to apply fixes.")
    else:
        print("\n[SUCCESS] Fixes applied. Re-run schema_audit.py to verify.")


if __name__ == "__main__":
    main()
