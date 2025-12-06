#!/usr/bin/env python3
"""
CLAUDE I — Structural Grammarian
Schema validation and syntactic conformance auditor
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml


class SchemaAuditor:
    def __init__(self, root_path: Path):
        self.root = root_path
        self.violations: List[Dict[str, Any]] = []
        self.passed: List[Dict[str, Any]] = []
        self.total_files = 0

    def scan_files(self) -> List[Path]:
        """Discover all JSON/YAML config and schema files"""
        patterns = ["**/*.json", "**/*.yaml", "**/*.yml"]

        # Exclude patterns
        exclude_dirs = {
            "node_modules", ".git", "__pycache__", "venv",
            ".venv", "dist", "build", ".pytest_cache", "lean_proj"
        }

        files = []
        for pattern in patterns:
            for f in self.root.rglob(pattern):
                # Skip excluded directories
                if any(excl in f.parts for excl in exclude_dirs):
                    continue
                # Skip package-lock.json (too large, auto-generated)
                if f.name == "package-lock.json":
                    continue
                files.append(f)

        return sorted(files)

    def check_encoding(self, file_path: Path) -> Dict[str, Any]:
        """Check file encoding for BOM, null bytes, non-ASCII"""
        result = {"has_bom": False, "has_null_bytes": False, "non_ascii_count": 0, "encoding": "ASCII"}

        try:
            with open(file_path, "rb") as f:
                raw = f.read()

            # Check for BOM
            if raw.startswith(b'\xef\xbb\xbf'):
                result["has_bom"] = True
                result["encoding"] = "UTF-8-BOM"

            # Check for null bytes (like in openapi.json)
            if b'\x00' in raw:
                result["has_null_bytes"] = True
                result["null_byte_count"] = raw.count(b'\x00')

            # Check for non-ASCII characters
            try:
                raw.decode('ascii')
            except UnicodeDecodeError:
                result["encoding"] = "UTF-8"
                result["non_ascii_count"] = sum(1 for b in raw if b > 127)

        except Exception as e:
            result["error"] = str(e)

        return result

    def check_newlines(self, file_path: Path) -> Dict[str, Any]:
        """Check newline consistency (LF vs CRLF)"""
        result = {"type": None, "consistent": True, "lf_count": 0, "crlf_count": 0}

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            result["lf_count"] = content.count(b'\n') - content.count(b'\r\n')
            result["crlf_count"] = content.count(b'\r\n')

            if result["crlf_count"] > 0 and result["lf_count"] > 0:
                result["consistent"] = False
                result["type"] = "MIXED"
            elif result["crlf_count"] > 0:
                result["type"] = "CRLF"
            elif result["lf_count"] > 0:
                result["type"] = "LF"

        except Exception as e:
            result["error"] = str(e)

        return result

    def validate_json(self, file_path: Path) -> Dict[str, Any]:
        """Validate JSON syntax and check for canonical key ordering"""
        result = {"valid": False, "error": None, "keys_ordered": None, "formatted": None}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse JSON
            data = json.loads(content)
            result["valid"] = True

            # Check if it's formatted (not minified)
            result["formatted"] = "\n" in content

            # For objects, check if keys are sorted alphabetically at root level
            if isinstance(data, dict):
                keys = list(data.keys())
                sorted_keys = sorted(keys)
                result["keys_ordered"] = (keys == sorted_keys)
                if not result["keys_ordered"]:
                    result["expected_order"] = sorted_keys[:5]  # First 5 for brevity
                    result["actual_order"] = keys[:5]

        except json.JSONDecodeError as e:
            result["error"] = f"JSON decode error: {e}"
        except UnicodeDecodeError as e:
            result["error"] = f"Unicode decode error: {e}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def validate_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Validate YAML syntax"""
        result = {"valid": False, "error": None}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            result["valid"] = True

        except yaml.YAMLError as e:
            result["error"] = f"YAML parse error: {e}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def audit_file(self, file_path: Path) -> Dict[str, Any]:
        """Perform comprehensive audit on a single file"""
        rel_path = file_path.relative_to(self.root)

        audit = {
            "file": str(rel_path),
            "size_bytes": file_path.stat().st_size,
            "encoding": self.check_encoding(file_path),
            "newlines": self.check_newlines(file_path),
        }

        # Type-specific validation
        if file_path.suffix == ".json":
            audit["json"] = self.validate_json(file_path)
            audit["type"] = "json"
        elif file_path.suffix in [".yaml", ".yml"]:
            audit["yaml"] = self.validate_yaml(file_path)
            audit["type"] = "yaml"

        # Determine if this file has violations
        violations = []

        # Check for critical issues
        if audit["encoding"]["has_bom"]:
            violations.append("UTF-8 BOM present (should be removed)")

        if audit["encoding"]["has_null_bytes"]:
            violations.append(f"Null bytes detected ({audit['encoding']['null_byte_count']})")

        if audit["newlines"]["type"] == "CRLF":
            violations.append("CRLF newlines (should be LF)")
        elif audit["newlines"]["type"] == "MIXED":
            violations.append("Mixed newline types (inconsistent)")

        # JSON-specific checks
        if audit["type"] == "json":
            if not audit["json"]["valid"]:
                violations.append(f"Invalid JSON: {audit['json']['error']}")
            elif not audit["json"]["formatted"]:
                violations.append("JSON is minified (should be formatted)")

        # YAML-specific checks
        if audit["type"] == "yaml":
            if not audit["yaml"]["valid"]:
                violations.append(f"Invalid YAML: {audit['yaml']['error']}")

        audit["violations"] = violations

        return audit

    def run_audit(self) -> Dict[str, Any]:
        """Run complete audit and generate report"""
        files = self.scan_files()
        self.total_files = len(files)

        print(f"[Structural Grammarian] Scanning {self.total_files} schema/config files...")

        for file_path in files:
            audit = self.audit_file(file_path)

            if audit["violations"]:
                self.violations.append(audit)
            else:
                self.passed.append(audit)

        # Generate summary
        summary = {
            "total_files": self.total_files,
            "passed": len(self.passed),
            "violations": len(self.violations),
            "pass_rate": round(len(self.passed) / self.total_files * 100, 2) if self.total_files > 0 else 0,
        }

        return {
            "summary": summary,
            "violations": self.violations,
            "passed": self.passed,
        }

    def format_report(self, report: Dict[str, Any]) -> str:
        """Format audit report for output"""
        lines = []
        lines.append("=" * 80)
        lines.append("CLAUDE I — Structural Grammarian")
        lines.append("Schema Validation & Syntactic Conformance Audit")
        lines.append("=" * 80)
        lines.append("")

        summary = report["summary"]
        lines.append(f"Total Files Scanned: {summary['total_files']}")
        lines.append(f"Passed: {summary['passed']}")
        lines.append(f"Violations: {summary['violations']}")
        lines.append(f"Pass Rate: {summary['pass_rate']}%")
        lines.append("")

        if report["violations"]:
            lines.append("=" * 80)
            lines.append("VIOLATIONS DETECTED")
            lines.append("=" * 80)
            lines.append("")

            for i, v in enumerate(report["violations"], 1):
                lines.append(f"[{i}] {v['file']}")
                lines.append(f"    Type: {v['type'].upper()}")
                lines.append(f"    Size: {v['size_bytes']} bytes")

                for violation in v["violations"]:
                    lines.append(f"    ⚠ {violation}")

                # Show encoding details if relevant
                if v["encoding"]["has_bom"] or v["encoding"]["has_null_bytes"]:
                    lines.append(f"    Encoding: {v['encoding']['encoding']}")
                    if v["encoding"]["has_null_bytes"]:
                        lines.append(f"    Null bytes: {v['encoding']['null_byte_count']}")

                # Show newline details if relevant
                if v["newlines"]["type"] in ["CRLF", "MIXED"]:
                    lines.append(f"    Newlines: {v['newlines']['type']} " +
                               f"(LF={v['newlines']['lf_count']}, CRLF={v['newlines']['crlf_count']})")

                lines.append("")

        lines.append("=" * 80)
        if summary["violations"] == 0:
            lines.append("[PASS] Schema Conformance OK [files=" + str(summary['total_files']) + "]")
        else:
            lines.append(f"[FAIL] {summary['violations']} file(s) with violations")
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    root = Path(__file__).parent.parent
    auditor = SchemaAuditor(root)

    report = auditor.run_audit()

    # Print formatted report
    print(auditor.format_report(report))

    # Save detailed report as JSON
    report_path = root / "artifacts" / "schema_audit_report.json"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved: {report_path.relative_to(root)}")

    # Exit with non-zero if violations found
    sys.exit(1 if report["summary"]["violations"] > 0 else 0)


if __name__ == "__main__":
    main()
