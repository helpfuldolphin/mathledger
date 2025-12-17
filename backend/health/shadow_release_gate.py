"""
Shadow Release Gate — Distribution Gate for SHADOW MODE Artifacts

This module implements a CI-enforced contract check that prevents release/distribution
of artifacts or claims that violate SHADOW MODE semantics.

WHAT IS GATED:
- Documents declaring SHADOW-GATED that use prohibited phrases
- SHADOW-GATED contexts without gate registry references
- Unqualified "SHADOW MODE" usage (must specify OBSERVE or GATED)

WHAT IS NOT GATED:
- SHADOW-OBSERVE documents using "observational only" (correct usage)
- Runtime governance decisions
- Frozen harnesses (CAL-EXP-1/2/3, P5)

CONTRACT REFERENCE:
- docs/system_law/SHADOW_MODE_CONTRACT.md v1.0.0

SHADOW MODE: SHADOW-GATED
GATE REGISTRY: SRG-001 (shadow_release_gate)

Usage:
    python -m backend.health.shadow_release_gate --scan-dir docs/
    python -m backend.health.shadow_release_gate --file path/to/doc.md

Exit codes:
    0 = PASS (no violations)
    1 = FAIL (violations found)
    2 = ERROR (scan error)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

__all__ = [
    "ShadowReleaseGate",
    "GateViolation",
    "GateReport",
    "scan_file",
    "scan_directory",
    "load_gate_registry",
]

# =============================================================================
# PROHIBITED PHRASES (per SHADOW_MODE_CONTRACT.md §4.1)
# =============================================================================

PROHIBITED_PHRASES_IN_GATED = [
    "observational only",
    "advisory",
    "informational",
    "passive",
    "non-blocking",
]

# Patterns that indicate SHADOW-GATED context
SHADOW_GATED_PATTERNS = [
    r"SHADOW-GATED",
    r"SHADOW_GATED",
    r"shadow-gated",
    r"shadow_gated",
    r"sub-mode:\s*SHADOW-GATED",
    r"mode:\s*SHADOW-GATED",
]

# Patterns that indicate SHADOW-OBSERVE context
SHADOW_OBSERVE_PATTERNS = [
    r"SHADOW-OBSERVE",
    r"SHADOW_OBSERVE",
    r"shadow-observe",
    r"shadow_observe",
    r"sub-mode:\s*SHADOW-OBSERVE",
    r"mode:\s*SHADOW-OBSERVE",
]

# Pattern for unqualified SHADOW MODE usage
UNQUALIFIED_SHADOW_PATTERN = r"SHADOW\s+MODE(?!\s*[-_]?\s*(OBSERVE|GATED|CONTRACT|GOVERNANCE|AUTHORITY))"

# Gate registry reference patterns
GATE_REGISTRY_PATTERNS = [
    r"gate_id\s*[:=]",
    r"GATE\s*REGISTRY",
    r"gate[_-]?registry",
    r"GateRegistry",
    r"SRG-\d{3}",  # Standard gate ID format
]

# Allowed technical contexts for "fail-close" / "fail-safe"
ALLOWED_TECHNICAL_PHRASES = [
    "fail-close",
    "fail-closed",
    "fail-safe",
    "fail silently",  # Only in SHADOW-OBSERVE failure handling
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GateViolation:
    """Single violation of SHADOW MODE contract."""
    file_path: str
    line_number: int
    violation_type: str
    message: str
    context: str  # Surrounding text
    severity: str = "ERROR"  # ERROR | WARN

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateReport:
    """Report from shadow release gate scan."""
    scan_path: str
    files_scanned: int = 0
    violations: List[GateViolation] = field(default_factory=list)
    passed: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    contract_version: str = "1.0.0"
    gate_id: str = "SRG-001"

    def add_violation(self, violation: GateViolation) -> None:
        self.violations.append(violation)
        if violation.severity == "ERROR":
            self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_path": self.scan_path,
            "files_scanned": self.files_scanned,
            "violations": [v.to_dict() for v in self.violations],
            "violation_count": len(self.violations),
            "passed": self.passed,
            "timestamp": self.timestamp,
            "contract_version": self.contract_version,
            "gate_id": self.gate_id,
            "shadow_mode": "SHADOW-GATED",
            "system_impact": "BLOCKED" if not self.passed else "ALLOWED",
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass
class GateRegistryEntry:
    """Entry in the shadow gate registry."""
    gate_id: str
    operation: str
    condition: str
    enforcement_level: str  # BLOCK | WARN
    effective_date: str
    authority: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GateRegistryEntry":
        return cls(
            gate_id=d["gate_id"],
            operation=d["operation"],
            condition=d["condition"],
            enforcement_level=d["enforcement_level"],
            effective_date=d["effective_date"],
            authority=d["authority"],
        )


# =============================================================================
# GATE REGISTRY
# =============================================================================

def get_default_registry_path() -> Path:
    """Get default path to gate registry."""
    return Path(__file__).parent.parent.parent / "config" / "shadow_gate_registry.yaml"


def load_gate_registry(registry_path: Optional[Path] = None) -> List[GateRegistryEntry]:
    """
    Load gate registry from YAML file.

    Returns empty list if file doesn't exist (allows bootstrapping).
    """
    if registry_path is None:
        registry_path = get_default_registry_path()

    if not registry_path.exists():
        # Return minimal bootstrap registry
        return [
            GateRegistryEntry(
                gate_id="SRG-001",
                operation="shadow_release_gate",
                condition="Prohibited phrases in SHADOW-GATED context",
                enforcement_level="BLOCK",
                effective_date="2025-12-17",
                authority="SMGA",
            )
        ]

    try:
        import yaml
        with open(registry_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        entries = []
        for entry_data in data.get("gates", []):
            entries.append(GateRegistryEntry.from_dict(entry_data))
        return entries
    except ImportError:
        # YAML not available, use JSON fallback
        json_path = registry_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [GateRegistryEntry.from_dict(e) for e in data.get("gates", [])]
        return []
    except Exception:
        return []


# =============================================================================
# SCANNING LOGIC
# =============================================================================

class ShadowReleaseGate:
    """
    Shadow Release Gate scanner.

    Scans documents for SHADOW MODE contract violations:
    1. Prohibited phrases in SHADOW-GATED contexts
    2. SHADOW-GATED without gate registry reference
    3. Unqualified "SHADOW MODE" usage
    """

    def __init__(self, registry: Optional[List[GateRegistryEntry]] = None):
        self.registry = registry or load_gate_registry()
        self.gate_ids = {e.gate_id for e in self.registry}

    def _detect_context(self, content: str) -> Tuple[bool, bool]:
        """
        Detect SHADOW mode context in document.

        Returns: (is_gated, is_observe)

        Only marks as GATED if the document DECLARES itself as SHADOW-GATED,
        not if it merely describes or references the concept.
        """
        # Patterns that indicate the document DECLARES itself as SHADOW-GATED
        gated_declaration_patterns = [
            r"\*\*Mode\*\*:\s*SHADOW-GATED",
            r"\*\*SHADOW MODE\*\*:\s*SHADOW-GATED",
            r"Mode:\s*SHADOW-GATED",
            r"SHADOW MODE:\s*SHADOW-GATED",
            r"sub-mode:\s*SHADOW-GATED",
            r"operates\s+(?:in\s+)?SHADOW-GATED",
            r"running\s+(?:in\s+)?SHADOW-GATED",
            r"This\s+(?:workflow|gate|system|module|document)\s+.*SHADOW-GATED",
        ]

        # Check for declarations (not just mentions)
        is_gated = any(
            re.search(p, content, re.IGNORECASE | re.MULTILINE)
            for p in gated_declaration_patterns
        )

        # Similarly for SHADOW-OBSERVE
        observe_declaration_patterns = [
            r"\*\*Mode\*\*:\s*SHADOW-OBSERVE",
            r"\*\*SHADOW MODE\*\*:\s*SHADOW-OBSERVE",
            r"Mode:\s*SHADOW-OBSERVE",
            r"SHADOW MODE:\s*SHADOW-OBSERVE",
            r"sub-mode:\s*SHADOW-OBSERVE",
            r"operates\s+(?:in\s+)?SHADOW-OBSERVE",
            r"This\s+(?:workflow|gate|system|module|document)\s+.*SHADOW-OBSERVE",
        ]

        is_observe = any(
            re.search(p, content, re.IGNORECASE | re.MULTILINE)
            for p in observe_declaration_patterns
        )

        return is_gated, is_observe

    def _has_gate_registry_reference(self, content: str) -> bool:
        """Check if document references gate registry."""
        return any(re.search(p, content, re.IGNORECASE) for p in GATE_REGISTRY_PATTERNS)

    def _find_prohibited_phrases(
        self,
        content: str,
        lines: List[str],
        file_path: str
    ) -> List[GateViolation]:
        """Find prohibited phrases in SHADOW-GATED context."""
        violations = []

        for phrase in PROHIBITED_PHRASES_IN_GATED:
            # Case-insensitive search
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)

            for i, line in enumerate(lines):
                if pattern.search(line):
                    # Check if this is in an allowed technical context
                    line_lower = line.lower()
                    in_allowed_context = any(
                        allowed.lower() in line_lower
                        for allowed in ALLOWED_TECHNICAL_PHRASES
                    )

                    # Check if phrase is quoted (meta-reference, not usage)
                    # Patterns: "phrase", 'phrase', `phrase`, ``phrase``
                    quoted_patterns = [
                        rf'"{re.escape(phrase)}"',
                        rf"'{re.escape(phrase)}'",
                        rf'`{re.escape(phrase)}`',
                        rf'``{re.escape(phrase)}``',
                        rf'\*\*{re.escape(phrase)}\*\*',  # **phrase**
                    ]
                    in_quoted_context = any(
                        re.search(qp, line, re.IGNORECASE)
                        for qp in quoted_patterns
                    )

                    # Check if line is describing a replacement/removal
                    replacement_indicators = [
                        "replaced",
                        "removed",
                        "prohibited phrase",
                        "forbidden phrase",
                        "canonical replacement",
                        "prohibited language",
                        "forbidden language",
                    ]
                    in_replacement_context = any(
                        ind in line_lower for ind in replacement_indicators
                    )

                    if not (in_allowed_context or in_quoted_context or in_replacement_context):
                        violations.append(GateViolation(
                            file_path=file_path,
                            line_number=i + 1,
                            violation_type="PROHIBITED_PHRASE_IN_GATED",
                            message=f"Prohibited phrase '{phrase}' in SHADOW-GATED context",
                            context=line.strip()[:200],
                            severity="ERROR",
                        ))

        return violations

    def _find_unqualified_shadow(
        self,
        content: str,
        lines: List[str],
        file_path: str
    ) -> List[GateViolation]:
        """Find unqualified SHADOW MODE usage."""
        violations = []
        pattern = re.compile(UNQUALIFIED_SHADOW_PATTERN)

        for i, line in enumerate(lines):
            if pattern.search(line):
                # Exclude lines that are just referencing the contract
                if "SHADOW_MODE_CONTRACT" in line or "SHADOW MODE CONTRACT" in line:
                    continue
                # Exclude table headers or definitions
                if "|" in line and ("SHADOW-OBSERVE" in line or "SHADOW-GATED" in line):
                    continue

                violations.append(GateViolation(
                    file_path=file_path,
                    line_number=i + 1,
                    violation_type="UNQUALIFIED_SHADOW_MODE",
                    message="Unqualified 'SHADOW MODE' usage; must specify SHADOW-OBSERVE or SHADOW-GATED",
                    context=line.strip()[:200],
                    severity="WARN",
                ))

        return violations

    def _check_gated_registry_requirement(
        self,
        content: str,
        file_path: str
    ) -> List[GateViolation]:
        """Check that SHADOW-GATED documents reference gate registry."""
        violations = []

        is_gated, _ = self._detect_context(content)

        if is_gated and not self._has_gate_registry_reference(content):
            violations.append(GateViolation(
                file_path=file_path,
                line_number=1,
                violation_type="MISSING_GATE_REGISTRY",
                message="SHADOW-GATED document must reference gate registry (gate_id required)",
                context="Document declares SHADOW-GATED but lacks gate registry reference",
                severity="ERROR",
            ))

        return violations

    def scan_file(self, file_path: Path) -> List[GateViolation]:
        """
        Scan a single file for SHADOW MODE violations.

        Only scans .md files and certain config files.
        """
        violations = []

        # Only scan relevant file types
        if file_path.suffix not in [".md", ".yaml", ".yml", ".json"]:
            return violations

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return violations

        lines = content.split("\n")
        str_path = str(file_path)

        # Detect context
        is_gated, is_observe = self._detect_context(content)

        # If document declares SHADOW-GATED, check for prohibited phrases
        if is_gated:
            violations.extend(self._find_prohibited_phrases(content, lines, str_path))
            violations.extend(self._check_gated_registry_requirement(content, str_path))

        # Check for unqualified SHADOW MODE in any document with SHADOW references
        if "SHADOW" in content.upper():
            violations.extend(self._find_unqualified_shadow(content, lines, str_path))

        return violations

    def scan_directory(
        self,
        dir_path: Path,
        exclude_patterns: Optional[List[str]] = None
    ) -> GateReport:
        """
        Scan a directory recursively for SHADOW MODE violations.

        Args:
            dir_path: Directory to scan
            exclude_patterns: Glob patterns to exclude (e.g., ["**/node_modules/**"])
        """
        report = GateReport(scan_path=str(dir_path))
        exclude_patterns = exclude_patterns or []

        # Default exclusions
        default_excludes = [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/.venv/**",
            "**/__pycache__/**",
            "**/results/**",
            "**/build/**",
            "**/dist/**",
        ]
        all_excludes = set(default_excludes + exclude_patterns)

        for root, dirs, files in os.walk(dir_path):
            root_path = Path(root)

            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                root_path.joinpath(d).match(p) for p in all_excludes
            )]

            for file_name in files:
                file_path = root_path / file_name

                # Skip excluded files
                if any(file_path.match(p) for p in all_excludes):
                    continue

                violations = self.scan_file(file_path)
                report.files_scanned += 1

                for v in violations:
                    report.add_violation(v)

        return report


# =============================================================================
# PUBLIC API
# =============================================================================

def scan_file(file_path: str | Path) -> List[GateViolation]:
    """Scan a single file for violations."""
    gate = ShadowReleaseGate()
    return gate.scan_file(Path(file_path))


def scan_directory(dir_path: str | Path, exclude_patterns: Optional[List[str]] = None) -> GateReport:
    """Scan a directory for violations."""
    gate = ShadowReleaseGate()
    return gate.scan_directory(Path(dir_path), exclude_patterns)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Shadow Release Gate — SHADOW MODE contract enforcement"
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        help="Directory to scan recursively",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Single file to scan",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        action="append",
        default=[],
        help="Glob patterns to exclude",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON report to file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout output (still writes to --output if specified)",
    )

    args = parser.parse_args()

    if not args.scan_dir and not args.file:
        parser.error("Must specify --scan-dir or --file")

    gate = ShadowReleaseGate()

    if args.file:
        violations = gate.scan_file(Path(args.file))
        report = GateReport(scan_path=args.file, files_scanned=1)
        for v in violations:
            report.add_violation(v)
    else:
        report = gate.scan_directory(Path(args.scan_dir), args.exclude)

    # Output
    if args.output:
        Path(args.output).write_text(report.to_json(), encoding="utf-8")

    if not args.quiet:
        # Use ASCII-safe output for cross-platform compatibility
        print(f"\n{'='*60}")
        print("SHADOW RELEASE GATE REPORT")
        print(f"{'='*60}")
        print(f"Scan path: {report.scan_path}")
        print(f"Files scanned: {report.files_scanned}")
        print(f"Violations: {len(report.violations)}")
        print(f"Gate ID: {report.gate_id}")
        print(f"Contract version: {report.contract_version}")
        print()

        if report.violations:
            print("VIOLATIONS:")
            print("-" * 40)
            for v in report.violations:
                print(f"[{v.severity}] {v.file_path}:{v.line_number}")
                print(f"  Type: {v.violation_type}")
                print(f"  Message: {v.message}")
                # Sanitize context for ASCII output
                safe_context = v.context[:100].encode('ascii', 'replace').decode('ascii')
                print(f"  Context: {safe_context}...")
                print()

        print(f"{'='*60}")
        if report.passed:
            print("VERDICT: PASS")
        else:
            print("VERDICT: FAIL")
        print(f"{'='*60}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
