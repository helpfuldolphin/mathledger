#!/usr/bin/env python3
"""
PHASE II — NOT USED IN PHASE I

Manifest Verifier Module
========================

Verifies manifest integrity for U2 uplift experiments. Validates:
- Slice config hash matches declared hash in manifest
- Preregistration hash matches declared hash
- Hₜ series hash integrity (if present)
- PHASE II labelling and required fields

This module does NOT:
- Perform uplift significance testing
- Interpret experimental outcomes
- Modify any attestation files

Usage:
    from experiments.manifest_verifier import ManifestVerifier
    verifier = ManifestVerifier(manifest_path)
    result = verifier.validate_all()
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


@dataclass
class VerificationReport:
    """Complete verification report for a manifest."""
    manifest_path: str
    checks: List[CheckResult] = field(default_factory=list)
    overall_pass: bool = True
    raw_counts: Dict[str, int] = field(default_factory=dict)

    def add_check(self, check: CheckResult) -> None:
        self.checks.append(check)
        if not check.passed:
            self.overall_pass = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dict."""
        return {
            "manifest_path": self.manifest_path,
            "overall_pass": self.overall_pass,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "expected": c.expected,
                    "actual": c.actual,
                }
                for c in self.checks
            ],
            "raw_counts": self.raw_counts,
        }

    def to_markdown(self) -> str:
        """Convert report to Markdown format."""
        lines = [
            "# Manifest Verification Report",
            "",
            f"**Manifest:** `{self.manifest_path}`",
            "",
            f"**Overall Result:** {'✅ PASS' if self.overall_pass else '❌ FAIL'}",
            "",
            "## Check Results",
            "",
            "| Check | Status | Message |",
            "|-------|--------|---------|",
        ]
        for check in self.checks:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            lines.append(f"| {check.name} | {status} | {check.message} |")

        if self.raw_counts:
            lines.extend([
                "",
                "## Raw Counts",
                "",
            ])
            for key, value in self.raw_counts.items():
                lines.append(f"- **{key}:** {value}")

        return "\n".join(lines)


class ManifestVerifier:
    """
    Verifies manifest integrity for U2 uplift experiments.

    This verifier is strictly read-only and does NOT perform:
    - Uplift significance testing
    - Statistical interpretation
    - Modification of any files
    """

    # Required label for Phase II artifacts
    PHASE_II_LABEL = "PHASE II — NOT USED IN PHASE I"

    def __init__(self, manifest_path: Path, experiment_dir: Optional[Path] = None):
        """
        Initialize the verifier.

        Args:
            manifest_path: Path to the manifest JSON file
            experiment_dir: Optional base directory for relative paths in manifest
        """
        self.manifest_path = Path(manifest_path)
        self.experiment_dir = experiment_dir or self.manifest_path.parent
        self.manifest: Dict[str, Any] = {}
        self.report = VerificationReport(manifest_path=str(self.manifest_path))

    def _hash_file(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _hash_string(self, data: str) -> str:
        """Compute SHA256 hash of a string."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _load_manifest(self) -> CheckResult:
        """Load and parse the manifest JSON file."""
        if not self.manifest_path.exists():
            return CheckResult(
                name="manifest_exists",
                passed=False,
                message=f"Manifest file not found: {self.manifest_path}",
            )

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
            return CheckResult(
                name="manifest_exists",
                passed=True,
                message="Manifest file loaded successfully",
            )
        except json.JSONDecodeError as e:
            return CheckResult(
                name="manifest_exists",
                passed=False,
                message=f"Invalid JSON in manifest: {e}",
            )
        except Exception as e:
            return CheckResult(
                name="manifest_exists",
                passed=False,
                message=f"Error loading manifest: {e}",
            )

    def _check_phase_ii_label(self) -> CheckResult:
        """Verify PHASE II label is present in manifest."""
        label = self.manifest.get("label", "")
        if self.PHASE_II_LABEL in label:
            return CheckResult(
                name="phase_ii_label",
                passed=True,
                message="PHASE II label present",
                expected=self.PHASE_II_LABEL,
                actual=label,
            )
        return CheckResult(
            name="phase_ii_label",
            passed=False,
            message="Missing or incorrect PHASE II label",
            expected=self.PHASE_II_LABEL,
            actual=label or "(no label field)",
        )

    def _check_required_fields(self) -> CheckResult:
        """Verify all required manifest fields are present."""
        required_fields = [
            "slice",
            "mode",
            "cycles",
            "initial_seed",
            "slice_config_hash",
            "outputs",
        ]
        missing = [f for f in required_fields if f not in self.manifest]

        if not missing:
            return CheckResult(
                name="required_fields",
                passed=True,
                message="All required fields present",
            )
        return CheckResult(
            name="required_fields",
            passed=False,
            message=f"Missing required fields: {', '.join(missing)}",
        )

    def _check_slice_config_hash(
        self, config_path: Optional[Path] = None
    ) -> CheckResult:
        """Verify slice config hash matches actual config file hash."""
        declared_hash = self.manifest.get("slice_config_hash")
        if not declared_hash:
            return CheckResult(
                name="slice_config_hash",
                passed=False,
                message="No slice_config_hash in manifest",
            )

        if config_path is None:
            # Cannot verify without config file path
            return CheckResult(
                name="slice_config_hash",
                passed=True,
                message=f"Declared hash: {declared_hash[:16]}... (no config path to verify)",
                expected=declared_hash,
                actual="(no config file provided for verification)",
            )

        if not config_path.exists():
            return CheckResult(
                name="slice_config_hash",
                passed=False,
                message=f"Config file not found: {config_path}",
                expected=declared_hash,
            )

        # Load config and compute hash of the entire config file
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_content = f.read()

            actual_hash = self._hash_string(config_content)

            if declared_hash == actual_hash:
                return CheckResult(
                    name="slice_config_hash",
                    passed=True,
                    message="Slice config hash matches",
                    expected=declared_hash,
                    actual=actual_hash,
                )
            else:
                return CheckResult(
                    name="slice_config_hash",
                    passed=False,
                    message="Slice config hash mismatch",
                    expected=declared_hash,
                    actual=actual_hash,
                )
        except Exception as e:
            return CheckResult(
                name="slice_config_hash",
                passed=False,
                message=f"Error reading config: {e}",
                expected=declared_hash,
            )

    def _check_prereg_hash(
        self, prereg_path: Optional[Path] = None
    ) -> CheckResult:
        """Verify preregistration hash if present."""
        declared_hash = self.manifest.get("prereg_hash")
        if not declared_hash or declared_hash == "N/A":
            return CheckResult(
                name="prereg_hash",
                passed=True,
                message="No prereg_hash declared (optional)",
            )

        if prereg_path is None:
            return CheckResult(
                name="prereg_hash",
                passed=True,
                message=f"Declared hash: {declared_hash[:16]}... (no prereg path to verify)",
                expected=declared_hash,
                actual="(no prereg file provided for verification)",
            )

        if not prereg_path.exists():
            return CheckResult(
                name="prereg_hash",
                passed=False,
                message=f"Preregistration file not found: {prereg_path}",
                expected=declared_hash,
            )

        actual_hash = self._hash_file(prereg_path)
        if declared_hash == actual_hash:
            return CheckResult(
                name="prereg_hash",
                passed=True,
                message="Preregistration hash matches",
                expected=declared_hash,
                actual=actual_hash,
            )
        else:
            return CheckResult(
                name="prereg_hash",
                passed=False,
                message="Preregistration hash mismatch",
                expected=declared_hash,
                actual=actual_hash,
            )

    def _check_ht_series_hash(
        self, ht_series_path: Optional[Path] = None
    ) -> CheckResult:
        """Verify Hₜ series hash integrity if present."""
        declared_hash = self.manifest.get("ht_series_hash")
        if not declared_hash:
            return CheckResult(
                name="ht_series_hash",
                passed=True,
                message="No ht_series_hash declared (optional)",
            )

        if ht_series_path is None:
            # Try to find ht_series.json in experiment directory
            ht_series_path = self.experiment_dir / "ht_series.json"

        if not ht_series_path.exists():
            return CheckResult(
                name="ht_series_hash",
                passed=True,
                message=f"Declared hash: {declared_hash[:16]}... (no ht_series file found)",
                expected=declared_hash,
                actual="(ht_series.json not found)",
            )

        try:
            with open(ht_series_path, "r", encoding="utf-8") as f:
                ht_data = json.load(f)
            # Compute hash of the JSON data (sorted keys for determinism)
            ht_str = json.dumps(ht_data, sort_keys=True)
            actual_hash = self._hash_string(ht_str)

            if declared_hash == actual_hash:
                return CheckResult(
                    name="ht_series_hash",
                    passed=True,
                    message="Hₜ series hash matches",
                    expected=declared_hash,
                    actual=actual_hash,
                )
            else:
                return CheckResult(
                    name="ht_series_hash",
                    passed=False,
                    message="Hₜ series hash mismatch",
                    expected=declared_hash,
                    actual=actual_hash,
                )
        except Exception as e:
            return CheckResult(
                name="ht_series_hash",
                passed=False,
                message=f"Error reading ht_series.json: {e}",
                expected=declared_hash,
            )

    def _check_log_file(
        self, log_key: str, log_path: Optional[Path] = None
    ) -> CheckResult:
        """Check that a log file exists and is non-empty."""
        if log_path is None:
            outputs = self.manifest.get("outputs", {})
            log_path_str = outputs.get(log_key)
            if not log_path_str:
                return CheckResult(
                    name=f"{log_key}_exists",
                    passed=False,
                    message=f"No {log_key} path in manifest outputs",
                )
            log_path = Path(log_path_str)
            # Handle relative paths
            if not log_path.is_absolute():
                log_path = self.experiment_dir / log_path

        if not log_path.exists():
            return CheckResult(
                name=f"{log_key}_exists",
                passed=False,
                message=f"Log file not found: {log_path}",
            )

        # Check for non-empty file
        if log_path.stat().st_size == 0:
            return CheckResult(
                name=f"{log_key}_exists",
                passed=False,
                message=f"Log file is empty: {log_path}",
            )

        # Count lines for JSONL files
        line_count = 0
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        line_count += 1
        except Exception as e:
            return CheckResult(
                name=f"{log_key}_exists",
                passed=False,
                message=f"Error reading log file: {e}",
            )

        return CheckResult(
            name=f"{log_key}_exists",
            passed=True,
            message=f"Log file exists with {line_count} records",
        )

    def validate_all(
        self,
        config_path: Optional[Path] = None,
        prereg_path: Optional[Path] = None,
        ht_series_path: Optional[Path] = None,
        baseline_log_path: Optional[Path] = None,
        rfl_log_path: Optional[Path] = None,
    ) -> VerificationReport:
        """
        Run all validation checks.

        Args:
            config_path: Optional path to slice config file
            prereg_path: Optional path to preregistration file
            ht_series_path: Optional path to ht_series.json
            baseline_log_path: Optional path to baseline log
            rfl_log_path: Optional path to RFL log

        Returns:
            VerificationReport with all check results
        """
        # Load manifest first
        load_result = self._load_manifest()
        self.report.add_check(load_result)

        if not load_result.passed:
            return self.report

        # Core checks
        self.report.add_check(self._check_phase_ii_label())
        self.report.add_check(self._check_required_fields())
        self.report.add_check(self._check_slice_config_hash(config_path))
        self.report.add_check(self._check_prereg_hash(prereg_path))
        self.report.add_check(self._check_ht_series_hash(ht_series_path))

        # Log file checks
        if baseline_log_path:
            self.report.add_check(
                self._check_log_file("baseline", baseline_log_path)
            )
        if rfl_log_path:
            self.report.add_check(self._check_log_file("rfl", rfl_log_path))

        # Extract raw counts from manifest
        self.report.raw_counts = {
            "cycles": self.manifest.get("cycles", 0),
            "initial_seed": self.manifest.get("initial_seed", 0),
        }

        return self.report
