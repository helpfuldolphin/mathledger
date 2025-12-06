#!/usr/bin/env python3
"""
PHASE II — NOT RUN IN PHASE I

U2 Manifest Cryptographic Binding Verifier
===========================================

Verifies cryptographic bindings between:
1. Preregistration file (PREREG_UPLIFT_U2.yaml) → manifest → logs
2. Slice config (curriculum_uplift_phase2.yaml) → manifest
3. Hₜ-series integrity from experiment logs

Usage:
    uv run python experiments/manifest_verifier.py --manifest PATH [--prereg PATH] [--config PATH]

This module performs ZERO uplift interpretation. It only validates
cryptographic integrity of experiment artifacts.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ==============================================================================
# PHASE II — NOT RUN IN PHASE I
# ==============================================================================
PHASE_LABEL = "PHASE II — NOT RUN IN PHASE I"


def compute_sha256_file(path: Path) -> str:
    """
    Compute SHA-256 hash of a file's contents.
    
    Returns:
        Hexadecimal SHA-256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_sha256_string(data: str) -> str:
    """
    Compute SHA-256 hash of a string (UTF-8 encoded).
    
    Returns:
        Hexadecimal SHA-256 hash string
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def compute_sha256_canonical_json(obj: Any) -> str:
    """
    Compute SHA-256 hash of JSON object with canonical serialization.
    
    Uses sorted keys and no indentation for deterministic output.
    
    Returns:
        Hexadecimal SHA-256 hash string
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    return compute_sha256_string(canonical)


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load JSON file and return parsed content."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load YAML file and return parsed content (first document only)."""
    with open(path, 'r', encoding='utf-8') as f:
        # Use safe_load for single document, or safe_load_all for multi-doc
        content = f.read()
    
    # Try single document first
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError:
        # If multi-document, load first document only
        docs = list(yaml.safe_load_all(content))
        if docs:
            return docs[0]
        return {}


def load_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ==============================================================================
# Verification Functions
# ==============================================================================

class ManifestVerifier:
    """
    PHASE II — NOT RUN IN PHASE I
    
    Cryptographic binding verifier for U2 experiment manifests.
    Performs ZERO uplift interpretation—only validates integrity.
    """
    
    def __init__(
        self,
        manifest_path: Path,
        prereg_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
    ):
        """
        Initialize verifier with paths to artifacts.
        
        Args:
            manifest_path: Path to experiment manifest JSON
            prereg_path: Path to preregistration YAML (optional)
            config_path: Path to curriculum config YAML (optional)
            logs_dir: Directory containing experiment logs (optional)
        """
        self.manifest_path = manifest_path
        self.prereg_path = prereg_path
        self.config_path = config_path
        self.logs_dir = logs_dir
        
        self.manifest: Optional[Dict[str, Any]] = None
        self.prereg: Optional[Dict[str, Any]] = None
        self.config: Optional[Dict[str, Any]] = None
        
        self.findings: List[Dict[str, Any]] = []
        self.verdict = "PENDING"
        self.phase_label = PHASE_LABEL
    
    def _add_finding(
        self,
        check_name: str,
        status: str,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        message: str = "",
    ) -> None:
        """Add a verification finding."""
        self.findings.append({
            "check": check_name,
            "status": status,  # PASS | FAIL | SKIP | ERROR
            "expected": expected,
            "actual": actual,
            "message": message,
        })
    
    def load_artifacts(self) -> bool:
        """
        Load all artifacts for verification.
        
        Returns:
            True if manifest loaded successfully, False otherwise
        """
        # Manifest is required
        if not self.manifest_path.exists():
            self._add_finding(
                "load_manifest",
                "ERROR",
                message=f"Manifest file not found: {self.manifest_path}"
            )
            return False
        
        try:
            self.manifest = load_json_file(self.manifest_path)
            self._add_finding(
                "load_manifest",
                "PASS",
                message=f"Loaded manifest from {self.manifest_path}"
            )
        except Exception as e:
            self._add_finding(
                "load_manifest",
                "ERROR",
                message=f"Failed to load manifest: {e}"
            )
            return False
        
        # Prereg is optional
        if self.prereg_path:
            if self.prereg_path.exists():
                try:
                    self.prereg = load_yaml_file(self.prereg_path)
                    self._add_finding(
                        "load_prereg",
                        "PASS",
                        message=f"Loaded preregistration from {self.prereg_path}"
                    )
                except Exception as e:
                    self._add_finding(
                        "load_prereg",
                        "ERROR",
                        message=f"Failed to load preregistration: {e}"
                    )
            else:
                self._add_finding(
                    "load_prereg",
                    "SKIP",
                    message=f"Preregistration file not found: {self.prereg_path}"
                )
        
        # Config is optional
        if self.config_path:
            if self.config_path.exists():
                try:
                    self.config = load_yaml_file(self.config_path)
                    self._add_finding(
                        "load_config",
                        "PASS",
                        message=f"Loaded config from {self.config_path}"
                    )
                except Exception as e:
                    self._add_finding(
                        "load_config",
                        "ERROR",
                        message=f"Failed to load config: {e}"
                    )
            else:
                self._add_finding(
                    "load_config",
                    "SKIP",
                    message=f"Config file not found: {self.config_path}"
                )
        
        return True
    
    def verify_prereg_binding(self) -> bool:
        """
        Verify preregistration binding: PREREG_UPLIFT_U2.yaml → manifest.
        
        Computes SHA-256 of prereg file and compares with manifest["prereg_hash"].
        
        Returns:
            True if binding is valid, False otherwise
        """
        if not self.manifest:
            self._add_finding(
                "prereg_binding",
                "SKIP",
                message="No manifest loaded"
            )
            return False
        
        if not self.prereg_path or not self.prereg_path.exists():
            self._add_finding(
                "prereg_binding",
                "SKIP",
                message="No preregistration file available"
            )
            return False
        
        # Get expected hash from manifest
        expected_hash = self.manifest.get("prereg_hash")
        if not expected_hash or expected_hash == "N/A":
            self._add_finding(
                "prereg_binding",
                "SKIP",
                message="Manifest does not contain prereg_hash (or is N/A)"
            )
            return False
        
        # Compute actual hash of prereg file
        actual_hash = compute_sha256_file(self.prereg_path)
        
        if expected_hash == actual_hash:
            self._add_finding(
                "prereg_binding",
                "PASS",
                expected=expected_hash,
                actual=actual_hash,
                message="Preregistration hash matches manifest"
            )
            return True
        else:
            self._add_finding(
                "prereg_binding",
                "FAIL",
                expected=expected_hash,
                actual=actual_hash,
                message="Preregistration hash MISMATCH"
            )
            return False
    
    def verify_slice_config_binding(self) -> bool:
        """
        Verify slice config binding from curriculum_uplift_phase2.yaml.
        
        Loads slice config, computes canonical SHA-256, and compares
        with manifest["slice_config_hash"].
        
        Returns:
            True if binding is valid, False otherwise
        """
        if not self.manifest:
            self._add_finding(
                "slice_config_binding",
                "SKIP",
                message="No manifest loaded"
            )
            return False
        
        if not self.config:
            self._add_finding(
                "slice_config_binding",
                "SKIP",
                message="No config file loaded"
            )
            return False
        
        # Get expected hash from manifest
        expected_hash = self.manifest.get("slice_config_hash")
        if not expected_hash:
            self._add_finding(
                "slice_config_binding",
                "SKIP",
                message="Manifest does not contain slice_config_hash"
            )
            return False
        
        # Get slice name from manifest
        slice_name = self.manifest.get("slice")
        if not slice_name:
            self._add_finding(
                "slice_config_binding",
                "FAIL",
                message="Manifest does not contain slice name"
            )
            return False
        
        # Extract slice config from curriculum config
        slices = self.config.get("slices", {})
        slice_config = slices.get(slice_name)
        
        if not slice_config:
            self._add_finding(
                "slice_config_binding",
                "FAIL",
                message=f"Slice '{slice_name}' not found in config"
            )
            return False
        
        # Compute canonical hash of slice config
        actual_hash = compute_sha256_canonical_json(slice_config)
        
        if expected_hash == actual_hash:
            self._add_finding(
                "slice_config_binding",
                "PASS",
                expected=expected_hash,
                actual=actual_hash,
                message=f"Slice config hash matches for '{slice_name}'"
            )
            return True
        else:
            self._add_finding(
                "slice_config_binding",
                "FAIL",
                expected=expected_hash,
                actual=actual_hash,
                message=f"Slice config hash MISMATCH for '{slice_name}'"
            )
            return False
    
    def verify_ht_series_integrity(self, logs_path: Optional[Path] = None) -> bool:
        """
        Verify Hₜ-series integrity from experiment logs.
        
        Extracts roots.h_t for each cycle from logs, computes SHA-256
        over concatenated Hₜ values, and compares with manifest["h_t_series_hash"]
        or manifest["ht_series_hash"].
        
        Args:
            logs_path: Optional path to log file (overrides logs_dir)
        
        Returns:
            True if integrity is valid, False otherwise
        """
        if not self.manifest:
            self._add_finding(
                "ht_series_integrity",
                "SKIP",
                message="No manifest loaded"
            )
            return False
        
        # Get expected hash from manifest (support both naming conventions)
        expected_hash = self.manifest.get("h_t_series_hash") or self.manifest.get("ht_series_hash")
        if not expected_hash:
            self._add_finding(
                "ht_series_integrity",
                "SKIP",
                message="Manifest does not contain h_t_series_hash or ht_series_hash"
            )
            return False
        
        # Determine log file path
        if logs_path is None:
            # Try to get from manifest outputs
            outputs = self.manifest.get("outputs", {})
            results_path = outputs.get("results")
            if results_path:
                logs_path = Path(results_path)
            elif self.logs_dir:
                # Try to find log file in logs_dir
                slice_name = self.manifest.get("slice", "")
                mode = self.manifest.get("mode", "")
                pattern = f"uplift_u2_{slice_name}_{mode}.jsonl"
                potential_path = self.logs_dir / pattern
                if potential_path.exists():
                    logs_path = potential_path
        
        if logs_path is None or not logs_path.exists():
            self._add_finding(
                "ht_series_integrity",
                "SKIP",
                message=f"Log file not found: {logs_path}"
            )
            return False
        
        # Load logs and extract Hₜ values
        try:
            records = load_jsonl_file(logs_path)
        except Exception as e:
            self._add_finding(
                "ht_series_integrity",
                "ERROR",
                message=f"Failed to load log file: {e}"
            )
            return False
        
        if not records:
            self._add_finding(
                "ht_series_integrity",
                "FAIL",
                message="Log file is empty"
            )
            return False
        
        # Extract Hₜ values (support multiple field names)
        ht_values = []
        for record in records:
            # Try different field names for Hₜ
            h_t = None
            if "roots" in record and isinstance(record["roots"], dict):
                h_t = record["roots"].get("h_t")
            if h_t is None:
                h_t = record.get("h_t")
            if h_t is None:
                h_t = record.get("ht")
            
            if h_t is not None:
                ht_values.append(str(h_t))
        
        if not ht_values:
            # If no h_t field, compute hash over full records for integrity
            # This matches the pattern in run_uplift_u2.py which hashes ht_series
            ht_series_str = json.dumps(records, sort_keys=True)
            actual_hash = compute_sha256_string(ht_series_str)
        else:
            # Concatenate Hₜ values and compute hash
            concatenated = "".join(ht_values)
            actual_hash = compute_sha256_string(concatenated)
        
        if expected_hash == actual_hash:
            self._add_finding(
                "ht_series_integrity",
                "PASS",
                expected=expected_hash,
                actual=actual_hash,
                message=f"Hₜ-series hash matches ({len(records)} records)"
            )
            return True
        else:
            self._add_finding(
                "ht_series_integrity",
                "FAIL",
                expected=expected_hash,
                actual=actual_hash,
                message=f"Hₜ-series hash MISMATCH ({len(records)} records)"
            )
            return False
    
    def verify_all(self, logs_path: Optional[Path] = None) -> str:
        """
        Run all verification checks and determine overall verdict.
        
        Args:
            logs_path: Optional path to log file for Hₜ verification
        
        Returns:
            Overall verdict: "PASS" or "FAIL"
        """
        # Load artifacts first
        if not self.load_artifacts():
            self.verdict = "FAIL"
            return self.verdict
        
        # Run all checks
        results = []
        results.append(self.verify_prereg_binding())
        results.append(self.verify_slice_config_binding())
        results.append(self.verify_ht_series_integrity(logs_path))
        
        # Determine verdict
        # PASS only if all non-skipped checks pass
        # FAIL if any check fails
        failures = [f for f in self.findings if f["status"] == "FAIL"]
        errors = [f for f in self.findings if f["status"] == "ERROR"]
        
        if failures or errors:
            self.verdict = "FAIL"
        else:
            self.verdict = "PASS"
        
        return self.verdict
    
    def generate_json_report(self) -> Dict[str, Any]:
        """
        Generate machine-readable JSON report.
        
        Returns:
            Dictionary containing full verification report
        """
        return {
            "label": self.phase_label,
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "verdict": self.verdict,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "prereg_path": str(self.prereg_path) if self.prereg_path else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "findings": self.findings,
            "summary": {
                "total_checks": len(self.findings),
                "passed": len([f for f in self.findings if f["status"] == "PASS"]),
                "failed": len([f for f in self.findings if f["status"] == "FAIL"]),
                "skipped": len([f for f in self.findings if f["status"] == "SKIP"]),
                "errors": len([f for f in self.findings if f["status"] == "ERROR"]),
            },
            "fail_reasons": [
                f["message"] for f in self.findings 
                if f["status"] in ("FAIL", "ERROR")
            ],
        }
    
    def generate_markdown_report(self) -> str:
        """
        Generate human-readable Markdown report.
        
        Returns:
            Markdown string containing verification report
        """
        lines = [
            f"# U2 Manifest Verification Report",
            "",
            f"**{self.phase_label}**",
            "",
            f"**Verdict: {self.verdict}**",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
            "",
            "## Files",
            "",
            f"- Manifest: `{self.manifest_path}`",
            f"- Preregistration: `{self.prereg_path}`" if self.prereg_path else "- Preregistration: N/A",
            f"- Config: `{self.config_path}`" if self.config_path else "- Config: N/A",
            "",
            "## Verification Results",
            "",
        ]
        
        for finding in self.findings:
            status_icon = {
                "PASS": "✅",
                "FAIL": "❌",
                "SKIP": "⏭️",
                "ERROR": "🔴",
            }.get(finding["status"], "❓")
            
            lines.append(f"### {status_icon} {finding['check']}")
            lines.append("")
            lines.append(f"**Status:** {finding['status']}")
            if finding.get("message"):
                lines.append(f"**Message:** {finding['message']}")
            if finding.get("expected"):
                lines.append(f"**Expected:** `{finding['expected'][:16]}...`")
            if finding.get("actual"):
                lines.append(f"**Actual:** `{finding['actual'][:16]}...`")
            lines.append("")
        
        # Summary
        summary = self.generate_json_report()["summary"]
        lines.extend([
            "## Summary",
            "",
            f"- Total checks: {summary['total_checks']}",
            f"- Passed: {summary['passed']}",
            f"- Failed: {summary['failed']}",
            f"- Skipped: {summary['skipped']}",
            f"- Errors: {summary['errors']}",
            "",
        ])
        
        if summary["failed"] > 0 or summary["errors"] > 0:
            lines.extend([
                "## Failure Reasons",
                "",
            ])
            for finding in self.findings:
                if finding["status"] in ("FAIL", "ERROR"):
                    lines.append(f"- {finding['check']}: {finding['message']}")
            lines.append("")
        
        lines.extend([
            "---",
            "",
            f"*{self.phase_label}*",
        ])
        
        return "\n".join(lines)


def verify_manifest(
    manifest_path: Path,
    prereg_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    logs_path: Optional[Path] = None,
    output_json: Optional[Path] = None,
    output_md: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to verify a manifest and generate reports.
    
    Args:
        manifest_path: Path to manifest JSON
        prereg_path: Path to preregistration YAML
        config_path: Path to curriculum config YAML
        logs_path: Path to experiment logs JSONL
        output_json: Path to write JSON report
        output_md: Path to write Markdown report
    
    Returns:
        Tuple of (verdict, json_report)
    """
    verifier = ManifestVerifier(
        manifest_path=manifest_path,
        prereg_path=prereg_path,
        config_path=config_path,
    )
    
    verdict = verifier.verify_all(logs_path=logs_path)
    json_report = verifier.generate_json_report()
    md_report = verifier.generate_markdown_report()
    
    # Write reports if paths provided
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2)
        print(f"JSON report written to: {output_json}")
    
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(md_report)
        print(f"Markdown report written to: {output_md}")
    
    return verdict, json_report


def main():
    """CLI entry point."""
    print(f"=" * 60)
    print(PHASE_LABEL)
    print(f"=" * 60)
    print()
    
    parser = argparse.ArgumentParser(
        description="Verify cryptographic bindings in U2 experiment manifest. " + PHASE_LABEL,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to experiment manifest JSON file",
    )
    parser.add_argument(
        "--prereg",
        type=str,
        default=None,
        help="Path to preregistration YAML file (default: experiments/prereg/PREREG_UPLIFT_U2.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to curriculum config YAML file (default: config/curriculum_uplift_phase2.yaml)",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default=None,
        help="Path to experiment logs JSONL file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Path to write Markdown report",
    )
    
    args = parser.parse_args()
    
    # Set defaults
    project_root = Path(__file__).resolve().parents[1]
    
    manifest_path = Path(args.manifest)
    prereg_path = Path(args.prereg) if args.prereg else project_root / "experiments" / "prereg" / "PREREG_UPLIFT_U2.yaml"
    config_path = Path(args.config) if args.config else project_root / "config" / "curriculum_uplift_phase2.yaml"
    logs_path = Path(args.logs) if args.logs else None
    output_json = Path(args.output_json) if args.output_json else None
    output_md = Path(args.output_md) if args.output_md else None
    
    verdict, json_report = verify_manifest(
        manifest_path=manifest_path,
        prereg_path=prereg_path,
        config_path=config_path,
        logs_path=logs_path,
        output_json=output_json,
        output_md=output_md,
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for finding in json_report["findings"]:
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "SKIP": "⏭️",
            "ERROR": "🔴",
        }.get(finding["status"], "❓")
        print(f"{status_icon} {finding['check']}: {finding['status']}")
        if finding.get("message"):
            print(f"   {finding['message']}")
    
    print()
    print(f"VERDICT: {verdict}")
    print()
    print(PHASE_LABEL)
    
    # Exit code based on verdict
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
