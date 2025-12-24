#!/usr/bin/env python3
"""
Manifest Verifier for Phase II Uplift Experiments

This module provides cryptographic + structural integrity checks on Phase II
experiment manifests. It does NOT interpret uplift; it verifies that artifacts
are internally and externally consistent.

Exit Codes (when used as CLI):
    0 - PASS: All checks OK
    1 - FAIL: Structural/cryptographic failure
    2 - MISSING: Missing or ambiguous artifacts

Phase II only - does not touch Phase I attestation paths.

SOBER TRUTH GUARDRAILS:
- Do NOT modify manifests or logs; this is a read-only auditor
- Do NOT compute uplift or p-values
- Do NOT interpret audit findings as uplift evidence
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VerificationResult:
    """Result of a verification check."""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ManifestVerificationReport:
    """Complete verification report for a manifest."""
    manifest_path: str
    results: List[VerificationResult] = field(default_factory=list)
    overall_status: str = "unknown"  # "PASS", "FAIL", "MISSING"
    
    def all_passed(self) -> bool:
        """Check if all verification results passed."""
        return all(r.passed for r in self.results)
    
    def has_missing(self) -> bool:
        """Check if any result indicates missing artifacts."""
        return any(
            "missing" in r.message.lower() or "not found" in r.message.lower()
            for r in self.results if not r.passed
        )
    
    def compute_overall_status(self) -> str:
        """Compute overall status based on results."""
        if self.all_passed():
            self.overall_status = "PASS"
        elif self.has_missing():
            self.overall_status = "MISSING"
        else:
            self.overall_status = "FAIL"
        return self.overall_status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "manifest_path": self.manifest_path,
            "overall_status": self.overall_status,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ]
        }


class ManifestVerifier:
    """
    Verifier for Phase II experiment manifests.
    
    Performs structural and cryptographic integrity checks including:
    - Manifest JSON parsing and schema validation
    - Cycle count consistency between manifest and logs
    - Hₜ series cross-checks
    - Hash verification for referenced artifacts
    
    This is a read-only auditor; it does NOT modify any files.
    """
    
    def __init__(self, manifest_path: Path, experiment_dir: Optional[Path] = None):
        """
        Initialize the manifest verifier.
        
        Args:
            manifest_path: Path to the manifest JSON file
            experiment_dir: Optional path to the experiment directory.
                           If not provided, inferred from manifest_path parent.
        """
        self.manifest_path = Path(manifest_path)
        self.experiment_dir = experiment_dir or self.manifest_path.parent
        self.manifest: Optional[Dict[str, Any]] = None
        self.report = ManifestVerificationReport(manifest_path=str(self.manifest_path))
    
    def load_manifest(self) -> bool:
        """
        Load and parse the manifest JSON file.
        
        Returns:
            True if manifest loaded successfully, False otherwise.
        """
        if not self.manifest_path.exists():
            self.report.results.append(VerificationResult(
                check_name="manifest_exists",
                passed=False,
                message=f"Manifest file not found: {self.manifest_path}"
            ))
            return False
        
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                self.manifest = json.load(f)
            self.report.results.append(VerificationResult(
                check_name="manifest_parse",
                passed=True,
                message="Manifest parsed successfully"
            ))
            return True
        except json.JSONDecodeError as e:
            self.report.results.append(VerificationResult(
                check_name="manifest_parse",
                passed=False,
                message=f"Invalid JSON in manifest: {e}"
            ))
            return False
    
    def _count_jsonl_records(self, filepath: Path) -> Optional[int]:
        """
        Count the number of valid JSON records in a JSONL file.
        
        Args:
            filepath: Path to the JSONL file
            
        Returns:
            Number of valid records, or None if file cannot be read.
        """
        if not filepath.exists():
            return None
        
        count = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)
                            count += 1
                        except json.JSONDecodeError:
                            # Skip malformed lines without counting them
                            pass
        except (IOError, OSError):
            return None
        
        return count
    
    def _compute_file_hash(self, filepath: Path) -> Optional[str]:
        """
        Compute SHA256 hash of a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Hex-encoded SHA256 hash, or None if file cannot be read.
        """
        if not filepath.exists():
            return None
        
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except (IOError, OSError):
            return None
    
    def verify_cycle_count(self) -> VerificationResult:
        """
        Verify that manifest cycle counts match actual log record counts.
        
        Checks manifest["n_cycles"]["baseline"] against baseline log line count
        and manifest["n_cycles"]["rfl"] against RFL log line count.
        
        Returns:
            VerificationResult with pass/fail status and details.
        """
        if not self.manifest:
            return VerificationResult(
                check_name="cycle_count",
                passed=False,
                message="Manifest not loaded"
            )
        
        n_cycles = self.manifest.get("n_cycles", {})
        outputs = self.manifest.get("outputs", {})
        details = {"checks": []}
        all_passed = True
        messages = []
        
        # Check baseline cycle count
        baseline_expected = n_cycles.get("baseline")
        baseline_log_path = outputs.get("baseline_log") or outputs.get("results_baseline")
        
        if baseline_expected is not None and baseline_log_path:
            log_path = self._resolve_path(baseline_log_path)
            actual_count = self._count_jsonl_records(log_path)
            
            if actual_count is None:
                all_passed = False
                messages.append(f"Baseline log not found: {baseline_log_path}")
                details["checks"].append({
                    "type": "baseline",
                    "expected": baseline_expected,
                    "actual": None,
                    "log_path": str(baseline_log_path),
                    "status": "missing"
                })
            elif actual_count != baseline_expected:
                all_passed = False
                messages.append(
                    f"Baseline cycle count mismatch: manifest={baseline_expected}, "
                    f"actual={actual_count}"
                )
                details["checks"].append({
                    "type": "baseline",
                    "expected": baseline_expected,
                    "actual": actual_count,
                    "log_path": str(baseline_log_path),
                    "status": "mismatch"
                })
            else:
                details["checks"].append({
                    "type": "baseline",
                    "expected": baseline_expected,
                    "actual": actual_count,
                    "log_path": str(baseline_log_path),
                    "status": "match"
                })
        
        # Check RFL cycle count
        rfl_expected = n_cycles.get("rfl")
        rfl_log_path = outputs.get("rfl_log") or outputs.get("results_rfl")
        
        if rfl_expected is not None and rfl_log_path:
            log_path = self._resolve_path(rfl_log_path)
            actual_count = self._count_jsonl_records(log_path)
            
            if actual_count is None:
                all_passed = False
                messages.append(f"RFL log not found: {rfl_log_path}")
                details["checks"].append({
                    "type": "rfl",
                    "expected": rfl_expected,
                    "actual": None,
                    "log_path": str(rfl_log_path),
                    "status": "missing"
                })
            elif actual_count != rfl_expected:
                all_passed = False
                messages.append(
                    f"RFL cycle count mismatch: manifest={rfl_expected}, "
                    f"actual={actual_count}"
                )
                details["checks"].append({
                    "type": "rfl",
                    "expected": rfl_expected,
                    "actual": actual_count,
                    "log_path": str(rfl_log_path),
                    "status": "mismatch"
                })
            else:
                details["checks"].append({
                    "type": "rfl",
                    "expected": rfl_expected,
                    "actual": actual_count,
                    "log_path": str(rfl_log_path),
                    "status": "match"
                })
        
        # Also check "cycles" field for simpler manifests
        cycles = self.manifest.get("cycles")
        results_path = outputs.get("results")
        
        if cycles is not None and results_path and not n_cycles:
            log_path = self._resolve_path(results_path)
            actual_count = self._count_jsonl_records(log_path)
            
            if actual_count is None:
                all_passed = False
                messages.append(f"Results log not found: {results_path}")
                details["checks"].append({
                    "type": "results",
                    "expected": cycles,
                    "actual": None,
                    "log_path": str(results_path),
                    "status": "missing"
                })
            elif actual_count != cycles:
                all_passed = False
                messages.append(
                    f"Cycle count mismatch: manifest={cycles}, actual={actual_count}"
                )
                details["checks"].append({
                    "type": "results",
                    "expected": cycles,
                    "actual": actual_count,
                    "log_path": str(results_path),
                    "status": "mismatch"
                })
            else:
                details["checks"].append({
                    "type": "results",
                    "expected": cycles,
                    "actual": actual_count,
                    "log_path": str(results_path),
                    "status": "match"
                })
        
        if not details["checks"]:
            return VerificationResult(
                check_name="cycle_count",
                passed=True,
                message="No cycle count fields to verify in manifest",
                details={"skipped": True}
            )
        
        if all_passed:
            return VerificationResult(
                check_name="cycle_count",
                passed=True,
                message="All cycle counts match",
                details=details
            )
        else:
            return VerificationResult(
                check_name="cycle_count",
                passed=False,
                message="; ".join(messages),
                details=details
            )
    
    def _resolve_path(self, path_str: str) -> Path:
        """
        Resolve a path string, handling relative paths from experiment dir.
        
        Args:
            path_str: Path string from manifest
            
        Returns:
            Resolved Path object
        """
        path = Path(path_str)
        if path.is_absolute():
            return path
        # Try relative to experiment dir first
        exp_relative = self.experiment_dir / path
        if exp_relative.exists():
            return exp_relative
        # Try relative to current directory
        return path
    
    def verify_ht_series(self) -> VerificationResult:
        """
        Verify Hₜ series consistency.
        
        Checks if ht_series.json exists in the experiment directory and verifies:
        - Length matches both logs and manifest
        - First/last H_t entries match manifest ht_series fields (if present)
        
        This is a read-only, non-interpretive check.
        
        Returns:
            VerificationResult with pass/fail status and details.
        """
        if not self.manifest:
            return VerificationResult(
                check_name="ht_series",
                passed=False,
                message="Manifest not loaded"
            )
        
        # Look for ht_series.json in experiment directory
        ht_series_path = self.experiment_dir / "ht_series.json"
        
        if not ht_series_path.exists():
            # Also check if manifest specifies an ht_series path
            outputs = self.manifest.get("outputs", {})
            ht_path_str = outputs.get("ht_series")
            if ht_path_str:
                ht_series_path = self._resolve_path(ht_path_str)
        
        if not ht_series_path.exists():
            # Check if manifest has ht_series info that should be verified
            manifest_ht = self.manifest.get("ht_series", {})
            if manifest_ht:
                return VerificationResult(
                    check_name="ht_series",
                    passed=False,
                    message=f"ht_series.json not found but manifest declares ht_series data",
                    details={"expected_path": str(ht_series_path), "manifest_ht_series": manifest_ht}
                )
            return VerificationResult(
                check_name="ht_series",
                passed=True,
                message="ht_series.json not present (optional check skipped)",
                details={"skipped": True}
            )
        
        # Load and verify ht_series.json
        try:
            with open(ht_series_path, 'r', encoding='utf-8') as f:
                ht_series = json.load(f)
        except json.JSONDecodeError as e:
            return VerificationResult(
                check_name="ht_series",
                passed=False,
                message=f"Invalid JSON in ht_series.json: {e}",
                details={"path": str(ht_series_path)}
            )
        
        # Determine ht_series structure (list or dict with entries)
        if isinstance(ht_series, list):
            ht_entries = ht_series
        elif isinstance(ht_series, dict) and "entries" in ht_series:
            ht_entries = ht_series["entries"]
        else:
            ht_entries = []
        
        ht_length = len(ht_entries)
        details = {"ht_series_path": str(ht_series_path), "ht_series_length": ht_length}
        messages = []
        all_passed = True
        
        # Check length matches manifest cycles
        n_cycles = self.manifest.get("n_cycles", {})
        cycles = self.manifest.get("cycles")
        
        expected_length = None
        if n_cycles.get("baseline"):
            expected_length = n_cycles["baseline"]
        elif n_cycles.get("rfl"):
            expected_length = n_cycles["rfl"]
        elif cycles:
            expected_length = cycles
        
        if expected_length is not None:
            if ht_length != expected_length:
                all_passed = False
                messages.append(
                    f"ht_series length ({ht_length}) does not match "
                    f"manifest cycles ({expected_length})"
                )
                details["expected_length"] = expected_length
                details["length_match"] = False
            else:
                details["expected_length"] = expected_length
                details["length_match"] = True
        
        # Check first/last H_t entries against manifest
        manifest_ht = self.manifest.get("ht_series", {})
        
        if manifest_ht.get("ht_first") and ht_entries:
            actual_first = ht_entries[0] if isinstance(ht_entries[0], str) else ht_entries[0].get("h_t")
            expected_first = manifest_ht["ht_first"]
            if actual_first != expected_first:
                all_passed = False
                messages.append(
                    f"ht_series first entry mismatch: expected={expected_first}, "
                    f"actual={actual_first}"
                )
                details["ht_first_match"] = False
                details["ht_first_expected"] = expected_first
                details["ht_first_actual"] = actual_first
            else:
                details["ht_first_match"] = True
        
        if manifest_ht.get("ht_last") and ht_entries:
            actual_last = ht_entries[-1] if isinstance(ht_entries[-1], str) else ht_entries[-1].get("h_t")
            expected_last = manifest_ht["ht_last"]
            if actual_last != expected_last:
                all_passed = False
                messages.append(
                    f"ht_series last entry mismatch: expected={expected_last}, "
                    f"actual={actual_last}"
                )
                details["ht_last_match"] = False
                details["ht_last_expected"] = expected_last
                details["ht_last_actual"] = actual_last
            else:
                details["ht_last_match"] = True
        
        if all_passed:
            return VerificationResult(
                check_name="ht_series",
                passed=True,
                message=f"ht_series.json verified ({ht_length} entries)",
                details=details
            )
        else:
            return VerificationResult(
                check_name="ht_series",
                passed=False,
                message="; ".join(messages),
                details=details
            )
    
    def verify_label_constraint(self) -> VerificationResult:
        """
        Verify that Phase II label constraints are satisfied.
        
        Checks that manifest contains appropriate Phase II labels/markers.
        
        Returns:
            VerificationResult with pass/fail status.
        """
        if not self.manifest:
            return VerificationResult(
                check_name="label_constraint",
                passed=False,
                message="Manifest not loaded"
            )
        
        # Check for Phase II label
        label = self.manifest.get("label", "")
        phase = self.manifest.get("phase", "")
        governance_phase = self.manifest.get("governance", {}).get("phase", "")
        
        is_phase_ii = (
            "PHASE II" in label or
            "phase_ii" in label.lower() or
            phase == "II" or
            phase == "2" or
            governance_phase == "II"
        )
        
        details = {
            "label": label,
            "phase": phase,
            "governance_phase": governance_phase
        }
        
        if is_phase_ii:
            return VerificationResult(
                check_name="label_constraint",
                passed=True,
                message="Phase II label constraint satisfied",
                details=details
            )
        else:
            return VerificationResult(
                check_name="label_constraint",
                passed=False,
                message="Missing or invalid Phase II label/marker",
                details=details
            )
    
    def verify_binding(self) -> VerificationResult:
        """
        Verify manifest binding to preregistration.
        
        Checks that prereg_hash or similar binding field is present.
        
        Returns:
            VerificationResult with pass/fail status.
        """
        if not self.manifest:
            return VerificationResult(
                check_name="binding",
                passed=False,
                message="Manifest not loaded"
            )
        
        prereg_hash = self.manifest.get("prereg_hash")
        slice_config_hash = self.manifest.get("slice_config_hash")
        
        details = {
            "prereg_hash": prereg_hash,
            "slice_config_hash": slice_config_hash
        }
        
        if prereg_hash or slice_config_hash:
            return VerificationResult(
                check_name="binding",
                passed=True,
                message="Manifest has binding hash(es)",
                details=details
            )
        else:
            return VerificationResult(
                check_name="binding",
                passed=False,
                message="No prereg_hash or slice_config_hash found in manifest",
                details=details
            )
    
    def verify_artifact_hashes(self) -> VerificationResult:
        """
        Verify SHA256 hashes of artifacts declared in manifest.
        
        Checks that artifacts listed in manifest["artifacts"] have matching
        SHA256 hashes to their actual file contents.
        
        Returns:
            VerificationResult with pass/fail status and details.
        """
        if not self.manifest:
            return VerificationResult(
                check_name="artifact_hashes",
                passed=False,
                message="Manifest not loaded"
            )
        
        artifacts = self.manifest.get("artifacts", {})
        logs = artifacts.get("logs", [])
        figures = artifacts.get("figures", [])
        
        if not logs and not figures:
            return VerificationResult(
                check_name="artifact_hashes",
                passed=True,
                message="No artifacts with declared hashes to verify",
                details={"skipped": True}
            )
        
        all_passed = True
        messages = []
        details = {"checks": []}
        
        for artifact_list, artifact_type in [(logs, "log"), (figures, "figure")]:
            for artifact in artifact_list:
                artifact_path = artifact.get("path")
                declared_hash = artifact.get("sha256")
                
                if not artifact_path or not declared_hash:
                    continue
                
                resolved_path = self._resolve_path(artifact_path)
                actual_hash = self._compute_file_hash(resolved_path)
                
                check_result = {
                    "type": artifact_type,
                    "path": artifact_path,
                    "declared_hash": declared_hash,
                    "actual_hash": actual_hash
                }
                
                if actual_hash is None:
                    all_passed = False
                    check_result["status"] = "missing"
                    messages.append(f"{artifact_type} not found: {artifact_path}")
                elif actual_hash != declared_hash:
                    all_passed = False
                    check_result["status"] = "mismatch"
                    messages.append(
                        f"{artifact_type} hash mismatch: {artifact_path} "
                        f"(declared={declared_hash[:16]}..., actual={actual_hash[:16]}...)"
                    )
                else:
                    check_result["status"] = "match"
                
                details["checks"].append(check_result)
        
        if not details["checks"]:
            return VerificationResult(
                check_name="artifact_hashes",
                passed=True,
                message="No artifacts with declared hashes to verify",
                details={"skipped": True}
            )
        
        if all_passed:
            return VerificationResult(
                check_name="artifact_hashes",
                passed=True,
                message=f"All {len(details['checks'])} artifact hashes verified",
                details=details
            )
        else:
            return VerificationResult(
                check_name="artifact_hashes",
                passed=False,
                message="; ".join(messages),
                details=details
            )
    
    def verify_all(self) -> ManifestVerificationReport:
        """
        Run all verification checks and return complete report.
        
        Returns:
            ManifestVerificationReport with all results.
        """
        # Load manifest first
        if not self.load_manifest():
            self.report.compute_overall_status()
            return self.report
        
        # Run all checks
        self.report.results.append(self.verify_label_constraint())
        self.report.results.append(self.verify_binding())
        self.report.results.append(self.verify_cycle_count())
        self.report.results.append(self.verify_ht_series())
        self.report.results.append(self.verify_artifact_hashes())
        
        # Compute overall status
        self.report.compute_overall_status()
        
        return self.report


def verify_manifest_file(manifest_path: Path) -> ManifestVerificationReport:
    """
    Convenience function to verify a single manifest file.
    
    Args:
        manifest_path: Path to the manifest JSON file
        
    Returns:
        ManifestVerificationReport with all results.
    """
    verifier = ManifestVerifier(manifest_path)
    return verifier.verify_all()
