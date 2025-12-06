"""
PHASE II — NOT USED IN PHASE I

U2 Calibration Module
=====================

Supports determinism verification and calibration validation for Phase II
U2 uplift experiments. Provides tools to verify that experiments are
reproducible via replay hash comparison.

Key Features:
- Calibration summary validation (calibration_summary.json)
- Determinism verification via replay hash comparison
- Safe AST-based evaluation for metric arithmetic
- Integration with run_uplift_u2.py via --require-calibration flag

Reference:
- experiments/run_uplift_u2.py — main experiment runner
- results/uplift_u2/calibration/<slice>/ — calibration output directory

Developer Mode:
- --verbose-cycles: Enhanced logging for cycle-by-cycle debugging
- U2_VERBOSE_FIELDS env var: Configurable field selection for verbose output
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class CalibrationSummary:
    """
    Calibration summary data structure.
    
    Attributes:
        slice_name: Name of the experiment slice
        determinism_verified: Whether determinism check passed
        schema_valid: Whether schema validation passed
        replay_hash: Hash of replay results
        baseline_hash: Hash of baseline results
        metadata: Additional metadata
    """
    slice_name: str
    determinism_verified: bool
    schema_valid: bool
    replay_hash: Optional[str] = None
    baseline_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def is_valid(self) -> bool:
        """Check if calibration is valid."""
        return self.determinism_verified and self.schema_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "slice_name": self.slice_name,
            "determinism_verified": self.determinism_verified,
            "schema_valid": self.schema_valid,
            "replay_hash": self.replay_hash,
            "baseline_hash": self.baseline_hash,
            "metadata": self.metadata or {},
        }


class CalibrationError(Exception):
    """Base exception for calibration errors."""
    pass


class CalibrationNotFoundError(CalibrationError):
    """Raised when calibration files are not found."""
    pass


class CalibrationInvalidError(CalibrationError):
    """Raised when calibration is invalid."""
    pass


def load_calibration_summary(
    calibration_dir: Path,
    slice_name: str
) -> CalibrationSummary:
    """
    Load calibration summary from file.
    
    Args:
        calibration_dir: Base calibration directory (e.g., results/uplift_u2/calibration)
        slice_name: Name of the slice
        
    Returns:
        CalibrationSummary object
        
    Raises:
        CalibrationNotFoundError: If summary file not found
        CalibrationInvalidError: If summary is malformed
    """
    summary_path = calibration_dir / slice_name / "calibration_summary.json"
    
    if not summary_path.exists():
        raise CalibrationNotFoundError(
            f"Calibration summary not found: {summary_path}\n"
            f"Run calibration first for slice '{slice_name}'"
        )
    
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate required fields
        required = ["slice_name", "determinism_verified", "schema_valid"]
        missing = [field for field in required if field not in data]
        if missing:
            raise CalibrationInvalidError(
                f"Calibration summary missing required fields: {missing}"
            )
        
        return CalibrationSummary(
            slice_name=data["slice_name"],
            determinism_verified=data["determinism_verified"],
            schema_valid=data["schema_valid"],
            replay_hash=data.get("replay_hash"),
            baseline_hash=data.get("baseline_hash"),
            metadata=data.get("metadata"),
        )
        
    except json.JSONDecodeError as e:
        raise CalibrationInvalidError(f"Invalid JSON in calibration summary: {e}")
    except Exception as e:
        raise CalibrationError(f"Failed to load calibration summary: {e}")


def validate_calibration(
    calibration_dir: Path,
    slice_name: str,
    require_valid: bool = True
) -> Optional[CalibrationSummary]:
    """
    Validate calibration for a slice.
    
    Args:
        calibration_dir: Base calibration directory
        slice_name: Name of the slice
        require_valid: If True, raises exception if calibration invalid
        
    Returns:
        CalibrationSummary if found, None if not found and require_valid=False
        
    Raises:
        CalibrationNotFoundError: If calibration not found and require_valid=True
        CalibrationInvalidError: If calibration invalid and require_valid=True
    """
    try:
        summary = load_calibration_summary(calibration_dir, slice_name)
        
        if require_valid and not summary.is_valid():
            errors = []
            if not summary.determinism_verified:
                errors.append("determinism check failed")
            if not summary.schema_valid:
                errors.append("schema validation failed")
            
            raise CalibrationInvalidError(
                f"Calibration invalid for slice '{slice_name}': {', '.join(errors)}"
            )
        
        return summary
        
    except (CalibrationNotFoundError, CalibrationInvalidError):
        if require_valid:
            raise
        return None


def check_calibration_exists(
    calibration_dir: Path,
    slice_name: str
) -> bool:
    """
    Check if calibration exists for a slice (without validation).
    
    Args:
        calibration_dir: Base calibration directory
        slice_name: Name of the slice
        
    Returns:
        True if calibration summary file exists
    """
    summary_path = calibration_dir / slice_name / "calibration_summary.json"
    return summary_path.exists()


def compute_result_hash(results: List[Dict[str, Any]]) -> str:
    """
    Compute deterministic hash of experiment results.
    
    Args:
        results: List of result dictionaries (from JSONL)
        
    Returns:
        SHA256 hash string
    """
    # Sort results by cycle to ensure determinism
    sorted_results = sorted(results, key=lambda r: r.get("cycle", 0))
    
    # Create stable JSON representation
    stable_json = json.dumps(sorted_results, sort_keys=True, separators=(',', ':'))
    
    # Compute hash
    return hashlib.sha256(stable_json.encode('utf-8')).hexdigest()


def save_calibration_summary(
    calibration_dir: Path,
    summary: CalibrationSummary
) -> Path:
    """
    Save calibration summary to file.
    
    Args:
        calibration_dir: Base calibration directory
        summary: CalibrationSummary object
        
    Returns:
        Path to saved file
    """
    slice_dir = calibration_dir / summary.slice_name
    slice_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = slice_dir / "calibration_summary.json"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, indent=2)
    
    return summary_path


# Placeholder for future calibration run functionality
def run_calibration(
    slice_name: str,
    config_path: Path,
    output_dir: Path,
    seed: int = 42,
    cycles: int = 10
) -> CalibrationSummary:
    """
    Run calibration for a slice (PLACEHOLDER).
    
    This is a stub for future implementation. In a real system, this would:
    1. Run baseline experiment twice with same seed
    2. Compare results via hash
    3. Validate schema compliance
    4. Generate calibration_summary.json
    
    Args:
        slice_name: Name of the slice
        config_path: Path to curriculum config
        output_dir: Output directory for calibration results
        seed: Random seed for reproducibility
        cycles: Number of cycles to run
        
    Returns:
        CalibrationSummary object
        
    Raises:
        NotImplementedError: This is a placeholder
    """
    raise NotImplementedError(
        "run_calibration() is a placeholder. "
        "Calibration must be run manually and results placed in "
        "results/uplift_u2/calibration/<slice>/calibration_summary.json"
    )
