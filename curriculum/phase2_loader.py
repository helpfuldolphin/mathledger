"""
Phase II Curriculum Loader with Uplift Slice Support

Implements CurriculumLoaderV2 for Phase II uplift experiments with:
- SuccessMetricSpec for configurable success definitions
- UpliftSlice dataclass for uplift-specific slice parameters
- schema_version "phase2-v1" validation
- Fingerprint generation for drift detection
- CLI introspection and drift checking
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore


@dataclass(frozen=True)
class SuccessMetricSpec:
    """
    Defines how success is measured for an uplift slice.
    
    Maps to compute functions in experiments/slice_success_metrics.py:
    - "goal_hit": compute_goal_hit()
    - "sparse_reward" / "sparse_success": compute_sparse_reward() / compute_sparse_success()
    - "tree_depth" / "chain_depth": compute_tree_depth() / compute_chain_depth()
    - "dependency_coordination": compute_dependency_coordination()
    """
    kind: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        # Accept flexible naming for success metrics
        valid_kinds = {
            "goal_hit", "multi_goal_success",
            "sparse_reward", "sparse_success",
            "tree_depth", "chain_depth", "chain_success",
            "dependency_coordination", "dependency_success"
        }
        if self.kind not in valid_kinds:
            raise ValueError(
                f"Invalid success metric kind '{self.kind}'. "
                f"Must be one of: {', '.join(sorted(valid_kinds))}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class UpliftSlice:
    """
    Phase II uplift slice definition with success metrics.
    
    Attributes:
        name: Unique slice identifier (e.g., "slice_uplift_goal")
        description: Human-readable explanation
        parameters: Generator parameters (atoms, depth_max, breadth_max, etc.)
        success_metric: How to measure slice success
        uplift: Phase II metadata (phase, experiment_family, etc.)
        budget: Resource constraints (max_candidates_per_cycle, etc.)
        formula_pool_entries: Optional initial formula set
        metadata: Additional fields from config
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    success_metric: SuccessMetricSpec
    uplift: Dict[str, Any] = field(default_factory=dict)
    budget: Dict[str, Any] = field(default_factory=dict)
    formula_pool_entries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "UpliftSlice":
        """Parse a slice from YAML configuration dictionary."""
        if "success_metric" not in data:
            raise ValueError(f"Slice '{name}' missing required 'success_metric' field")
        if "parameters" not in data:
            raise ValueError(f"Slice '{name}' missing required 'parameters' field")
        
        metric_data = data["success_metric"]
        if not isinstance(metric_data, dict) or "kind" not in metric_data:
            raise ValueError(f"Slice '{name}' has invalid success_metric format")
        
        success_metric = SuccessMetricSpec(
            kind=metric_data["kind"],
            parameters=metric_data.get("parameters", {})
        )
        
        # Extract known fields
        description = data.get("description", "")
        parameters = dict(data["parameters"])
        uplift = dict(data.get("uplift", {}))
        budget = dict(data.get("budget", {}))
        formula_pool = list(data.get("formula_pool_entries", []))
        
        # Capture unknown fields as metadata
        known_keys = {
            "description", "parameters", "success_metric", 
            "uplift", "budget", "formula_pool_entries"
        }
        metadata = {k: v for k, v in data.items() if k not in known_keys}
        
        return cls(
            name=name,
            description=description,
            parameters=parameters,
            success_metric=success_metric,
            uplift=uplift,
            budget=budget,
            formula_pool_entries=formula_pool,
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "success_metric": self.success_metric.to_dict(),
        }
        if self.uplift:
            result["uplift"] = self.uplift
        if self.budget:
            result["budget"] = self.budget
        if self.formula_pool_entries:
            result["formula_pool_entries"] = self.formula_pool_entries
        result.update(self.metadata)
        return result


@dataclass
class CurriculumLoaderV2:
    """
    Phase II curriculum loader with validation and drift detection.
    
    Loads curriculum_uplift_phase2.yaml and provides:
    - Structured validation with clear error messages
    - Slice introspection (list, show details)
    - Fingerprint generation for drift detection
    - Diff computation between configurations
    """
    schema_version: str
    slices: List[UpliftSlice]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "CurriculumLoaderV2":
        """
        Load curriculum from YAML file.
        
        Args:
            config_path: Path to curriculum YAML. If None, uses default location.
        
        Returns:
            Loaded curriculum
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if config_path is None:
            # Default to config/curriculum_uplift_phase2.yaml
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config",
                "curriculum_uplift_phase2.yaml"
            )
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Curriculum config not found: {config_path}")
        
        if yaml is None:
            raise ImportError("pyyaml is required for curriculum loading")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f.read())
        
        return cls.from_dict(config)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CurriculumLoaderV2":
        """
        Parse curriculum from configuration dictionary.
        
        Validates schema_version and slice definitions.
        """
        errors = cls.validate_config(config)
        if errors:
            raise ValueError(f"Invalid curriculum configuration:\n" + "\n".join(f"  - {e}" for e in errors))
        
        version = config.get("version", "")
        # Accept version as string or number
        schema_version = str(version)
        
        # Extract slices
        slices_data = config.get("slices", {})
        if not isinstance(slices_data, dict):
            raise ValueError("'slices' must be a dictionary")
        
        slices = []
        for slice_name, slice_data in slices_data.items():
            slices.append(UpliftSlice.from_dict(slice_name, slice_data))
        
        # Capture metadata (everything except version and slices)
        metadata = {k: v for k, v in config.items() if k not in {"version", "slices"}}
        
        return cls(
            schema_version=f"phase2-v{schema_version}",
            slices=slices,
            metadata=metadata
        )
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """
        Validate curriculum configuration.
        
        Returns:
            List of error messages. Empty if valid.
        """
        errors: List[str] = []
        
        # Check version
        if "version" not in config:
            errors.append("Missing 'version' field")
        else:
            version = str(config["version"])
            # Phase 2 should use version 2.x.x
            if not version.startswith("2."):
                errors.append(f"Expected version 2.x.x for Phase II, got '{version}'")
        
        # Check slices
        if "slices" not in config:
            errors.append("Missing 'slices' field")
            return errors
        
        slices = config["slices"]
        if not isinstance(slices, dict):
            errors.append("'slices' must be a dictionary")
            return errors
        
        if not slices:
            errors.append("No slices defined")
        
        # Validate each slice
        for slice_name, slice_data in slices.items():
            if not isinstance(slice_data, dict):
                errors.append(f"Slice '{slice_name}' is not a dictionary")
                continue
            
            # Check required fields
            if "parameters" not in slice_data:
                errors.append(f"Slice '{slice_name}' missing 'parameters'")
            if "success_metric" not in slice_data:
                errors.append(f"Slice '{slice_name}' missing 'success_metric'")
            else:
                metric = slice_data["success_metric"]
                if not isinstance(metric, dict):
                    errors.append(f"Slice '{slice_name}' success_metric is not a dictionary")
                elif "kind" not in metric:
                    errors.append(f"Slice '{slice_name}' success_metric missing 'kind'")
            
            # Verify uplift metadata
            uplift = slice_data.get("uplift", {})
            if uplift:
                if uplift.get("phase") != "II":
                    errors.append(f"Slice '{slice_name}' uplift.phase should be 'II' for Phase II")
        
        return errors
    
    def list_slices(self) -> List[str]:
        """Return list of slice names."""
        return [s.name for s in self.slices]
    
    def get_slice(self, name: str) -> Optional[UpliftSlice]:
        """Get slice by name."""
        for s in self.slices:
            if s.name == name:
                return s
        return None
    
    def show_slice(self, name: str) -> Dict[str, Any]:
        """Get detailed slice information."""
        s = self.get_slice(name)
        if s is None:
            raise ValueError(f"Slice '{name}' not found")
        return s.to_dict()
    
    def show_metrics(self) -> Dict[str, Any]:
        """Show success metrics for all slices."""
        return {
            s.name: {
                "kind": s.success_metric.kind,
                "parameters": s.success_metric.parameters
            }
            for s in self.slices
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "slices": {s.name: s.to_dict() for s in self.slices},
            "metadata": self.metadata
        }


@dataclass
class CurriculumFingerprint:
    """
    Canonical fingerprint for curriculum drift detection.
    
    Captures configuration state with SHA-256 hash for comparison.
    """
    schema_version: str
    slice_count: int
    slice_names: List[str]
    slice_fingerprints: Dict[str, str]
    timestamp: str
    sha256: str
    
    @classmethod
    def generate(cls, curriculum: CurriculumLoaderV2, run_id: Optional[str] = None) -> "CurriculumFingerprint":
        """
        Generate fingerprint from curriculum.
        
        Args:
            curriculum: Loaded curriculum
            run_id: Optional run identifier for timestamp suffix
        
        Returns:
            Fingerprint with SHA-256 hash
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        if run_id:
            timestamp = f"{timestamp}@{run_id}"
        
        slice_names = [s.name for s in curriculum.slices]
        
        # Generate per-slice fingerprints (hash of canonical JSON)
        slice_fingerprints = {}
        for s in curriculum.slices:
            slice_dict = s.to_dict()
            canonical = json.dumps(slice_dict, sort_keys=True, separators=(',', ':'))
            slice_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            slice_fingerprints[s.name] = slice_hash
        
        # Generate overall fingerprint
        curriculum_dict = curriculum.to_dict()
        canonical = json.dumps(curriculum_dict, sort_keys=True, separators=(',', ':'))
        sha256 = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
        
        return cls(
            schema_version=curriculum.schema_version,
            slice_count=len(curriculum.slices),
            slice_names=slice_names,
            slice_fingerprints=slice_fingerprints,
            timestamp=timestamp,
            sha256=sha256
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "slice_count": self.slice_count,
            "slice_names": self.slice_names,
            "slice_fingerprints": self.slice_fingerprints,
            "timestamp": self.timestamp,
            "sha256": self.sha256
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumFingerprint":
        """Load fingerprint from dictionary."""
        return cls(
            schema_version=data["schema_version"],
            slice_count=data["slice_count"],
            slice_names=data["slice_names"],
            slice_fingerprints=data["slice_fingerprints"],
            timestamp=data["timestamp"],
            sha256=data["sha256"]
        )
    
    def save(self, path: str) -> None:
        """Save fingerprint to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, path: str) -> "CurriculumFingerprint":
        """Load fingerprint from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_curriculum_diff(fp1: CurriculumFingerprint, fp2: CurriculumFingerprint) -> Dict[str, Any]:
    """
    Compute diff between two curriculum fingerprints.
    
    Returns:
        Dictionary with diff details:
        - changed: bool
        - schema_version_changed: bool
        - slices_added: List[str]
        - slices_removed: List[str]
        - slices_modified: List[str]
        - slice_details: Dict[str, Dict]
    """
    changed = fp1.sha256 != fp2.sha256
    schema_changed = fp1.schema_version != fp2.schema_version
    
    set1 = set(fp1.slice_names)
    set2 = set(fp2.slice_names)
    
    added = sorted(set2 - set1)
    removed = sorted(set1 - set2)
    common = sorted(set1 & set2)
    
    modified = []
    slice_details = {}
    
    for name in common:
        hash1 = fp1.slice_fingerprints.get(name)
        hash2 = fp2.slice_fingerprints.get(name)
        if hash1 != hash2:
            modified.append(name)
            slice_details[name] = {
                "old_hash": hash1,
                "new_hash": hash2
            }
    
    return {
        "changed": changed,
        "schema_version_changed": schema_changed,
        "old_schema_version": fp1.schema_version,
        "new_schema_version": fp2.schema_version,
        "slices_added": added,
        "slices_removed": removed,
        "slices_modified": modified,
        "slice_details": slice_details,
        "old_sha256": fp1.sha256,
        "new_sha256": fp2.sha256,
        "old_timestamp": fp1.timestamp,
        "new_timestamp": fp2.timestamp
    }
