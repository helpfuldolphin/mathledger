"""
Reference TDA Profiles for Healthy Reasoning.

This module manages calibration profiles that define "healthy" topology
for each slice. Profiles contain:

- N_ref: Reference node count for SNS normalization
- τ (lifetime_threshold): Threshold for PCS long-lived features
- δ_max (deviation_max): Normalization constant for DRS
- D_ref^{(1)}: Reference H_1 persistence diagram for bottleneck distance

Profiles are built offline from known-good proof attempts and stored
as JSON for runtime loading.

References:
    - TDA_MIND_SCANNER_SPEC.md Sections 3.3-3.5
    - Appendix A.2 (Default Parameters)
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.tda.metric_complex import PersistenceDiagram, TDAResult


@dataclass
class ReferenceTDAProfile:
    """
    Calibration profile for a specific slice.

    Contains reference parameters and diagrams for computing SNS, PCS, DRS.

    Attributes:
        slice_name: Identifier for the slice (e.g., "PL-1", "U2")
        n_ref: Reference node count for SNS (Section 3.3)
        lifetime_threshold: τ for PCS computation (Section 3.4)
        deviation_max: δ_max for DRS normalization (Section 3.5)
        reference_diagram_h1: Reference H_1 persistence diagram for DRS
        reference_diagram_h0: Optional reference H_0 diagram
        num_samples: Number of samples used to build this profile
        version: Profile version for governance tracking
        metadata: Additional calibration metadata
    """
    slice_name: str
    n_ref: int = 50
    lifetime_threshold: float = 0.05
    deviation_max: float = 0.5
    reference_diagram_h1: Optional["PersistenceDiagram"] = None
    reference_diagram_h0: Optional["PersistenceDiagram"] = None
    num_samples: int = 0
    version: str = "0.1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """
        Compute deterministic hash of profile contents.

        Used for governance versioning and change detection.
        """
        content = {
            "slice_name": self.slice_name,
            "n_ref": self.n_ref,
            "lifetime_threshold": self.lifetime_threshold,
            "deviation_max": self.deviation_max,
            "num_samples": self.num_samples,
            "version": self.version,
        }

        # Include diagram summaries if present
        if self.reference_diagram_h1 is not None:
            content["h1_summary"] = {
                "num_intervals": self.reference_diagram_h1.num_intervals,
                "total_lifetime": self.reference_diagram_h1.total_lifetime,
            }

        serialized = json.dumps(content, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        result = {
            "slice_name": self.slice_name,
            "n_ref": self.n_ref,
            "lifetime_threshold": self.lifetime_threshold,
            "deviation_max": self.deviation_max,
            "num_samples": self.num_samples,
            "version": self.version,
            "metadata": self.metadata,
        }

        if self.reference_diagram_h1 is not None:
            result["reference_diagram_h1"] = _diagram_to_dict(self.reference_diagram_h1)

        if self.reference_diagram_h0 is not None:
            result["reference_diagram_h0"] = _diagram_to_dict(self.reference_diagram_h0)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceTDAProfile":
        """Deserialize from dictionary."""
        profile = cls(
            slice_name=data["slice_name"],
            n_ref=data.get("n_ref", 50),
            lifetime_threshold=data.get("lifetime_threshold", 0.05),
            deviation_max=data.get("deviation_max", 0.5),
            num_samples=data.get("num_samples", 0),
            version=data.get("version", "0.1.0"),
            metadata=data.get("metadata", {}),
        )

        if "reference_diagram_h1" in data:
            profile.reference_diagram_h1 = _dict_to_diagram(
                data["reference_diagram_h1"], dimension=1
            )

        if "reference_diagram_h0" in data:
            profile.reference_diagram_h0 = _dict_to_diagram(
                data["reference_diagram_h0"], dimension=0
            )

        return profile

    @classmethod
    def default(cls, slice_name: str) -> "ReferenceTDAProfile":
        """
        Create a default profile with spec-defined parameters.

        Per TDA_MIND_SCANNER_SPEC.md Appendix A.2:
        - n_ref = 50 (default reference size)
        - τ = 0.05 (lifetime threshold)
        - δ_max = 0.5 (deviation normalization)
        """
        return cls(
            slice_name=slice_name,
            n_ref=50,
            lifetime_threshold=0.05,
            deviation_max=0.5,
        )


def _diagram_to_dict(diagram: "PersistenceDiagram") -> Dict[str, Any]:
    """Convert PersistenceDiagram to serializable dict."""
    return {
        "dimension": diagram.dimension,
        "intervals": [
            {"birth": iv.birth, "death": iv.death}
            for iv in diagram.intervals
        ],
    }


def _dict_to_diagram(data: Dict[str, Any], dimension: int) -> "PersistenceDiagram":
    """Convert dict back to PersistenceDiagram."""
    from backend.tda.metric_complex import PersistenceDiagram, PersistenceInterval

    intervals = [
        PersistenceInterval(
            birth=iv["birth"],
            death=iv["death"],
            dimension=dimension,
        )
        for iv in data.get("intervals", [])
    ]

    return PersistenceDiagram(dimension=dimension, intervals=intervals)


def build_reference_profile(
    slice_name: str,
    tda_results: List["TDAResult"],
    node_counts: Optional[List[int]] = None,
    percentile: float = 95.0,
) -> ReferenceTDAProfile:
    """
    Build a reference profile from a collection of healthy TDA results.

    This is the offline calibration step that produces profiles for runtime use.

    Args:
        slice_name: Identifier for the slice
        tda_results: List of TDAResult from known-good reasoning attempts
        node_counts: Optional list of node counts (for N_ref computation)
        percentile: Percentile for δ_max computation (default: 95th)

    Returns:
        ReferenceTDAProfile calibrated from the input data
    """
    if not tda_results:
        return ReferenceTDAProfile.default(slice_name)

    # Compute N_ref from node counts (median)
    if node_counts and len(node_counts) > 0:
        n_ref = int(np.median(node_counts))
    else:
        n_ref = 50  # Default

    # Compute τ from H_1 lifetime distributions
    all_h1_lifetimes: List[float] = []
    for result in tda_results:
        h1_diagram = result.diagram(1)
        all_h1_lifetimes.extend(h1_diagram.lifetimes(exclude_essential=True))

    if all_h1_lifetimes:
        # τ at 25th percentile separates "short" from "long"
        lifetime_threshold = float(np.percentile(all_h1_lifetimes, 25))
    else:
        lifetime_threshold = 0.05

    # Compute δ_max from pairwise bottleneck distances
    from backend.tda.metric_complex import bottleneck_distance

    h1_diagrams = [r.diagram(1) for r in tda_results]
    distances: List[float] = []

    # Compute pairwise distances (expensive for large collections)
    for i in range(len(h1_diagrams)):
        for j in range(i + 1, len(h1_diagrams)):
            d = bottleneck_distance(h1_diagrams[i], h1_diagrams[j])
            distances.append(d)

    if distances:
        deviation_max = float(np.percentile(distances, percentile))
    else:
        deviation_max = 0.5

    # Build reference H_1 diagram (mean/representative)
    # For simplicity, use the diagram from the median-lifetime result
    if tda_results:
        median_idx = len(tda_results) // 2
        reference_diagram_h1 = tda_results[median_idx].diagram(1)
    else:
        reference_diagram_h1 = None

    return ReferenceTDAProfile(
        slice_name=slice_name,
        n_ref=n_ref,
        lifetime_threshold=lifetime_threshold,
        deviation_max=deviation_max,
        reference_diagram_h1=reference_diagram_h1,
        num_samples=len(tda_results),
        metadata={
            "percentile": percentile,
            "num_h1_lifetimes": len(all_h1_lifetimes),
            "num_pairwise_distances": len(distances),
        },
    )


def load_reference_profiles(
    path: Path,
) -> Dict[str, ReferenceTDAProfile]:
    """
    Load reference profiles from a JSON file.

    Args:
        path: Path to JSON file containing profiles

    Returns:
        Dict mapping slice_name to ReferenceTDAProfile
    """
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles: Dict[str, ReferenceTDAProfile] = {}

    for slice_name, profile_data in data.get("profiles", {}).items():
        profiles[slice_name] = ReferenceTDAProfile.from_dict(profile_data)

    return profiles


def save_reference_profiles(
    profiles: Dict[str, ReferenceTDAProfile],
    path: Path,
) -> None:
    """
    Save reference profiles to a JSON file.

    Args:
        profiles: Dict mapping slice_name to ReferenceTDAProfile
        path: Output path for JSON file
    """
    data = {
        "version": "1.0",
        "spec": "TDA_MIND_SCANNER_SPEC v0.1",
        "profiles": {
            name: profile.to_dict()
            for name, profile in profiles.items()
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def get_or_create_profile(
    profiles: Dict[str, ReferenceTDAProfile],
    slice_name: str,
) -> ReferenceTDAProfile:
    """
    Get an existing profile or create a default one.

    Args:
        profiles: Dict of existing profiles
        slice_name: Slice identifier

    Returns:
        ReferenceTDAProfile for the slice
    """
    if slice_name in profiles:
        return profiles[slice_name]

    # Check for partial matches (e.g., "PL-1" matches "PL")
    for name, profile in profiles.items():
        if slice_name.startswith(name) or name.startswith(slice_name):
            return profile

    # Return default profile
    return ReferenceTDAProfile.default(slice_name)
