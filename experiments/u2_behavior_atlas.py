#!/usr/bin/env python3
"""
PHASE II — NOT USED IN PHASE I

Institutional Behavior Atlas Generator
======================================

This module generates cross-slice behavioral atlases that profile and cluster
slices by structural patterns. It produces purely descriptive outputs:
profiles, distances, and clusters — NO evaluation or uplift inference.

Agent: metrics-engineer-6 (D6)

Usage:
    uv run python experiments/u2_behavior_atlas.py --input-dir results --out-dir artifacts/atlas
    uv run python experiments/u2_behavior_atlas.py --input-dir results --atlas --matrix --fingerprints

ABSOLUTE SAFEGUARDS:
    - No uplift inference.
    - No slice ranking.
    - No governance-level claims.
    - Output must remain purely descriptive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Use Agg backend for deterministic, non-interactive plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add project root to sys.path for imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.u2_cross_slice_analysis import (
    SliceResults,
    BehavioralFingerprint,
    load_slice_results,
    generate_behavior_signature,
    fingerprint_to_dict,
    discover_slices,
    _compute_js_divergence,
    _compute_trend_correlation,
    _compute_abstention_trend,
    _safe_mean,
)


# ===========================================================================
# PHASE II — DATA STRUCTURES
# ===========================================================================

@dataclass
class SliceProfile:
    """
    Complete behavioral profile for a slice (both modes).
    Purely descriptive — no quality or performance implications.
    """
    slice_name: str
    baseline_fingerprint: BehavioralFingerprint
    rfl_fingerprint: BehavioralFingerprint
    baseline_record_count: int
    rfl_record_count: int
    # Derived behavioral features for clustering
    feature_vector: List[float] = field(default_factory=list)


@dataclass
class BehaviorAtlas:
    """
    Institutional Behavior Atlas containing all slice profiles and matrices.
    """
    phase_label: str = "PHASE II — NOT USED IN PHASE I"
    slice_profiles: Dict[str, SliceProfile] = field(default_factory=dict)
    js_divergence_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    trend_similarity_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    abstention_similarity_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    archetypes: Dict[str, str] = field(default_factory=dict)
    manifest_hash: str = ""


@dataclass
class ArchetypeAssignment:
    """
    Archetype assignment for a slice.
    Labels are purely descriptive behavioral categories.
    """
    slice_name: str
    archetype_label: str
    cluster_id: int
    centroid_distance: float


# ===========================================================================
# ARCHETYPE LABELS (Purely Descriptive)
# ===========================================================================

# These labels describe BEHAVIORAL PATTERNS only.
# They do NOT imply quality, performance, or preference.
ARCHETYPE_LABELS = {
    0: "sparse-shallow",      # Low metric variance, shallow chains
    1: "sparse-deep",         # Low metric variance, deeper chains
    2: "dense-shallow",       # High metric variance, shallow chains
    3: "dense-deep",          # High metric variance, deeper chains
    4: "volatile-trending",   # High abstention volatility
    5: "stable-uniform",      # Low abstention volatility
    6: "mixed-transitional",  # Mixed characteristics
    7: "concentrated-focal",  # Concentrated metric distribution
}


# ===========================================================================
# FEATURE EXTRACTION
# ===========================================================================

def extract_feature_vector(
    baseline_fp: BehavioralFingerprint,
    rfl_fp: BehavioralFingerprint,
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
) -> List[float]:
    """
    Extract a fixed-length feature vector for clustering.
    
    Features are purely structural/behavioral:
    - Histogram entropy
    - Chain depth statistics
    - Abstention trend statistics
    - Temporal smoothness
    
    NO features related to success rate, uplift, or performance.
    """
    features = []
    
    # Feature 1: Metric histogram entropy (baseline)
    features.append(_histogram_entropy(baseline_fp.metric_value_histogram))
    
    # Feature 2: Metric histogram entropy (rfl)
    features.append(_histogram_entropy(rfl_fp.metric_value_histogram))
    
    # Feature 3: Chain depth mean (baseline)
    features.append(_weighted_mean(baseline_fp.longest_chain_distribution))
    
    # Feature 4: Chain depth mean (rfl)
    features.append(_weighted_mean(rfl_fp.longest_chain_distribution))
    
    # Feature 5: Chain depth variance (baseline)
    features.append(_weighted_variance(baseline_fp.longest_chain_distribution))
    
    # Feature 6: Goal hit entropy (baseline)
    features.append(_histogram_entropy(baseline_fp.goal_hit_distribution))
    
    # Feature 7: Temporal smoothness (baseline)
    features.append(min(baseline_fp.temporal_smoothness_signature, 100.0))
    
    # Feature 8: Temporal smoothness (rfl)
    features.append(min(rfl_fp.temporal_smoothness_signature, 100.0))
    
    # Feature 9: Abstention trend volatility (baseline)
    baseline_trend = _compute_abstention_trend(baseline_records)
    features.append(_trend_volatility(baseline_trend))
    
    # Feature 10: Abstention trend volatility (rfl)
    rfl_trend = _compute_abstention_trend(rfl_records)
    features.append(_trend_volatility(rfl_trend))
    
    # Normalize features to [0, 1] range for clustering stability
    return [round(f, 6) for f in features]


def _histogram_entropy(hist: Dict[Any, int]) -> float:
    """Compute Shannon entropy of a histogram."""
    if not hist:
        return 0.0
    
    total = sum(hist.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in hist.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return round(entropy, 6)


def _weighted_mean(hist: Dict[Any, int]) -> float:
    """Compute weighted mean of a histogram."""
    if not hist:
        return 0.0
    
    total_weight = sum(hist.values())
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(float(k) * v for k, v in hist.items())
    return round(weighted_sum / total_weight, 6)


def _weighted_variance(hist: Dict[Any, int]) -> float:
    """Compute weighted variance of a histogram."""
    if not hist:
        return 0.0
    
    mean = _weighted_mean(hist)
    total_weight = sum(hist.values())
    if total_weight == 0:
        return 0.0
    
    variance = sum(v * (float(k) - mean) ** 2 for k, v in hist.items()) / total_weight
    return round(variance, 6)


def _trend_volatility(trend: List[float]) -> float:
    """Compute volatility (standard deviation) of a trend."""
    if len(trend) < 2:
        return 0.0
    
    mean = sum(trend) / len(trend)
    variance = sum((v - mean) ** 2 for v in trend) / len(trend)
    return round(math.sqrt(variance), 6)


# ===========================================================================
# MATRIX COMPUTATION
# ===========================================================================

def compute_js_divergence_matrix(
    profiles: Dict[str, SliceProfile],
) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise JS-divergence matrix across all slices.
    Uses baseline metric histograms for comparison.
    
    Returns symmetric matrix as nested dict.
    """
    slice_names = sorted(profiles.keys())
    matrix = {name: {} for name in slice_names}
    
    for i, name1 in enumerate(slice_names):
        for name2 in slice_names[i:]:
            hist1 = profiles[name1].baseline_fingerprint.metric_value_histogram
            hist2 = profiles[name2].baseline_fingerprint.metric_value_histogram
            
            divergence = _compute_js_divergence(hist1, hist2)
            
            matrix[name1][name2] = round(divergence, 6)
            matrix[name2][name1] = round(divergence, 6)
    
    return matrix


def compute_trend_similarity_matrix(
    profiles: Dict[str, SliceProfile],
    baseline_records: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise trend similarity (correlation) matrix.
    Uses baseline abstention trends.
    
    Returns symmetric matrix as nested dict.
    Similarity is correlation coefficient in [-1, 1].
    """
    slice_names = sorted(profiles.keys())
    matrix = {name: {} for name in slice_names}
    
    # Pre-compute trends
    trends = {}
    for name in slice_names:
        trends[name] = _compute_abstention_trend(baseline_records.get(name, []))
    
    for i, name1 in enumerate(slice_names):
        for name2 in slice_names[i:]:
            correlation = _compute_trend_correlation(trends[name1], trends[name2])
            
            matrix[name1][name2] = round(correlation, 6)
            matrix[name2][name1] = round(correlation, 6)
    
    return matrix


def compute_abstention_similarity_matrix(
    profiles: Dict[str, SliceProfile],
    baseline_records: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise abstention profile similarity.
    Uses cosine similarity of abstention trend vectors.
    
    Returns symmetric matrix as nested dict.
    """
    slice_names = sorted(profiles.keys())
    matrix = {name: {} for name in slice_names}
    
    # Pre-compute trends
    trends = {}
    for name in slice_names:
        trends[name] = _compute_abstention_trend(baseline_records.get(name, []))
    
    for i, name1 in enumerate(slice_names):
        for name2 in slice_names[i:]:
            similarity = _cosine_similarity(trends[name1], trends[name2])
            
            matrix[name1][name2] = round(similarity, 6)
            matrix[name2][name1] = round(similarity, 6)
    
    return matrix


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0
    
    # Align lengths
    min_len = min(len(vec1), len(vec2))
    if min_len == 0:
        return 0.0
    
    v1 = vec1[:min_len]
    v2 = vec2[:min_len]
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


# ===========================================================================
# ARCHETYPE CLUSTERING
# ===========================================================================

def classify_archetypes(
    profiles: Dict[str, SliceProfile],
    n_clusters: int = 4,
    seed: int = 42,
    max_iterations: int = 100,
) -> Dict[str, ArchetypeAssignment]:
    """
    Classify slices into behavioral archetypes using k-means clustering.
    
    IMPORTANT: Archetypes are purely descriptive behavioral categories.
    They do NOT imply quality, performance, or preference.
    
    Args:
        profiles: Dict of slice profiles with feature vectors
        n_clusters: Number of archetype clusters (capped at available labels)
        seed: Random seed for deterministic clustering
        max_iterations: Maximum k-means iterations
        
    Returns:
        Dict mapping slice_name to ArchetypeAssignment
    """
    slice_names = sorted(profiles.keys())
    
    if len(slice_names) == 0:
        return {}
    
    # Cap clusters at available slices and labels
    n_clusters = min(n_clusters, len(slice_names), len(ARCHETYPE_LABELS))
    
    if n_clusters <= 1:
        # Single cluster case
        return {
            name: ArchetypeAssignment(
                slice_name=name,
                archetype_label=ARCHETYPE_LABELS[0],
                cluster_id=0,
                centroid_distance=0.0,
            )
            for name in slice_names
        }
    
    # Extract feature vectors
    feature_matrix = []
    for name in slice_names:
        fv = profiles[name].feature_vector
        if not fv:
            fv = [0.0] * 10  # Default zero vector
        feature_matrix.append(fv)
    
    # Run deterministic k-means
    assignments, centroids = _kmeans(
        feature_matrix,
        n_clusters=n_clusters,
        seed=seed,
        max_iterations=max_iterations,
    )
    
    # Build archetype assignments
    result = {}
    for i, name in enumerate(slice_names):
        cluster_id = assignments[i]
        centroid = centroids[cluster_id]
        distance = _euclidean_distance(feature_matrix[i], centroid)
        
        result[name] = ArchetypeAssignment(
            slice_name=name,
            archetype_label=ARCHETYPE_LABELS.get(cluster_id, f"archetype-{cluster_id}"),
            cluster_id=cluster_id,
            centroid_distance=round(distance, 6),
        )
    
    return result


def _kmeans(
    data: List[List[float]],
    n_clusters: int,
    seed: int,
    max_iterations: int,
) -> Tuple[List[int], List[List[float]]]:
    """
    Deterministic k-means clustering implementation.
    
    Uses fixed seed for reproducibility.
    """
    import random
    rng = random.Random(seed)
    
    n_samples = len(data)
    n_features = len(data[0]) if data else 0
    
    if n_samples == 0 or n_features == 0:
        return [], []
    
    # Initialize centroids using k-means++ style (deterministic with seed)
    centroids = []
    available_indices = list(range(n_samples))
    
    # First centroid: random
    first_idx = rng.choice(available_indices)
    centroids.append(list(data[first_idx]))
    
    # Subsequent centroids: proportional to squared distance
    for _ in range(1, n_clusters):
        distances = []
        for i in range(n_samples):
            min_dist = min(_euclidean_distance(data[i], c) for c in centroids)
            distances.append(min_dist ** 2)
        
        total_dist = sum(distances)
        if total_dist == 0:
            # All points are identical to centroids
            idx = rng.choice(available_indices)
        else:
            # Weighted random selection
            threshold = rng.random() * total_dist
            cumsum = 0.0
            idx = 0
            for i, d in enumerate(distances):
                cumsum += d
                if cumsum >= threshold:
                    idx = i
                    break
        
        centroids.append(list(data[idx]))
    
    # Iterative refinement
    assignments = [0] * n_samples
    
    for _ in range(max_iterations):
        # Assignment step
        new_assignments = []
        for i in range(n_samples):
            distances = [_euclidean_distance(data[i], c) for c in centroids]
            new_assignments.append(distances.index(min(distances)))
        
        # Check convergence
        if new_assignments == assignments:
            break
        
        assignments = new_assignments
        
        # Update step
        for k in range(n_clusters):
            cluster_points = [data[i] for i in range(n_samples) if assignments[i] == k]
            if cluster_points:
                centroids[k] = [
                    sum(p[j] for p in cluster_points) / len(cluster_points)
                    for j in range(n_features)
                ]
    
    return assignments, centroids


def _euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    if len(vec1) != len(vec2):
        return float('inf')
    
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


# ===========================================================================
# HEATMAP GENERATION
# ===========================================================================

def generate_heatmap(
    matrix: Dict[str, Dict[str, float]],
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Generate a deterministic heatmap from a matrix.
    
    Uses Agg backend for non-interactive, reproducible output.
    """
    slice_names = sorted(matrix.keys())
    n = len(slice_names)
    
    if n == 0:
        return
    
    # Build numpy-free matrix
    data = [[matrix[row][col] for col in slice_names] for row in slice_names]
    
    # Create figure with deterministic settings
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))
    
    # Plot heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(slice_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(slice_names, fontsize=8)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)
    
    # Add title
    ax.set_title(f"{title}\n(PHASE II — Descriptive Only)", fontsize=10)
    
    # Add values to cells if matrix is small enough
    if n <= 12:
        for i in range(n):
            for j in range(n):
                value = data[i][j]
                text_color = 'white' if value > (vmax or 1) * 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontsize=6)
    
    plt.tight_layout()
    
    # Save with deterministic settings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def generate_archetype_chart(
    archetypes: Dict[str, ArchetypeAssignment],
    output_path: Path,
) -> None:
    """
    Generate a bar chart showing archetype distribution.
    Purely descriptive — no ranking or quality implication.
    """
    if not archetypes:
        return
    
    # Count archetypes
    archetype_counts: Dict[str, int] = {}
    for assignment in archetypes.values():
        label = assignment.archetype_label
        archetype_counts[label] = archetype_counts.get(label, 0) + 1
    
    labels = sorted(archetype_counts.keys())
    counts = [archetype_counts[label] for label in labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a neutral color palette (no red/green to avoid quality connotation)
    colors = plt.cm.tab20(range(len(labels)))
    
    bars = ax.bar(range(len(labels)), counts, color=colors)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of Slices', fontsize=10)
    ax.set_title('Behavioral Archetype Distribution\n(PHASE II — Descriptive Only, No Quality Implication)',
                fontsize=10)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
               str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# ===========================================================================
# ATLAS GENERATION
# ===========================================================================

def build_behavior_atlas(
    input_dir: Path,
    slice_names: Optional[List[str]] = None,
    n_clusters: int = 4,
    clustering_seed: int = 42,
) -> Tuple[BehaviorAtlas, Dict[str, List[Dict[str, Any]]]]:
    """
    Build complete behavioral atlas from slice results.
    
    Returns:
        (BehaviorAtlas, baseline_records_dict)
    """
    # Discover slices if not specified
    if slice_names is None:
        discovered = discover_slices(input_dir)
        slice_names = sorted(set(s[0] for s in discovered))
    
    profiles: Dict[str, SliceProfile] = {}
    baseline_records: Dict[str, List[Dict[str, Any]]] = {}
    
    for slice_name in slice_names:
        try:
            baseline_results = load_slice_results(slice_name, "baseline", input_dir)
            rfl_results = load_slice_results(slice_name, "rfl", input_dir)
        except FileNotFoundError:
            continue
        
        baseline_fp = generate_behavior_signature(baseline_results)
        rfl_fp = generate_behavior_signature(rfl_results)
        
        # Extract feature vector
        feature_vector = extract_feature_vector(
            baseline_fp, rfl_fp,
            baseline_results.records, rfl_results.records,
        )
        
        profiles[slice_name] = SliceProfile(
            slice_name=slice_name,
            baseline_fingerprint=baseline_fp,
            rfl_fingerprint=rfl_fp,
            baseline_record_count=len(baseline_results.records),
            rfl_record_count=len(rfl_results.records),
            feature_vector=feature_vector,
        )
        
        baseline_records[slice_name] = baseline_results.records
    
    if not profiles:
        return BehaviorAtlas(), {}
    
    # Compute matrices
    js_matrix = compute_js_divergence_matrix(profiles)
    trend_matrix = compute_trend_similarity_matrix(profiles, baseline_records)
    abstention_matrix = compute_abstention_similarity_matrix(profiles, baseline_records)
    
    # Classify archetypes
    archetype_assignments = classify_archetypes(
        profiles,
        n_clusters=n_clusters,
        seed=clustering_seed,
    )
    
    archetypes = {
        name: assignment.archetype_label
        for name, assignment in archetype_assignments.items()
    }
    
    # Build atlas
    atlas = BehaviorAtlas(
        slice_profiles=profiles,
        js_divergence_matrix=js_matrix,
        trend_similarity_matrix=trend_matrix,
        abstention_similarity_matrix=abstention_matrix,
        archetypes=archetypes,
    )
    
    # Compute manifest hash
    atlas.manifest_hash = _compute_atlas_hash(atlas)
    
    return atlas, baseline_records


def _compute_atlas_hash(atlas: BehaviorAtlas) -> str:
    """Compute deterministic hash of atlas contents."""
    # Build hashable representation
    data = {
        "slice_names": sorted(atlas.slice_profiles.keys()),
        "fingerprint_hashes": {
            name: {
                "baseline": profile.baseline_fingerprint.fingerprint_hash,
                "rfl": profile.rfl_fingerprint.fingerprint_hash,
            }
            for name, profile in sorted(atlas.slice_profiles.items())
        },
        "js_matrix_hash": hashlib.sha256(
            json.dumps(atlas.js_divergence_matrix, sort_keys=True).encode()
        ).hexdigest()[:16],
        "trend_matrix_hash": hashlib.sha256(
            json.dumps(atlas.trend_similarity_matrix, sort_keys=True).encode()
        ).hexdigest()[:16],
        "archetypes": atlas.archetypes,
    }
    
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def atlas_to_dict(atlas: BehaviorAtlas) -> Dict[str, Any]:
    """Convert BehaviorAtlas to JSON-serializable dict."""
    return {
        "phase_label": atlas.phase_label,
        "disclaimer": "This atlas is purely descriptive. Archetypes do NOT imply quality, performance, or preference.",
        "slice_count": len(atlas.slice_profiles),
        "slice_names": sorted(atlas.slice_profiles.keys()),
        "fingerprints": {
            name: {
                "baseline": fingerprint_to_dict(profile.baseline_fingerprint),
                "rfl": fingerprint_to_dict(profile.rfl_fingerprint),
                "baseline_record_count": profile.baseline_record_count,
                "rfl_record_count": profile.rfl_record_count,
                "feature_vector": profile.feature_vector,
            }
            for name, profile in sorted(atlas.slice_profiles.items())
        },
        "js_divergence_matrix": atlas.js_divergence_matrix,
        "trend_similarity_matrix": atlas.trend_similarity_matrix,
        "abstention_similarity_matrix": atlas.abstention_similarity_matrix,
        "archetypes": atlas.archetypes,
        "manifest_hash": atlas.manifest_hash,
    }


def archetype_assignments_to_dict(
    assignments: Dict[str, ArchetypeAssignment],
) -> Dict[str, Any]:
    """Convert archetype assignments to JSON-serializable dict."""
    return {
        "phase_label": "PHASE II — NOT USED IN PHASE I",
        "disclaimer": "Archetypes are purely descriptive behavioral categories. They do NOT imply quality.",
        "assignments": {
            name: {
                "archetype_label": a.archetype_label,
                "cluster_id": a.cluster_id,
                "centroid_distance": a.centroid_distance,
            }
            for name, a in sorted(assignments.items())
        },
    }


# ===========================================================================
# SLICE BEHAVIOR DOSSIER GENERATOR
# ===========================================================================

def generate_slice_dossier(
    slice_name: str,
    atlas_path: str,
    fingerprints_path: str,
    out_path: str,
    k_neighbors: int = 3,
) -> Dict[str, Any]:
    """
    Generate a slice-specific behavior dossier.
    
    The dossier provides a detailed behavioral profile for a single slice,
    including its feature vector, archetype, and closest behavioral neighbors.
    
    PHASE II — Purely descriptive. No ranking or quality implication.
    
    Args:
        slice_name: Name of the slice to profile
        atlas_path: Path to behavior_atlas.json
        fingerprints_path: Path to fingerprints.json
        out_path: Output path for the dossier JSON
        k_neighbors: Number of closest behavioral neighbors to include (default: 3)
        
    Returns:
        Dossier dictionary with slice profile and neighbors
        
    Raises:
        ValueError: If slice_name not found in atlas
    """
    # Load atlas and fingerprints
    with open(atlas_path, 'r', encoding='utf-8') as f:
        atlas = json.load(f)
    
    with open(fingerprints_path, 'r', encoding='utf-8') as f:
        fingerprints = json.load(f)
    
    # Validate slice exists
    if slice_name not in atlas.get("slice_names", []):
        raise ValueError(f"Slice '{slice_name}' not found in atlas. Available: {atlas.get('slice_names', [])}")
    
    # Extract slice data
    slice_fingerprints = atlas.get("fingerprints", {}).get(slice_name, {})
    slice_archetype = atlas.get("archetypes", {}).get(slice_name, "unknown")
    
    # Get feature vector
    feature_vector = slice_fingerprints.get("feature_vector", [])
    
    # Find closest behavioral neighbors by JS-divergence (excluding self)
    js_matrix = atlas.get("js_divergence_matrix", {})
    js_row = js_matrix.get(slice_name, {})
    
    # Sort by JS-divergence (ascending = most similar behavior)
    neighbors_by_similarity = sorted(
        [(name, divergence) for name, divergence in js_row.items() if name != slice_name],
        key=lambda x: x[1]
    )[:k_neighbors]
    
    # Build nearest_neighbors list with their fingerprints
    # NOTE: Uses "neighbor_slice" and "distance" per spec. No ranking terms.
    nearest_neighbors = []
    fp_data = fingerprints.get("fingerprints", {})
    for neighbor_name, divergence in neighbors_by_similarity:
        neighbor_fp = fp_data.get(neighbor_name, {})
        nearest_neighbors.append({
            "neighbor_slice": neighbor_name,
            "distance": round(divergence, 6),
            "neighbor_archetype": atlas.get("archetypes", {}).get(neighbor_name, "unknown"),
            "baseline_hash": neighbor_fp.get("baseline_hash", ""),
            "rfl_hash": neighbor_fp.get("rfl_hash", ""),
        })
    
    # Extract trend similarity row
    trend_matrix = atlas.get("trend_similarity_matrix", {})
    trend_row = {k: round(v, 6) for k, v in sorted(trend_matrix.get(slice_name, {}).items())}
    
    # Extract abstention profile row
    abstention_matrix = atlas.get("abstention_similarity_matrix", {})
    abstention_row = {k: round(v, 6) for k, v in sorted(abstention_matrix.get(slice_name, {}).items())}
    
    # Compute metrics_summary: simple descriptive stats (NO Δp, NO p-values)
    # Purely structural statistics about the slice's behavioral profile
    js_distances = [d for _, d in neighbors_by_similarity] if neighbors_by_similarity else []
    trend_values = list(trend_row.values()) if trend_row else []
    abstention_values = list(abstention_row.values()) if abstention_row else []
    
    metrics_summary = {
        "feature_vector_dimension": len(feature_vector),
        "feature_vector_mean": round(sum(feature_vector) / len(feature_vector), 6) if feature_vector else 0.0,
        "neighbor_distance_mean": round(sum(js_distances) / len(js_distances), 6) if js_distances else 0.0,
        "neighbor_distance_min": round(min(js_distances), 6) if js_distances else 0.0,
        "trend_similarity_mean": round(sum(trend_values) / len(trend_values), 6) if trend_values else 0.0,
        "abstention_similarity_mean": round(sum(abstention_values) / len(abstention_values), 6) if abstention_values else 0.0,
        "neighbor_count": len(nearest_neighbors),
    }
    
    # Build dossier
    slice_fp = fp_data.get(slice_name, {})
    dossier = {
        "phase_label": "PHASE II — NOT USED IN PHASE I",
        "disclaimer": "This dossier is purely descriptive. It does NOT imply quality or performance.",
        "slice_name": slice_name,
        "assigned_archetype": slice_archetype,
        "feature_vector": feature_vector,
        "nearest_neighbors": nearest_neighbors,
        "metrics_summary": metrics_summary,
        "fingerprint_hashes": {
            "baseline": slice_fp.get("baseline_hash", ""),
            "rfl": slice_fp.get("rfl_hash", ""),
        },
        "trend_similarity_profile": trend_row,
        "abstention_similarity_profile": abstention_row,
        "lineage": {
            "atlas_path": atlas_path,
            "fingerprints_path": fingerprints_path,
            "atlas_manifest_hash": atlas.get("manifest_hash", ""),
        },
    }
    
    # Write dossier to file
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path_obj, 'w', encoding='utf-8') as f:
        json.dump(dossier, f, indent=2, sort_keys=True)
    
    return dossier


# ===========================================================================
# REAL VS SYNTHETIC ATLAS COMPARISON
# ===========================================================================

def compare_real_vs_synthetic(
    real_atlas_path: str,
    synthetic_atlas_path: str,
) -> Dict[str, Any]:
    """
    Compare two atlases (real vs synthetic) and report descriptive statistics.
    
    This is a READ-ONLY comparison for sanity checking synthetic worlds.
    It does NOT evaluate uplift or performance — only structural differences.
    
    For each slice present in both atlases:
    - Compare JS-divergence distributions
    - Compare archetype allocations
    - Compare trend similarity distributions
    
    PHASE II — Purely descriptive. No ranking or quality claims.
    
    Args:
        real_atlas_path: Path to the real experiment atlas JSON
        synthetic_atlas_path: Path to the synthetic experiment atlas JSON
        
    Returns:
        Dictionary with descriptive statistics comparing the two atlases,
        including slice_overlaps and distance_summary.
    """
    # Load both atlases
    with open(real_atlas_path, 'r', encoding='utf-8') as f:
        real_atlas = json.load(f)
    
    with open(synthetic_atlas_path, 'r', encoding='utf-8') as f:
        synthetic_atlas = json.load(f)
    
    # Compute slice overlaps
    real_slices = set(real_atlas.get("slice_names", []))
    synthetic_slices = set(synthetic_atlas.get("slice_names", []))
    overlapping_slices = sorted(real_slices & synthetic_slices)
    real_only_slices = sorted(real_slices - synthetic_slices)
    synthetic_only_slices = sorted(synthetic_slices - real_slices)
    
    # Per-slice comparison for overlapping slices
    slice_overlaps = {}
    for slice_name in overlapping_slices:
        real_archetype = real_atlas.get("archetypes", {}).get(slice_name, "unknown")
        synthetic_archetype = synthetic_atlas.get("archetypes", {}).get(slice_name, "unknown")
        
        # Get JS-divergence row for this slice
        real_js_row = real_atlas.get("js_divergence_matrix", {}).get(slice_name, {})
        synthetic_js_row = synthetic_atlas.get("js_divergence_matrix", {}).get(slice_name, {})
        
        # Compare trend similarity
        real_trend_row = real_atlas.get("trend_similarity_matrix", {}).get(slice_name, {})
        synthetic_trend_row = synthetic_atlas.get("trend_similarity_matrix", {}).get(slice_name, {})
        
        slice_overlaps[slice_name] = {
            "real_archetype": real_archetype,
            "synthetic_archetype": synthetic_archetype,
            "archetype_match": real_archetype == synthetic_archetype,
            "js_divergence_stats": {
                "real": _compute_distribution_stats(list(real_js_row.values())),
                "synthetic": _compute_distribution_stats(list(synthetic_js_row.values())),
            },
            "trend_similarity_stats": {
                "real": _compute_distribution_stats(list(real_trend_row.values())),
                "synthetic": _compute_distribution_stats(list(synthetic_trend_row.values())),
            },
        }
    
    # Extract JS-divergence distributions (global)
    real_js_values = _extract_matrix_values(real_atlas.get("js_divergence_matrix", {}))
    synthetic_js_values = _extract_matrix_values(synthetic_atlas.get("js_divergence_matrix", {}))
    
    # Extract trend similarity distributions (global)
    real_trend_values = _extract_matrix_values(real_atlas.get("trend_similarity_matrix", {}))
    synthetic_trend_values = _extract_matrix_values(synthetic_atlas.get("trend_similarity_matrix", {}))
    
    # Extract abstention similarity distributions
    real_abstention_values = _extract_matrix_values(real_atlas.get("abstention_similarity_matrix", {}))
    synthetic_abstention_values = _extract_matrix_values(synthetic_atlas.get("abstention_similarity_matrix", {}))
    
    # Count archetypes
    real_archetypes = real_atlas.get("archetypes", {})
    synthetic_archetypes = synthetic_atlas.get("archetypes", {})
    
    real_archetype_counts = _count_values(real_archetypes)
    synthetic_archetype_counts = _count_values(synthetic_archetypes)
    
    # Build distance_summary (global distribution comparison)
    distance_summary = {
        "js_divergence_distribution": {
            "real": _compute_distribution_stats(real_js_values),
            "synthetic": _compute_distribution_stats(synthetic_js_values),
        },
        "trend_similarity_distribution": {
            "real": _compute_distribution_stats(real_trend_values),
            "synthetic": _compute_distribution_stats(synthetic_trend_values),
        },
        "abstention_similarity_distribution": {
            "real": _compute_distribution_stats(real_abstention_values),
            "synthetic": _compute_distribution_stats(synthetic_abstention_values),
        },
    }
    
    # Build comparison report per spec (FROZEN CONTRACT)
    # NOTE: slice_counts uses total_real/total_synthetic per contract spec
    comparison = {
        "phase_label": "PHASE II — NOT USED IN PHASE I",
        "disclaimer": "This comparison is purely descriptive. No uplift or quality claims.",
        "real_atlas_path": real_atlas_path,
        "synthetic_atlas_path": synthetic_atlas_path,
        "slice_counts": {
            "total_real": len(real_slices),
            "total_synthetic": len(synthetic_slices),
            "overlapping": len(overlapping_slices),
            "real_only": len(real_only_slices),
            "synthetic_only": len(synthetic_only_slices),
        },
        "slice_overlaps": slice_overlaps,
        "distance_summary": distance_summary,
        "archetype_counts": {
            "real": real_archetype_counts,
            "synthetic": synthetic_archetype_counts,
        },
    }
    
    return comparison


def _extract_matrix_values(matrix: Dict[str, Dict[str, float]]) -> List[float]:
    """Extract upper-triangle values from a symmetric matrix (excluding diagonal)."""
    values = []
    slice_names = sorted(matrix.keys())
    for i, name1 in enumerate(slice_names):
        for name2 in slice_names[i + 1:]:
            val = matrix.get(name1, {}).get(name2)
            if val is not None:
                values.append(val)
    return values


def _count_values(d: Dict[str, str]) -> Dict[str, int]:
    """Count occurrences of each value in a dict."""
    counts: Dict[str, int] = {}
    for v in d.values():
        counts[v] = counts.get(v, 0) + 1
    return {k: counts[k] for k in sorted(counts.keys())}


def _compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Compute basic distribution statistics."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    std = math.sqrt(variance)
    
    return {
        "count": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


# ===========================================================================
# ATLAS HEALTH CHECK
# ===========================================================================

def evaluate_atlas_health(atlas: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check structural health of the atlas.
    
    This function validates that the atlas is structurally sound:
    - All slices present in fingerprints
    - Matrices are square and symmetric
    - Diagonal entries of JS matrix are approximately zero
    - No NaNs or infinities in matrices
    
    PHASE II — Structural validation only. No quality or performance claims.
    
    Args:
        atlas: Atlas dictionary (as loaded from behavior_atlas.json)
        
    Returns:
        Dictionary with (FROZEN CONTRACT):
            - status: "OK" | "WARN" | "BLOCK"
            - issues: List of issue descriptions
            - matrix_checks: Individual boolean results for each check
    """
    issues = []
    status = "OK"
    
    # Individual check results (FROZEN CONTRACT: matrix_checks)
    matrix_checks = {
        "fingerprints_complete": True,
        "matrices_square": True,
        "matrices_symmetric": True,
        "js_diagonal_zero": True,
        "no_invalid_values": True,
        "slice_count_match": True,
    }
    
    slice_names = atlas.get("slice_names", [])
    fingerprints = atlas.get("fingerprints", {})
    js_matrix = atlas.get("js_divergence_matrix", {})
    trend_matrix = atlas.get("trend_similarity_matrix", {})
    abstention_matrix = atlas.get("abstention_similarity_matrix", {})
    
    # Check 1: All slices present in fingerprints
    missing_fingerprints = [s for s in slice_names if s not in fingerprints]
    if missing_fingerprints:
        issues.append(f"Slices missing from fingerprints: {missing_fingerprints}")
        matrix_checks["fingerprints_complete"] = False
        status = "WARN"
    
    # Check 2: Matrices are square
    matrices_square = True
    for matrix_name, matrix in [
        ("js_divergence_matrix", js_matrix),
        ("trend_similarity_matrix", trend_matrix),
        ("abstention_similarity_matrix", abstention_matrix),
    ]:
        if not matrix:
            continue
        
        row_names = sorted(matrix.keys())
        for row_name in row_names:
            row = matrix.get(row_name, {})
            col_names = sorted(row.keys())
            if col_names != row_names:
                issues.append(f"{matrix_name} is not square: row '{row_name}' has columns {col_names} but expected {row_names}")
                matrices_square = False
                status = "BLOCK"
                break
    matrix_checks["matrices_square"] = matrices_square
    
    # Check 3: Matrices are symmetric
    matrices_symmetric = True
    for matrix_name, matrix in [
        ("js_divergence_matrix", js_matrix),
        ("trend_similarity_matrix", trend_matrix),
        ("abstention_similarity_matrix", abstention_matrix),
    ]:
        if not matrix:
            continue
        
        asymmetric_pairs = []
        names = sorted(matrix.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1:]:
                val1 = matrix.get(name1, {}).get(name2)
                val2 = matrix.get(name2, {}).get(name1)
                if val1 is not None and val2 is not None:
                    if abs(val1 - val2) > 1e-9:
                        asymmetric_pairs.append((name1, name2, val1, val2))
        
        if asymmetric_pairs:
            issues.append(f"{matrix_name} is not symmetric: {len(asymmetric_pairs)} asymmetric pairs found")
            matrices_symmetric = False
            status = "BLOCK"
    matrix_checks["matrices_symmetric"] = matrices_symmetric
    
    # Check 4: JS-divergence diagonal is approximately zero
    js_diagonal_ok = True
    if js_matrix:
        non_zero_diagonal = []
        for name in sorted(js_matrix.keys()):
            diag_val = js_matrix.get(name, {}).get(name)
            if diag_val is not None and abs(diag_val) > 1e-6:
                non_zero_diagonal.append((name, diag_val))
        
        if non_zero_diagonal:
            issues.append(f"JS-divergence diagonal has non-zero entries: {non_zero_diagonal}")
            js_diagonal_ok = False
            status = "WARN" if status == "OK" else status
    matrix_checks["js_diagonal_zero"] = js_diagonal_ok
    
    # Check 5: No NaNs or infinities
    no_invalid = True
    for matrix_name, matrix in [
        ("js_divergence_matrix", js_matrix),
        ("trend_similarity_matrix", trend_matrix),
        ("abstention_similarity_matrix", abstention_matrix),
    ]:
        if not matrix:
            continue
        
        invalid_values = []
        for row_name, row in matrix.items():
            for col_name, val in row.items():
                if val is None:
                    invalid_values.append((row_name, col_name, "None"))
                elif isinstance(val, float):
                    if math.isnan(val):
                        invalid_values.append((row_name, col_name, "NaN"))
                    elif math.isinf(val):
                        invalid_values.append((row_name, col_name, "Inf"))
        
        if invalid_values:
            issues.append(f"{matrix_name} contains invalid values: {invalid_values[:5]}{'...' if len(invalid_values) > 5 else ''}")
            no_invalid = False
            status = "BLOCK"
    matrix_checks["no_invalid_values"] = no_invalid
    
    # Check 6: Slice count matches actual slices
    declared_count = atlas.get("slice_count", 0)
    actual_count = len(slice_names)
    if declared_count != actual_count:
        issues.append(f"Declared slice_count ({declared_count}) does not match actual ({actual_count})")
        matrix_checks["slice_count_match"] = False
        status = "WARN" if status == "OK" else status
    
    # FROZEN CONTRACT OUTPUT
    return {
        "status": status,
        "issues": issues,
        "matrix_checks": matrix_checks,
        "checks_performed": [
            "fingerprints_complete",
            "matrices_square",
            "matrices_symmetric",
            "js_diagonal_zero",
            "no_invalid_values",
            "slice_count_match",
        ],
    }


def load_atlas_from_file(atlas_path: str) -> Dict[str, Any]:
    """Load an atlas from a JSON file."""
    with open(atlas_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ===========================================================================
# ATLAS ROUTING & ALIGNMENT LAYER (Phase II v1.2)
# ===========================================================================
# These helpers provide lightweight interfaces for downstream systems
# (C5, D4, governance) to query atlas state without deep coupling.
#
# All functions are:
# - Deterministic
# - Read-only (no side effects)
# - Descriptive (no ranking/value-loaded language)
# ===========================================================================

def build_routing_hint(dossier: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a small routing hint object from a dossier.
    
    This provides a compact, read-only view for routing / grouping decisions
    by other components. Contains no normative or value-loaded language.
    
    PHASE II CONTRACT — Advisory routing layer.
    
    Args:
        dossier: A slice dossier (from generate_slice_dossier)
        
    Returns:
        Routing hint dictionary (FROZEN CONTRACT):
        {
            "slice_name": str,
            "archetype": str,
            "neighbor_count": int,
            "feature_vector_dimension": int
        }
    """
    metrics = dossier.get("metrics_summary", {})
    
    return {
        "slice_name": dossier.get("slice_name", ""),
        "archetype": dossier.get("assigned_archetype", "unknown"),
        "neighbor_count": metrics.get("neighbor_count", 0),
        "feature_vector_dimension": metrics.get("feature_vector_dimension", 0),
    }


def compute_atlas_compatibility(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a compact compatibility overview from a real vs synthetic comparison.
    
    Provides summary counts for downstream systems to assess structural
    alignment between atlases. Contains no judgments, only counts.
    
    PHASE II CONTRACT — Compatibility overview layer.
    
    Args:
        comparison: Output from compare_real_vs_synthetic()
        
    Returns:
        Compatibility overview (FROZEN CONTRACT):
        {
            "schema_version": "1.0.0",
            "overlap_slice_count": int,
            "exact_archetype_match_count": int,
            "real_only_count": int,
            "synthetic_only_count": int
        }
    """
    slice_overlaps = comparison.get("slice_overlaps", {})
    slice_counts = comparison.get("slice_counts", {})
    
    # Count exact archetype matches
    exact_matches = sum(
        1 for overlap in slice_overlaps.values()
        if overlap.get("archetype_match", False)
    )
    
    return {
        "schema_version": "1.0.0",
        "overlap_slice_count": slice_counts.get("overlapping", 0),
        "exact_archetype_match_count": exact_matches,
        "real_only_count": slice_counts.get("real_only", 0),
        "synthetic_only_count": slice_counts.get("synthetic_only", 0),
    }


def is_atlas_structurally_sound(health: Dict[str, Any]) -> bool:
    """
    Minimal boolean predicate for "atlas structurally usable".
    
    This is a pure function with no side effects, intended for downstream
    systems to quickly gate on atlas validity.
    
    PHASE II CONTRACT — Structural soundness predicate.
    
    Definition:
        Returns True iff:
        - status != "BLOCK"
        - matrix_checks["matrices_square"] is True
        - matrix_checks["no_invalid_values"] is True
    
    Args:
        health: Output from evaluate_atlas_health()
        
    Returns:
        True if atlas is structurally sound, False otherwise
    """
    # BLOCK status is always unsound
    if health.get("status") == "BLOCK":
        return False
    
    # Check critical matrix properties
    matrix_checks = health.get("matrix_checks", {})
    
    if not matrix_checks.get("matrices_square", False):
        return False
    
    if not matrix_checks.get("no_invalid_values", False):
        return False
    
    return True


# ===========================================================================
# PHASE III — ATLAS GOVERNANCE & GLOBAL ROUTING LAYER
# ===========================================================================
# These functions provide governance snapshots and routing overviews for
# downstream systems (director console, global health checks).
#
# All functions are:
# - Deterministic
# - Read-only (no side effects)
# - Descriptive (no ranking/value-loaded language)
# ===========================================================================

def build_atlas_governance_snapshot(
    dossiers: Sequence[Dict[str, Any]],
    comparisons: Sequence[Dict[str, Any]],
    health_reports: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a governance snapshot from dossiers, comparisons, and health reports.
    
    This provides a high-level view of atlas state for governance decisions.
    Combines structural health, compatibility metrics, and diversity indices.
    
    PHASE III CONTRACT — Governance snapshot layer.
    
    Args:
        dossiers: Sequence of slice dossiers (from generate_slice_dossier)
        comparisons: Sequence of atlas comparisons (from compare_real_vs_synthetic)
        health_reports: Sequence of health reports (from evaluate_atlas_health)
        
    Returns:
        Governance snapshot (FROZEN CONTRACT):
        {
            "schema_version": "1.0.0",
            "total_slices_indexed": int,
            "real_vs_synthetic_overlap": dict (from compute_atlas_compatibility),
            "structurally_sound": bool,
            "archetype_diversity_index": float
        }
    """
    # Compute total slices indexed from dossiers
    total_slices = len(dossiers)
    
    # Aggregate real vs synthetic overlap from all comparisons
    # If multiple comparisons exist, we aggregate their compatibility metrics
    real_vs_synthetic_overlap = {
        "total_overlap_slices": 0,
        "total_exact_matches": 0,
        "total_real_only": 0,
        "total_synthetic_only": 0,
    }
    
    for comparison in comparisons:
        compat = compute_atlas_compatibility(comparison)
        real_vs_synthetic_overlap["total_overlap_slices"] += compat.get("overlap_slice_count", 0)
        real_vs_synthetic_overlap["total_exact_matches"] += compat.get("exact_archetype_match_count", 0)
        real_vs_synthetic_overlap["total_real_only"] += compat.get("real_only_count", 0)
        real_vs_synthetic_overlap["total_synthetic_only"] += compat.get("synthetic_only_count", 0)
    
    # Check if all health reports are structurally sound
    structurally_sound = all(
        is_atlas_structurally_sound(health) for health in health_reports
    )
    
    # Compute archetype diversity index
    # Number of distinct archetypes divided by total slices
    archetypes = set()
    for dossier in dossiers:
        archetype = dossier.get("assigned_archetype", "unknown")
        if archetype and archetype != "unknown":
            archetypes.add(archetype)
    
    distinct_archetypes = len(archetypes)
    diversity_index = (
        distinct_archetypes / total_slices if total_slices > 0 else 0.0
    )
    
    return {
        "schema_version": "1.0.0",
        "total_slices_indexed": total_slices,
        "real_vs_synthetic_overlap": real_vs_synthetic_overlap,
        "structurally_sound": structurally_sound,
        "archetype_diversity_index": round(diversity_index, 6),
    }


def build_routing_overview(
    routing_hints: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a routing overview from routing hints for director console.
    
    This provides a high-level view of slice distribution and clustering
    patterns for routing decisions. No normative language, only counts.
    
    PHASE III CONTRACT — Routing overview layer.
    
    Args:
        routing_hints: Sequence of routing hints (from build_routing_hint)
        
    Returns:
        Routing overview (FROZEN CONTRACT):
        {
            "slices_by_neighbor_band": {
                "low": int,
                "medium": int,
                "high": int
            },
            "archetype_frequency": dict[str, int],
            "status": "OK" | "SPARSE" | "CLUSTERED"
        }
    """
    if not routing_hints:
        return {
            "slices_by_neighbor_band": {"low": 0, "medium": 0, "high": 0},
            "archetype_frequency": {},
            "status": "SPARSE",
        }
    
    # Bucket slices by neighbor count
    # Thresholds: low < 3, medium 3-5, high > 5
    neighbor_bands = {"low": 0, "medium": 0, "high": 0}
    archetype_frequency: Dict[str, int] = {}
    
    for hint in routing_hints:
        neighbor_count = hint.get("neighbor_count", 0)
        
        if neighbor_count < 3:
            neighbor_bands["low"] += 1
        elif neighbor_count <= 5:
            neighbor_bands["medium"] += 1
        else:
            neighbor_bands["high"] += 1
        
        # Count archetypes
        archetype = hint.get("archetype", "unknown")
        archetype_frequency[archetype] = archetype_frequency.get(archetype, 0) + 1
    
    # Determine routing status
    total_slices = len(routing_hints)
    high_band_ratio = neighbor_bands["high"] / total_slices if total_slices > 0 else 0.0
    low_band_ratio = neighbor_bands["low"] / total_slices if total_slices > 0 else 1.0
    
    # Status logic:
    # - CLUSTERED: > 50% of slices have high neighbor counts (>5)
    # - SPARSE: > 50% of slices have low neighbor counts (<3)
    # - OK: otherwise (balanced distribution)
    if high_band_ratio > 0.5:
        status = "CLUSTERED"
    elif low_band_ratio > 0.5:
        status = "SPARSE"
    else:
        status = "OK"
    
    return {
        "slices_by_neighbor_band": neighbor_bands,
        "archetype_frequency": archetype_frequency,
        "status": status,
    }


def summarize_atlas_for_global_health(
    governance_snapshot: Dict[str, Any],
    routing_overview: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize atlas state for global health checks.
    
    This combines governance and routing views into a single health summary
    for downstream systems. Provides clear OK/WARN/BLOCK status.
    
    PHASE III CONTRACT — Global health summary layer.
    
    Args:
        governance_snapshot: Output from build_atlas_governance_snapshot
        routing_overview: Output from build_routing_overview
        
    Returns:
        Global health summary (FROZEN CONTRACT):
        {
            "atlas_ok": bool,
            "structurally_sound": bool,
            "routing_status": str,
            "status": "OK" | "WARN" | "BLOCK"
        }
    """
    structurally_sound = governance_snapshot.get("structurally_sound", False)
    routing_status = routing_overview.get("status", "SPARSE")
    
    # Determine overall status
    # BLOCK: structurally unsound
    # WARN: structurally sound but routing issues (SPARSE) or no slices
    # OK: structurally sound and routing OK or CLUSTERED
    if not structurally_sound:
        status = "BLOCK"
    elif routing_status == "SPARSE" or governance_snapshot.get("total_slices_indexed", 0) == 0:
        status = "WARN"
    else:
        status = "OK"
    
    atlas_ok = (status == "OK")
    
    return {
        "atlas_ok": atlas_ok,
        "structurally_sound": structurally_sound,
        "routing_status": routing_status,
        "status": status,
    }


# ===========================================================================
# PHASE IV — ATLAS-GUIDED ROUTING & CROSS-SLICE STRUCTURAL GOVERNANCE
# ===========================================================================
# These functions provide routing policies and structural governance views
# for director-level decision making and cross-slice coordination.
#
# All functions are:
# - Deterministic
# - Read-only (no side effects)
# - Descriptive (no ranking/value-loaded language)
# ===========================================================================

def derive_atlas_routing_policy(
    governance_snapshot: Dict[str, Any],
    routing_overview: Dict[str, Any],
    dossiers: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Derive an atlas-guided routing policy from governance snapshot and routing overview.
    
    This provides routing preferences based on archetype patterns and slice distribution.
    Purely descriptive - no normative judgments about which routing is "better".
    
    PHASE IV CONTRACT — Routing policy layer.
    
    Args:
        governance_snapshot: Output from build_atlas_governance_snapshot
        routing_overview: Output from build_routing_overview
        dossiers: Optional sequence of dossiers to extract actual slice names
        
    Returns:
        Routing policy (FROZEN CONTRACT):
        {
            "slices_preferring_dense_archetypes": list[str],
            "slices_preferring_sparse_archetypes": list[str],
            "routing_status": "BALANCED" | "SPARSE" | "CLUSTERED",
            "policy_notes": str
        }
    """
    routing_status = routing_overview.get("status", "OK")
    
    # Extract slice names by archetype from dossiers if available
    slices_preferring_dense = []
    slices_preferring_sparse = []
    
    if dossiers:
        for dossier in dossiers:
            slice_name = dossier.get("slice_name", "")
            archetype = dossier.get("assigned_archetype", "unknown")
            
            if "dense" in archetype.lower():
                slices_preferring_dense.append(slice_name)
            elif "sparse" in archetype.lower():
                slices_preferring_sparse.append(slice_name)
    
    # Sort for determinism
    slices_preferring_dense = sorted(slices_preferring_dense)
    slices_preferring_sparse = sorted(slices_preferring_sparse)
    
    # Map routing_status to policy routing_status
    # OK/CLUSTERED -> CLUSTERED, SPARSE -> SPARSE, otherwise BALANCED
    if routing_status in ["OK", "CLUSTERED"]:
        policy_routing_status = "CLUSTERED"
    elif routing_status == "SPARSE":
        policy_routing_status = "SPARSE"
    else:
        policy_routing_status = "BALANCED"
    
    # Generate neutral policy notes
    archetype_freq = routing_overview.get("archetype_frequency", {})
    total_archetypes = len(archetype_freq)
    diversity_index = governance_snapshot.get("archetype_diversity_index", 0.0)
    
    if total_archetypes == 0:
        policy_notes = "No archetype data available for routing guidance."
    elif diversity_index > 0.7:
        policy_notes = f"Atlas shows high archetype diversity ({total_archetypes} distinct types)."
    elif diversity_index < 0.3:
        policy_notes = f"Atlas shows low archetype diversity ({total_archetypes} distinct types)."
    else:
        policy_notes = f"Atlas shows moderate archetype diversity ({total_archetypes} distinct types)."
    
    return {
        "slices_preferring_dense_archetypes": slices_preferring_dense,
        "slices_preferring_sparse_archetypes": slices_preferring_sparse,
        "routing_status": policy_routing_status,
        "policy_notes": policy_notes,
    }


def build_structural_governance_view(
    atlas_snapshot: Dict[str, Any],
    topology_analytics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a cross-slice structural governance view.
    
    This combines atlas snapshot data with topology analytics from C5
    to identify structural patterns and governance status across slices.
    
    PHASE IV CONTRACT — Structural governance layer.
    
    Args:
        atlas_snapshot: Governance snapshot (from build_atlas_governance_snapshot)
        topology_analytics: Topology analytics from C5 agent
                          (expected to have structural metrics)
        
    Returns:
        Structural governance view (FROZEN CONTRACT):
        {
            "slices_with_structure_vs_routing_mismatch": list[str],
            "slices_with_consistent_archetypes": list[str],
            "governance_status": "OK" | "ATTENTION" | "VOLATILE"
        }
    """
    # Extract topology stability/consistency metrics
    # topology_analytics may contain: stability_index, consistency_flags, etc.
    stability_index = topology_analytics.get("stability_index", 1.0)
    consistency_flags = topology_analytics.get("consistency_flags", {})
    slice_consistency = topology_analytics.get("slice_consistency", {})
    
    # Identify slices with structural mismatches
    # This would compare routing hints vs topology structure
    # For now, we use consistency flags from topology_analytics
    mismatch_slices = []
    if isinstance(consistency_flags, dict):
        mismatch_slices = [
            slice_name for slice_name, flag in consistency_flags.items()
            if flag is False or flag == "mismatch"
        ]
    
    # Identify slices with consistent archetypes
    consistent_slices = []
    if isinstance(slice_consistency, dict):
        consistent_slices = [
            slice_name for slice_name, consistent in slice_consistency.items()
            if consistent is True
        ]
    
    # Determine governance status based on stability and structure
    # VOLATILE: Low stability (< 0.5)
    # ATTENTION: Moderate stability (0.5-0.8) or mismatches present
    # OK: High stability (> 0.8) and no mismatches
    if stability_index < 0.5:
        governance_status = "VOLATILE"
    elif stability_index < 0.8 or len(mismatch_slices) > 0:
        governance_status = "ATTENTION"
    else:
        governance_status = "OK"
    
    # If topology_analytics doesn't have slice-level data, return empty lists
    # but still compute governance_status from available metrics
    if not mismatch_slices and not consistent_slices:
        # Fallback: use archetype diversity to infer consistency
        diversity = atlas_snapshot.get("archetype_diversity_index", 0.0)
        if diversity > 0.7:
            # High diversity might indicate more varied (potentially inconsistent) patterns
            governance_status = "ATTENTION" if governance_status == "OK" else governance_status
    
    return {
        "slices_with_structure_vs_routing_mismatch": sorted(mismatch_slices),
        "slices_with_consistent_archetypes": sorted(consistent_slices),
        "governance_status": governance_status,
    }


def build_atlas_director_panel(
    governance_snapshot: Dict[str, Any],
    routing_policy: Dict[str, Any],
    structural_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a director-facing atlas panel combining all governance views.
    
    This provides a single unified view for director-level decision making,
    combining governance, routing, and structural perspectives.
    
    PHASE IV CONTRACT — Director panel layer.
    
    Args:
        governance_snapshot: Output from build_atlas_governance_snapshot
        routing_policy: Output from derive_atlas_routing_policy
        structural_view: Output from build_structural_governance_view
        
    Returns:
        Director panel (FROZEN CONTRACT):
        {
            "status_light": "GREEN" | "YELLOW" | "RED",
            "atlas_ok": bool,
            "structurally_sound": bool,
            "routing_status": str,
            "governance_status": str,
            "headline": str
        }
    """
    structurally_sound = governance_snapshot.get("structurally_sound", False)
    routing_status = routing_policy.get("routing_status", "BALANCED")
    governance_status = structural_view.get("governance_status", "OK")
    
    # Determine status light based on all indicators
    # RED: Structurally unsound OR VOLATILE governance
    # YELLOW: ATTENTION governance OR SPARSE routing OR not structurally sound in some way
    # GREEN: Structurally sound, OK governance, and CLUSTERED/BALANCED routing
    if not structurally_sound or governance_status == "VOLATILE":
        status_light = "RED"
    elif governance_status == "ATTENTION" or routing_status == "SPARSE":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Compute atlas_ok (similar to global health summary)
    atlas_ok = (
        structurally_sound and
        governance_status == "OK" and
        routing_status != "SPARSE"
    )
    
    # Generate neutral headline
    total_slices = governance_snapshot.get("total_slices_indexed", 0)
    diversity_index = governance_snapshot.get("archetype_diversity_index", 0.0)
    
    if total_slices == 0:
        headline = "Atlas contains no indexed slices."
    elif not structurally_sound:
        headline = f"Atlas structural health check indicates issues with {total_slices} indexed slices."
    elif governance_status == "VOLATILE":
        headline = f"Atlas shows volatile structural patterns across {total_slices} slices."
    elif governance_status == "ATTENTION":
        headline = f"Atlas requires attention with {total_slices} slices and {diversity_index:.2f} diversity index."
    elif routing_status == "SPARSE":
        headline = f"Atlas shows sparse routing patterns across {total_slices} slices."
    else:
        headline = f"Atlas operating normally with {total_slices} slices organized into {diversity_index:.2f} diversity index."
    
    return {
        "status_light": status_light,
        "atlas_ok": atlas_ok,
        "structurally_sound": structurally_sound,
        "routing_status": routing_status,
        "governance_status": governance_status,
        "headline": headline,
    }


# ===========================================================================
# ATLAS-CURRICULUM COUPLER & PHASE TRANSITION ADVISOR
# ===========================================================================
# These functions bind atlas structure directly into curriculum decisions
# and provide phase transition guidance.
#
# All functions are:
# - Deterministic
# - Read-only (no side effects)
# - Descriptive (no ranking/value-loaded language)
# ===========================================================================

def build_atlas_curriculum_coupling_view(
    atlas_governance: Dict[str, Any],
    curriculum_alignment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a view of how atlas structure couples with curriculum alignment.
    
    This identifies which slices have atlas support and which need more
    structural coverage. Purely descriptive - no judgments about quality.
    
    PHASE IV CONTRACT — Atlas-curriculum coupling layer.
    
    Args:
        atlas_governance: Governance snapshot (from build_atlas_governance_snapshot)
        curriculum_alignment: Curriculum alignment data
                          (expected to have slice names and alignment status)
        
    Returns:
        Coupling view (FROZEN CONTRACT):
        {
            "slices_with_atlas_support": list[str],
            "slices_without_atlas_support": list[str],
            "coupling_status": "TIGHT" | "LOOSE" | "MISSING",
            "neutral_notes": list[str]
        }
    """
    # Extract slice information from curriculum alignment
    # Expected structure: curriculum_alignment may have:
    # - "slices": list of slice names
    # - "aligned_slices": list of aligned slice names
    # - "slice_alignment": dict mapping slice_name -> alignment_status
    
    curriculum_slices = set()
    if "slices" in curriculum_alignment:
        curriculum_slices = set(curriculum_alignment["slices"])
    elif "slice_alignment" in curriculum_alignment:
        curriculum_slices = set(curriculum_alignment["slice_alignment"].keys())
    
    # Get atlas-indexed slices from governance
    total_atlas_slices = atlas_governance.get("total_slices_indexed", 0)
    
    # For now, we'll use a simple heuristic:
    # If curriculum_alignment has slice_alignment dict, slices with True/OK status
    # are considered to have atlas support
    slices_with_support = []
    slices_without_support = []
    
    if "slice_alignment" in curriculum_alignment:
        slice_alignment = curriculum_alignment["slice_alignment"]
        for slice_name, alignment_status in slice_alignment.items():
            # Consider aligned/OK/True as having support
            if alignment_status in [True, "OK", "aligned", "supported"]:
                slices_with_support.append(slice_name)
            else:
                slices_without_support.append(slice_name)
    elif "aligned_slices" in curriculum_alignment:
        # Alternative structure: explicit aligned_slices list
        aligned = set(curriculum_alignment.get("aligned_slices", []))
        slices_with_support = sorted(list(curriculum_slices & aligned))
        slices_without_support = sorted(list(curriculum_slices - aligned))
    else:
        # Fallback: if we can't determine, assume all curriculum slices need support
        slices_without_support = sorted(list(curriculum_slices))
    
    # Determine coupling status
    total_curriculum_slices = len(curriculum_slices)
    supported_count = len(slices_with_support)
    
    if total_curriculum_slices == 0:
        coupling_status = "MISSING"
    elif supported_count == 0:
        coupling_status = "MISSING"
    elif supported_count / total_curriculum_slices >= 0.8:
        coupling_status = "TIGHT"
    elif supported_count / total_curriculum_slices >= 0.5:
        coupling_status = "LOOSE"
    else:
        coupling_status = "MISSING"
    
    # Generate neutral notes
    notes = []
    if total_curriculum_slices > 0:
        coverage_ratio = supported_count / total_curriculum_slices
        notes.append(f"Curriculum contains {total_curriculum_slices} slices.")
        notes.append(f"Atlas provides support for {supported_count} slices ({coverage_ratio:.1%} coverage).")
    else:
        notes.append("No curriculum slices identified for coupling analysis.")
    
    if total_atlas_slices > 0:
        notes.append(f"Atlas indexes {total_atlas_slices} slices total.")
    
    return {
        "slices_with_atlas_support": sorted(slices_with_support),
        "slices_without_atlas_support": sorted(slices_without_support),
        "coupling_status": coupling_status,
        "neutral_notes": notes,
    }


def derive_phase_transition_advice(
    coupling_view: Dict[str, Any],
    structural_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Derive phase transition advice based on coupling and structural views.
    
    This provides guidance on whether phase transitions are safe and which
    slices are ready for phase upgrades. Purely advisory - no enforcement.
    
    PHASE IV CONTRACT — Phase transition advisor layer.
    
    Args:
        coupling_view: Output from build_atlas_curriculum_coupling_view
        structural_view: Output from build_structural_governance_view
        
    Returns:
        Phase transition advice (FROZEN CONTRACT):
        {
            "phase_transition_safe": bool,
            "status": "OK" | "ATTENTION" | "BLOCK",
            "suggested_slices_for_phase_upgrade": list[str],
            "slices_needing_more_atlas_support": list[str],
            "headline": str
        }
    """
    coupling_status = coupling_view.get("coupling_status", "MISSING")
    governance_status = structural_view.get("governance_status", "OK")
    slices_with_support = coupling_view.get("slices_with_atlas_support", [])
    slices_without_support = coupling_view.get("slices_without_atlas_support", [])
    
    # Determine phase_transition_safe
    # Safe if: TIGHT coupling AND OK governance
    phase_transition_safe = (
        coupling_status == "TIGHT" and
        governance_status == "OK"
    )
    
    # Determine status
    # BLOCK: MISSING coupling OR VOLATILE governance
    # ATTENTION: LOOSE coupling OR ATTENTION governance
    # OK: TIGHT coupling AND OK governance
    if coupling_status == "MISSING" or governance_status == "VOLATILE":
        status = "BLOCK"
    elif coupling_status == "LOOSE" or governance_status == "ATTENTION":
        status = "ATTENTION"
    else:
        status = "OK"
    
    # Slices ready for phase upgrade are those with atlas support
    # and consistent archetypes
    consistent_slices = set(structural_view.get("slices_with_consistent_archetypes", []))
    suggested_slices = sorted(list(set(slices_with_support) & consistent_slices))
    
    # If no consistent slices overlap, use all slices with support
    if not suggested_slices and slices_with_support:
        suggested_slices = sorted(slices_with_support)
    
    # Slices needing more support are those without atlas support
    slices_needing_support = sorted(slices_without_support)
    
    # Generate neutral headline
    total_with_support = len(slices_with_support)
    total_without_support = len(slices_without_support)
    
    if status == "BLOCK":
        if coupling_status == "MISSING":
            headline = f"Phase transition blocked: {total_without_support} slices lack atlas support."
        else:
            headline = f"Phase transition blocked: structural governance status is volatile."
    elif status == "ATTENTION":
        headline = f"Phase transition requires attention: {total_without_support} slices need atlas support, {total_with_support} have support."
    else:
        headline = f"Phase transition appears safe: {total_with_support} slices have atlas support and consistent structure."
    
    return {
        "phase_transition_safe": phase_transition_safe,
        "status": status,
        "suggested_slices_for_phase_upgrade": suggested_slices,
        "slices_needing_more_atlas_support": slices_needing_support,
        "headline": headline,
    }


# ===========================================================================
# PHASE VI — ATLAS CONVERGENCE LATTICE & PHASE TRANSITION GATE v2
# ===========================================================================
# These functions provide lattice-level convergence geometry and enhanced
# phase transition gating with uplift safety inference.
#
# All functions are:
# - Deterministic
# - Read-only (no side effects)
# - Descriptive (no ranking/value-loaded language)
# ===========================================================================

def build_atlas_convergence_lattice(
    routing_policy: Dict[str, Any],
    structural_view: Dict[str, Any],
    curriculum_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an atlas convergence lattice representing geometric alignment of slices.
    
    The lattice represents how well slices converge across routing, structure,
    and curriculum dimensions. Higher lattice vectors indicate better alignment.
    
    PHASE VI CONTRACT — Convergence lattice layer.
    
    Args:
        routing_policy: Output from derive_atlas_routing_policy
        structural_view: Output from build_structural_governance_view
        curriculum_view: Output from build_atlas_curriculum_coupling_view
        
    Returns:
        Convergence lattice (FROZEN CONTRACT):
        {
            "lattice_vectors": { "slice_name": float },
            "global_lattice_norm": float,
            "convergence_band": "COHERENT" | "PARTIAL" | "MISALIGNED",
            "neutral_notes": list[str]
        }
    """
    # Collect all unique slice names from all three views
    all_slices = set()
    
    # From routing policy
    dense_slices = set(routing_policy.get("slices_preferring_dense_archetypes", []))
    sparse_slices = set(routing_policy.get("slices_preferring_sparse_archetypes", []))
    all_slices.update(dense_slices)
    all_slices.update(sparse_slices)
    
    # From structural view
    consistent_slices = set(structural_view.get("slices_with_consistent_archetypes", []))
    mismatch_slices = set(structural_view.get("slices_with_structure_vs_routing_mismatch", []))
    all_slices.update(consistent_slices)
    all_slices.update(mismatch_slices)
    
    # From curriculum view
    supported_slices = set(curriculum_view.get("slices_with_atlas_support", []))
    unsupported_slices = set(curriculum_view.get("slices_without_atlas_support", []))
    all_slices.update(supported_slices)
    all_slices.update(unsupported_slices)
    
    # Compute lattice vector for each slice
    # Vector components:
    # - Routing alignment: 1.0 if in dense/sparse lists, 0.5 if neither, 0.0 if mismatch
    # - Structural alignment: 1.0 if consistent, 0.0 if mismatch
    # - Curriculum alignment: 1.0 if supported, 0.0 if unsupported
    # Final vector = normalized average of components
    lattice_vectors = {}
    
    for slice_name in sorted(all_slices):
        routing_score = 0.0
        structural_score = 0.0
        curriculum_score = 0.0
        
        # Routing component
        if slice_name in dense_slices or slice_name in sparse_slices:
            routing_score = 1.0
        elif slice_name in mismatch_slices:
            routing_score = 0.0
        else:
            routing_score = 0.5  # Neutral/unknown
        
        # Structural component
        if slice_name in consistent_slices:
            structural_score = 1.0
        elif slice_name in mismatch_slices:
            structural_score = 0.0
        else:
            structural_score = 0.5  # Neutral/unknown
        
        # Curriculum component
        if slice_name in supported_slices:
            curriculum_score = 1.0
        elif slice_name in unsupported_slices:
            curriculum_score = 0.0
        else:
            curriculum_score = 0.5  # Neutral/unknown
        
        # Compute normalized vector (L2 norm of 3D vector)
        vector_norm = math.sqrt(
            routing_score ** 2 + structural_score ** 2 + curriculum_score ** 2
        )
        # Normalize to [0, 1] range (max possible norm is sqrt(3) ≈ 1.732)
        normalized_vector = vector_norm / math.sqrt(3.0)
        lattice_vectors[slice_name] = round(normalized_vector, 6)
    
    # Compute global lattice norm (L2 norm of all slice vectors)
    if lattice_vectors:
        vector_values = list(lattice_vectors.values())
        global_norm = math.sqrt(sum(v ** 2 for v in vector_values))
        # Normalize by number of slices
        normalized_global_norm = global_norm / math.sqrt(len(vector_values))
    else:
        normalized_global_norm = 0.0
    
    global_lattice_norm = round(normalized_global_norm, 6)
    
    # Determine convergence band
    # COHERENT: global_norm >= 0.8 (high alignment)
    # PARTIAL: 0.5 <= global_norm < 0.8 (moderate alignment)
    # MISALIGNED: global_norm < 0.5 (low alignment)
    if global_lattice_norm >= 0.8:
        convergence_band = "COHERENT"
    elif global_lattice_norm >= 0.5:
        convergence_band = "PARTIAL"
    else:
        convergence_band = "MISALIGNED"
    
    # Generate neutral notes
    notes = []
    total_slices = len(lattice_vectors)
    if total_slices > 0:
        notes.append(f"Lattice computed for {total_slices} slices.")
        notes.append(f"Global lattice norm: {global_lattice_norm:.3f}.")
        notes.append(f"Convergence band: {convergence_band}.")
        
        # Count slices by vector strength
        high_vectors = sum(1 for v in lattice_vectors.values() if v >= 0.8)
        medium_vectors = sum(1 for v in lattice_vectors.values() if 0.5 <= v < 0.8)
        low_vectors = sum(1 for v in lattice_vectors.values() if v < 0.5)
        
        notes.append(f"High alignment slices: {high_vectors}, medium: {medium_vectors}, low: {low_vectors}.")
    else:
        notes.append("No slices available for lattice computation.")
    
    return {
        "lattice_vectors": lattice_vectors,
        "global_lattice_norm": global_lattice_norm,
        "convergence_band": convergence_band,
        "neutral_notes": notes,
    }


def derive_atlas_phase_transition_gate(
    lattice: Dict[str, Any],
    phase_transition_advice: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Derive enhanced phase transition gate using convergence lattice.
    
    This combines lattice coherence with phase transition advice to provide
    a more nuanced gating mechanism with uplift safety inference.
    
    PHASE VI CONTRACT — Phase transition gate v2 layer.
    
    Args:
        lattice: Output from build_atlas_convergence_lattice
        phase_transition_advice: Output from derive_phase_transition_advice
        
    Returns:
        Phase transition gate (FROZEN CONTRACT):
        {
            "transition_status": "OK" | "ATTENTION" | "BLOCK",
            "drivers": list[str],
            "slices_ready": list[str],
            "slices_needing_alignment": list[str],
            "headline": str
        }
    """
    convergence_band = lattice.get("convergence_band", "MISALIGNED")
    global_norm = lattice.get("global_lattice_norm", 0.0)
    lattice_vectors = lattice.get("lattice_vectors", {})
    
    phase_status = phase_transition_advice.get("status", "BLOCK")
    phase_safe = phase_transition_advice.get("phase_transition_safe", False)
    suggested_slices = set(phase_transition_advice.get("suggested_slices_for_phase_upgrade", []))
    slices_needing_support = set(phase_transition_advice.get("slices_needing_more_atlas_support", []))
    
    # Determine transition status
    # BLOCK: phase_status is BLOCK OR convergence_band is MISALIGNED
    # ATTENTION: phase_status is ATTENTION OR convergence_band is PARTIAL
    # OK: phase_status is OK AND convergence_band is COHERENT
    if phase_status == "BLOCK" or convergence_band == "MISALIGNED":
        transition_status = "BLOCK"
    elif phase_status == "ATTENTION" or convergence_band == "PARTIAL":
        transition_status = "ATTENTION"
    else:
        transition_status = "OK"
    
    # Identify drivers (factors influencing the status)
    drivers = []
    if phase_status == "BLOCK":
        drivers.append("Phase transition advice indicates blocking conditions")
    if convergence_band == "MISALIGNED":
        drivers.append("Lattice convergence band is misaligned")
    elif convergence_band == "PARTIAL":
        drivers.append("Lattice convergence band is partial")
    if global_norm < 0.5:
        drivers.append(f"Global lattice norm is low ({global_norm:.3f})")
    if not phase_safe:
        drivers.append("Phase transition safety check failed")
    
    if not drivers:
        drivers.append("All checks passed")
    
    # Slices ready: intersection of suggested slices and high-lattice-vector slices
    high_vector_slices = {
        slice_name for slice_name, vector in lattice_vectors.items()
        if vector >= 0.8
    }
    slices_ready = sorted(list(suggested_slices & high_vector_slices))
    
    # If no intersection, use suggested slices as fallback
    if not slices_ready and suggested_slices:
        slices_ready = sorted(list(suggested_slices))
    
    # Slices needing alignment: slices with low lattice vectors OR needing support
    low_vector_slices = {
        slice_name for slice_name, vector in lattice_vectors.items()
        if vector < 0.5
    }
    slices_needing_alignment = sorted(list(slices_needing_support | low_vector_slices))
    
    # Generate neutral headline
    if transition_status == "BLOCK":
        if convergence_band == "MISALIGNED":
            headline = f"Phase transition blocked: lattice convergence is misaligned (norm: {global_norm:.3f})."
        else:
            headline = f"Phase transition blocked: phase advice indicates blocking conditions."
    elif transition_status == "ATTENTION":
        headline = f"Phase transition requires attention: {len(slices_needing_alignment)} slices need alignment, {len(slices_ready)} slices ready."
    else:
        headline = f"Phase transition appears safe: {len(slices_ready)} slices ready with coherent lattice convergence (norm: {global_norm:.3f})."
    
    return {
        "transition_status": transition_status,
        "drivers": drivers,
        "slices_ready": slices_ready,
        "slices_needing_alignment": slices_needing_alignment,
        "headline": headline,
    }


def build_atlas_director_tile_v2(
    governance_snapshot: Dict[str, Any],
    routing_policy: Dict[str, Any],
    structural_view: Dict[str, Any],
    curriculum_view: Dict[str, Any],
    phase_transition_advice: Dict[str, Any],
    lattice: Dict[str, Any],
    phase_gate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build enhanced director tile v2 with lattice coherence and transition recommendations.
    
    This provides a comprehensive director-facing view combining all atlas dimensions
    including the new convergence lattice geometry.
    
    PHASE VI CONTRACT — Director tile v2 layer.
    
    Args:
        governance_snapshot: Output from build_atlas_governance_snapshot
        routing_policy: Output from derive_atlas_routing_policy
        structural_view: Output from build_structural_governance_view
        curriculum_view: Output from build_atlas_curriculum_coupling_view
        phase_transition_advice: Output from derive_phase_transition_advice
        lattice: Output from build_atlas_convergence_lattice
        phase_gate: Output from derive_atlas_phase_transition_gate
        
    Returns:
        Director tile v2 (FROZEN CONTRACT):
        {
            "status_light": "GREEN" | "YELLOW" | "RED",
            "lattice_coherence": str,
            "structural_status": str,
            "transition_recommendation": str,
            "atlas_ok": bool,
            "headline": str
        }
    """
    structurally_sound = governance_snapshot.get("structurally_sound", False)
    governance_status = structural_view.get("governance_status", "OK")
    routing_status = routing_policy.get("routing_status", "BALANCED")
    coupling_status = curriculum_view.get("coupling_status", "MISSING")
    convergence_band = lattice.get("convergence_band", "MISALIGNED")
    transition_status = phase_gate.get("transition_status", "BLOCK")
    
    # Determine status light
    # RED: Structurally unsound OR VOLATILE governance OR BLOCK transition OR MISALIGNED lattice
    # YELLOW: ATTENTION governance OR PARTIAL lattice OR ATTENTION transition
    # GREEN: All systems healthy
    if (not structurally_sound or 
        governance_status == "VOLATILE" or 
        transition_status == "BLOCK" or 
        convergence_band == "MISALIGNED"):
        status_light = "RED"
    elif (governance_status == "ATTENTION" or 
          convergence_band == "PARTIAL" or 
          transition_status == "ATTENTION"):
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Compute atlas_ok
    atlas_ok = (
        structurally_sound and
        governance_status == "OK" and
        convergence_band == "COHERENT" and
        transition_status == "OK"
    )
    
    # Generate transition recommendation
    if transition_status == "BLOCK":
        transition_recommendation = "BLOCK: Do not proceed with phase transition"
    elif transition_status == "ATTENTION":
        transition_recommendation = "ATTENTION: Review alignment before phase transition"
    else:
        transition_recommendation = "OK: Phase transition appears safe"
    
    # Generate neutral headline
    total_slices = governance_snapshot.get("total_slices_indexed", 0)
    global_norm = lattice.get("global_lattice_norm", 0.0)
    slices_ready = len(phase_gate.get("slices_ready", []))
    slices_needing_alignment = len(phase_gate.get("slices_needing_alignment", []))
    
    if status_light == "RED":
        headline = f"Atlas director tile: {total_slices} slices, lattice norm {global_norm:.3f} ({convergence_band}), transition {transition_status}."
    elif status_light == "YELLOW":
        headline = f"Atlas director tile: {total_slices} slices, {slices_ready} ready, {slices_needing_alignment} need alignment, lattice {convergence_band}."
    else:
        headline = f"Atlas director tile: {total_slices} slices, {slices_ready} ready, lattice coherent (norm: {global_norm:.3f}), transition safe."
    
    return {
        "status_light": status_light,
        "lattice_coherence": convergence_band,
        "structural_status": governance_status,
        "transition_recommendation": transition_recommendation,
        "atlas_ok": atlas_ok,
        "headline": headline,
    }


# ===========================================================================
# TELEMETRY PACK INTEGRATION (D4 Hook)
# ===========================================================================

# NOTE: This section provides integration between the Behavior Atlas (D6)
# and the Behavioral Telemetry Visualization Suite (D4).
#
# The atlas provides STRUCTURAL analysis (clustering, fingerprints, matrices).
# The telemetry pack provides TEMPORAL analysis (trajectories, heatmaps).
#
# Together they give a comprehensive behavioral profile of experiment runs.
#
# ===========================================================================
# ATLAS + TELEMETRY BRIDGE: Developer Workflow
# ===========================================================================
#
# The recommended workflow for comprehensive behavioral analysis:
#
#   1. Build the Behavior Atlas (structural clustering):
#
#       from pathlib import Path
#       from experiments.u2_behavior_atlas import build_behavior_atlas
#
#       atlas, baseline_records = build_behavior_atlas(
#           input_dir=Path('results'),
#           n_clusters=4,
#           clustering_seed=42,
#       )
#       # Outputs: atlas.slice_profiles, atlas.js_divergence_matrix, atlas.archetypes
#
#   2. Generate the Telemetry Pack (temporal visualization):
#
#       from experiments.behavioral_telemetry_viz import generate_telemetry_pack
#
#       pack_meta = generate_telemetry_pack(
#           baseline_path='results/fo_baseline.jsonl',
#           rfl_path='results/fo_rfl.jsonl',
#           out_dir='artifacts/telemetry',
#       )
#       # Outputs: abstention_heatmap.png, chain_depth_density.png,
#       #          candidate_entropy.png, metric_volatility.png,
#       #          pack_index.json, telemetry_manifest.json
#
#   3. Inspect both artifacts together:
#
#       # Atlas artifacts:
#       #   artifacts/atlas/behavior_atlas.json     -- full atlas with matrices
#       #   artifacts/atlas/fingerprints.json       -- per-slice fingerprint hashes
#       #   artifacts/atlas/js_divergence_heatmap.png
#       #   artifacts/atlas/trend_similarity_heatmap.png
#       #   artifacts/atlas/archetype_distribution.png
#       #
#       # Telemetry artifacts:
#       #   artifacts/telemetry/abstention_heatmap.png
#       #   artifacts/telemetry/chain_depth_density.png
#       #   artifacts/telemetry/candidate_entropy.png
#       #   artifacts/telemetry/metric_volatility.png
#       #   artifacts/telemetry/pack_index.json
#       #   artifacts/telemetry/telemetry_manifest.json
#
#   4. Use the combined analysis helper for one-shot generation:
#
#       from experiments.u2_behavior_atlas import generate_combined_analysis
#
#       meta = generate_combined_analysis(
#           input_dir=Path('results'),
#           baseline_jsonl='results/fo_baseline.jsonl',
#           rfl_jsonl='results/fo_rfl.jsonl',
#           out_dir=Path('artifacts/combined'),
#       )
#       # Outputs both atlas/ and telemetry/ subdirectories
#
# IMPORTANT: All outputs are purely descriptive. They visualize behavioral
# patterns without making claims about performance, uplift, or correctness.
#
# See docs/TELEMETRY_PLAYBOOK.md for usage recipes and interpretation guidance.
# ===========================================================================


def generate_combined_analysis(
    input_dir: Path,
    baseline_jsonl: str,
    rfl_jsonl: str,
    out_dir: Path,
    n_clusters: int = 4,
    seed: int = 42,
    window: int = 20,
) -> Dict[str, Any]:
    """
    Generate combined atlas + telemetry pack for comprehensive analysis.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    This helper combines:
    - Behavior Atlas: structural clustering, fingerprints, similarity matrices
    - Telemetry Pack: temporal visualizations (heatmaps, trajectories, volatility)
    
    Args:
        input_dir: Directory containing slice JSONL files for atlas.
        baseline_jsonl: Path to baseline JSONL for telemetry.
        rfl_jsonl: Path to RFL JSONL for telemetry.
        out_dir: Output directory for combined analysis.
        n_clusters: Number of archetype clusters.
        seed: Random seed for clustering.
        window: Rolling window size for telemetry plots.
    
    Returns:
        Combined metadata dictionary.
    
    Example:
        >>> from pathlib import Path
        >>> meta = generate_combined_analysis(
        ...     Path('results'),
        ...     'results/fo_baseline.jsonl',
        ...     'results/fo_rfl.jsonl',
        ...     Path('artifacts/combined')
        ... )
        >>> print(meta['atlas_hash'][:16], meta['telemetry_manifest_hash'][:16])
    """
    # Import telemetry pack generator (lazy to avoid circular imports)
    from experiments.behavioral_telemetry_viz import generate_telemetry_pack
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Behavior Atlas
    atlas_dir = out_dir / "atlas"
    atlas, baseline_records = build_behavior_atlas(
        input_dir,
        n_clusters=n_clusters,
        clustering_seed=seed,
    )
    
    atlas_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = atlas_dir / "behavior_atlas.json"
    with open(atlas_path, 'w', encoding='utf-8') as f:
        json.dump(atlas_to_dict(atlas), f, indent=2, sort_keys=True)
    
    # Generate atlas heatmaps
    if atlas.slice_profiles:
        generate_heatmap(
            atlas.js_divergence_matrix,
            "JS-Divergence Matrix",
            atlas_dir / "js_divergence_heatmap.png",
            cmap="YlOrRd", vmin=0.0, vmax=1.0,
        )
        generate_heatmap(
            atlas.trend_similarity_matrix,
            "Trend Similarity Matrix",
            atlas_dir / "trend_similarity_heatmap.png",
            cmap="RdYlGn", vmin=-1.0, vmax=1.0,
        )
        archetype_assignments = classify_archetypes(
            atlas.slice_profiles, n_clusters=n_clusters, seed=seed
        )
        generate_archetype_chart(archetype_assignments, atlas_dir / "archetype_distribution.png")
    
    # 2. Generate Telemetry Pack
    telemetry_dir = out_dir / "telemetry"
    telemetry_meta = generate_telemetry_pack(
        baseline_jsonl,
        rfl_jsonl,
        str(telemetry_dir),
        window=window,
    )
    
    # 3. Build combined metadata
    combined_meta = {
        "phase_label": "PHASE II — NOT USED IN PHASE I",
        "disclaimer": "Combined analysis is purely descriptive. No uplift claims.",
        "atlas": {
            "path": str(atlas_path),
            "hash": atlas.manifest_hash,
            "slice_count": len(atlas.slice_profiles),
            "archetypes": atlas.archetypes,
        },
        "telemetry": {
            "path": str(telemetry_dir),
            "manifest_hash": telemetry_meta['manifest_hash'],
            "plot_count": telemetry_meta['plot_count'],
        },
        "parameters": {
            "n_clusters": n_clusters,
            "seed": seed,
            "window": window,
        },
    }
    
    # Save combined metadata
    combined_path = out_dir / "combined_analysis.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined_meta, f, indent=2, sort_keys=True)
    
    return combined_meta


# ===========================================================================
# CLI
# ===========================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PHASE II Institutional Behavior Atlas Generator",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
ABSOLUTE SAFEGUARDS:
- No uplift inference.
- No slice ranking.
- No governance-level claims.
- Output is purely descriptive: profiles, distances, clusters.

Examples:
    # Generate full atlas
    uv run python experiments/u2_behavior_atlas.py --input-dir results --out-dir artifacts/atlas
    
    # Generate specific outputs
    uv run python experiments/u2_behavior_atlas.py --input-dir results --atlas --matrix --fingerprints
    
    # Generate slice dossier
    uv run python experiments/u2_behavior_atlas.py --dossier --slice slice_uplift_goal \\
        --atlas-file artifacts/atlas/behavior_atlas.json \\
        --fingerprints-file artifacts/atlas/fingerprints.json \\
        --out artifacts/atlas/dossiers/slice_uplift_goal.json
    
    # Health check
    uv run python experiments/u2_behavior_atlas.py --health-check \\
        --atlas-file artifacts/atlas/behavior_atlas.json \\
        --fingerprints-file artifacts/atlas/fingerprints.json
        """
    )
    
    # Atlas generation mode
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing JSONL result files (for atlas generation)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/atlas",
        help="Output directory for atlas files (default: artifacts/atlas)",
    )
    parser.add_argument(
        "--slices",
        type=str,
        default=None,
        help="Comma-separated list of slice names (auto-discovered if not specified)",
    )
    parser.add_argument(
        "--atlas",
        action="store_true",
        help="Generate full atlas JSON",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Generate matrix heatmaps",
    )
    parser.add_argument(
        "--fingerprints",
        action="store_true",
        help="Generate fingerprint summary JSON",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of archetype clusters (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for clustering (default: 42)",
    )
    
    # Dossier mode
    parser.add_argument(
        "--dossier",
        action="store_true",
        help="Generate slice behavior dossier (requires --slice, --atlas-file, --fingerprints-file)",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Slice name for dossier generation",
    )
    parser.add_argument(
        "--atlas-file",
        type=str,
        default=None,
        help="Path to existing behavior_atlas.json (for dossier/health-check)",
    )
    parser.add_argument(
        "--fingerprints-file",
        type=str,
        default=None,
        help="Path to existing fingerprints.json (for dossier/health-check)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for dossier JSON",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=3,
        help="Number of closest behavioral neighbors to include in dossier (default: 3)",
    )
    
    # Health check mode
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run atlas health check (requires --atlas-file)",
    )
    
    args = parser.parse_args()
    
    # === DOSSIER MODE ===
    if args.dossier:
        if not args.slice:
            print("ERROR: --dossier requires --slice", file=sys.stderr)
            sys.exit(1)
        if not args.atlas_file:
            print("ERROR: --dossier requires --atlas-file", file=sys.stderr)
            sys.exit(1)
        if not args.fingerprints_file:
            print("ERROR: --dossier requires --fingerprints-file", file=sys.stderr)
            sys.exit(1)
        if not args.out:
            print("ERROR: --dossier requires --out", file=sys.stderr)
            sys.exit(1)
        
        print("PHASE II — Slice Behavior Dossier Generator")
        print("=" * 60)
        print(f"Slice: {args.slice}")
        print(f"Atlas: {args.atlas_file}")
        print(f"Fingerprints: {args.fingerprints_file}")
        print(f"Output: {args.out}")
        print()
        
        try:
            dossier = generate_slice_dossier(
                slice_name=args.slice,
                atlas_path=args.atlas_file,
                fingerprints_path=args.fingerprints_file,
                out_path=args.out,
                k_neighbors=args.k_neighbors,
            )
            print(f"✓ Dossier generated for '{args.slice}'")
            print(f"  Archetype: {dossier['assigned_archetype']}")
            print(f"  Nearest neighbors: {[n['neighbor_slice'] for n in dossier['nearest_neighbors']]}")
            print(f"  Metrics summary: {dossier['metrics_summary']}")
            print(f"  Output: {args.out}")
            sys.exit(0)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"ERROR: File not found: {e}", file=sys.stderr)
            sys.exit(1)
    
    # === HEALTH CHECK MODE ===
    if args.health_check:
        if not args.atlas_file:
            print("ERROR: --health-check requires --atlas-file", file=sys.stderr)
            sys.exit(1)
        
        print("PHASE II — Atlas Health Check")
        print("=" * 60)
        print(f"Atlas: {args.atlas_file}")
        print()
        
        try:
            atlas = load_atlas_from_file(args.atlas_file)
            result = evaluate_atlas_health(atlas)
            
            print(f"Status: {result['status']}")
            print(f"Checks performed: {len(result['checks_performed'])}")
            
            if result['issues']:
                print(f"\nIssues found ({len(result['issues'])}):")
                for issue in result['issues']:
                    print(f"  - {issue}")
            else:
                print("\nNo issues found.")
            
            # Exit code based on status
            if result['status'] == "BLOCK":
                sys.exit(1)
            else:
                sys.exit(0)
                
        except FileNotFoundError as e:
            print(f"ERROR: File not found: {e}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in atlas file: {e}", file=sys.stderr)
            sys.exit(1)
    
    # === ATLAS GENERATION MODE ===
    if not args.input_dir:
        print("ERROR: --input-dir is required for atlas generation", file=sys.stderr)
        print("       Use --dossier or --health-check for other modes", file=sys.stderr)
        sys.exit(1)
    
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Parse slice list
    slice_names = None
    if args.slices:
        slice_names = [s.strip() for s in args.slices.split(",")]
    
    # Default: generate everything if no specific flag
    generate_all = not (args.atlas or args.matrix or args.fingerprints)
    
    print("PHASE II — Institutional Behavior Atlas Generator")
    print("=" * 60)
    print("DISCLAIMER: All outputs are purely descriptive.")
    print("Archetypes do NOT imply quality, performance, or preference.")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {out_dir}")
    print()
    
    # Build atlas
    print("Building behavioral atlas...")
    atlas, baseline_records = build_behavior_atlas(
        input_dir,
        slice_names=slice_names,
        n_clusters=args.n_clusters,
        clustering_seed=args.seed,
    )
    
    if not atlas.slice_profiles:
        print("ERROR: No slices were successfully processed", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processed {len(atlas.slice_profiles)} slices")
    print(f"Atlas manifest hash: {atlas.manifest_hash[:16]}...")
    print()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate outputs
    if generate_all or args.atlas:
        atlas_path = out_dir / "behavior_atlas.json"
        with open(atlas_path, 'w', encoding='utf-8') as f:
            json.dump(atlas_to_dict(atlas), f, indent=2, sort_keys=True)
        print(f"✓ Atlas written: {atlas_path}")
    
    if generate_all or args.fingerprints:
        fingerprints_path = out_dir / "fingerprints.json"
        fingerprints_data = {
            "phase_label": "PHASE II — NOT USED IN PHASE I",
            "fingerprints": {
                name: {
                    "baseline_hash": profile.baseline_fingerprint.fingerprint_hash,
                    "rfl_hash": profile.rfl_fingerprint.fingerprint_hash,
                }
                for name, profile in sorted(atlas.slice_profiles.items())
            },
        }
        with open(fingerprints_path, 'w', encoding='utf-8') as f:
            json.dump(fingerprints_data, f, indent=2, sort_keys=True)
        print(f"✓ Fingerprints written: {fingerprints_path}")
    
    if generate_all or args.matrix:
        # JS Divergence Heatmap
        js_path = out_dir / "js_divergence_heatmap.png"
        generate_heatmap(
            atlas.js_divergence_matrix,
            "JS-Divergence Matrix (Behavioral Distance)",
            js_path,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
        )
        print(f"✓ JS-divergence heatmap: {js_path}")
        
        # Trend Similarity Heatmap
        trend_path = out_dir / "trend_similarity_heatmap.png"
        generate_heatmap(
            atlas.trend_similarity_matrix,
            "Trend Similarity Matrix (Abstention Correlation)",
            trend_path,
            cmap="RdYlGn",
            vmin=-1.0,
            vmax=1.0,
        )
        print(f"✓ Trend similarity heatmap: {trend_path}")
        
        # Abstention Similarity Heatmap
        abstention_path = out_dir / "abstention_similarity_heatmap.png"
        generate_heatmap(
            atlas.abstention_similarity_matrix,
            "Abstention Profile Similarity (Cosine)",
            abstention_path,
            cmap="PuBu",
            vmin=0.0,
            vmax=1.0,
        )
        print(f"✓ Abstention similarity heatmap: {abstention_path}")
        
        # Archetype Distribution Chart
        archetype_assignments = classify_archetypes(
            atlas.slice_profiles,
            n_clusters=args.n_clusters,
            seed=args.seed,
        )
        archetype_path = out_dir / "archetype_distribution.png"
        generate_archetype_chart(archetype_assignments, archetype_path)
        print(f"✓ Archetype distribution chart: {archetype_path}")
    
    # Print archetype summary
    print()
    print("Archetype Assignments (Purely Descriptive):")
    print("-" * 40)
    for name in sorted(atlas.archetypes.keys()):
        print(f"  {name}: {atlas.archetypes[name]}")
    
    print()
    print("✓ Atlas generation complete")
    sys.exit(0)


if __name__ == "__main__":
    main()

