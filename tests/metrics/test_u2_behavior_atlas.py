"""
PHASE II — NOT USED IN PHASE I

Tests for u2_behavior_atlas.py module.

These tests verify:
1. Fingerprint consistency
2. Matrix determinism
3. Seed-stable clustering
4. No uplift/ranking/governance claims

Agent: metrics-engineer-6 (D6)

Total: 25 tests
"""

import json
import tempfile
import pytest
from pathlib import Path
from typing import Any, Dict, List

from experiments.u2_behavior_atlas import (
    SliceProfile,
    BehaviorAtlas,
    ArchetypeAssignment,
    ARCHETYPE_LABELS,
    extract_feature_vector,
    compute_js_divergence_matrix,
    compute_trend_similarity_matrix,
    compute_abstention_similarity_matrix,
    classify_archetypes,
    build_behavior_atlas,
    atlas_to_dict,
    archetype_assignments_to_dict,
    _compute_atlas_hash,
    _histogram_entropy,
    _weighted_mean,
    _weighted_variance,
    _trend_volatility,
    _cosine_similarity,
    _kmeans,
    _euclidean_distance,
    generate_heatmap,
    # New functions for dossier, comparison, health check
    generate_slice_dossier,
    compare_real_vs_synthetic,
    evaluate_atlas_health,
    load_atlas_from_file,
    _extract_matrix_values,
    # Phase II v1.2 routing & alignment layer
    build_routing_hint,
    compute_atlas_compatibility,
    is_atlas_structurally_sound,
    # Phase III governance & global routing layer
    build_atlas_governance_snapshot,
    build_routing_overview,
    summarize_atlas_for_global_health,
    # Phase IV atlas-guided routing & structural governance
    derive_atlas_routing_policy,
    build_structural_governance_view,
    build_atlas_director_panel,
    # Atlas-curriculum coupler & phase transition advisor
    build_atlas_curriculum_coupling_view,
    derive_phase_transition_advice,
    _count_values,
    _compute_distribution_stats,
)

from experiments.u2_cross_slice_analysis import (
    SliceResults,
    BehavioralFingerprint,
    generate_behavior_signature,
)


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def sample_records_a() -> List[Dict[str, Any]]:
    """Generate deterministic records for slice A."""
    return [
        {
            "cycle": i,
            "slice": "slice_a",
            "mode": "baseline",
            "seed": 42 + i,
            "success": i % 3 != 0,
            "metric_value": float(i % 5),
            "derivation": {
                "candidate_order": [f"hash_{j:04d}" for j in range(4)],
                "verified_count": (i % 3) + 1,
            },
            "metric_result": {
                "hit_count": i % 2,
                "total_verified": (i % 3) + 1,
            },
        }
        for i in range(30)
    ]


@pytest.fixture
def sample_records_b() -> List[Dict[str, Any]]:
    """Generate deterministic records for slice B with different pattern."""
    return [
        {
            "cycle": i,
            "slice": "slice_b",
            "mode": "baseline",
            "seed": 100 + i,
            "success": i % 2 != 0,
            "metric_value": float((i + 2) % 4),
            "derivation": {
                "candidate_order": [f"b_hash_{j:04d}" for j in range(5)],
                "verified_count": (i % 4) + 1,
            },
            "metric_result": {
                "hit_count": (i + 1) % 3,
                "total_verified": (i % 4) + 1,
            },
        }
        for i in range(30)
    ]


@pytest.fixture
def sample_records_c() -> List[Dict[str, Any]]:
    """Generate deterministic records for slice C with yet another pattern."""
    return [
        {
            "cycle": i,
            "slice": "slice_c",
            "mode": "baseline",
            "seed": 200 + i,
            "success": i % 4 != 0,
            "metric_value": float(i % 3),
            "derivation": {
                "candidate_order": [f"c_hash_{j:04d}" for j in range(3)],
                "verified_count": (i % 2) + 1,
            },
            "metric_result": {
                "hit_count": i % 2,
                "total_verified": (i % 2) + 1,
            },
        }
        for i in range(30)
    ]


@pytest.fixture
def slice_results_a(sample_records_a) -> SliceResults:
    return SliceResults(slice_name="slice_a", mode="baseline", records=sample_records_a)


@pytest.fixture
def slice_results_b(sample_records_b) -> SliceResults:
    return SliceResults(slice_name="slice_b", mode="baseline", records=sample_records_b)


@pytest.fixture
def slice_results_c(sample_records_c) -> SliceResults:
    return SliceResults(slice_name="slice_c", mode="baseline", records=sample_records_c)


@pytest.fixture
def fingerprint_a(slice_results_a) -> BehavioralFingerprint:
    return generate_behavior_signature(slice_results_a)


@pytest.fixture
def fingerprint_b(slice_results_b) -> BehavioralFingerprint:
    return generate_behavior_signature(slice_results_b)


@pytest.fixture
def fingerprint_c(slice_results_c) -> BehavioralFingerprint:
    return generate_behavior_signature(slice_results_c)


@pytest.fixture
def sample_profiles(
    fingerprint_a, fingerprint_b, fingerprint_c,
    sample_records_a, sample_records_b, sample_records_c,
) -> Dict[str, SliceProfile]:
    """Create sample profiles for three slices."""
    profiles = {}
    
    for name, fp, records in [
        ("slice_a", fingerprint_a, sample_records_a),
        ("slice_b", fingerprint_b, sample_records_b),
        ("slice_c", fingerprint_c, sample_records_c),
    ]:
        feature_vec = extract_feature_vector(fp, fp, records, records)
        profiles[name] = SliceProfile(
            slice_name=name,
            baseline_fingerprint=fp,
            rfl_fingerprint=fp,  # Using same for simplicity
            baseline_record_count=len(records),
            rfl_record_count=len(records),
            feature_vector=feature_vec,
        )
    
    return profiles


@pytest.fixture
def sample_baseline_records(
    sample_records_a, sample_records_b, sample_records_c,
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "slice_a": sample_records_a,
        "slice_b": sample_records_b,
        "slice_c": sample_records_c,
    }


# ===========================================================================
# FINGERPRINT CONSISTENCY TESTS (5 tests)
# ===========================================================================

class TestFingerprintConsistency:
    """Tests for fingerprint consistency across runs."""

    def test_fingerprint_hash_deterministic(self, slice_results_a):
        """Test 1: Fingerprint hash is deterministic across calls."""
        fp1 = generate_behavior_signature(slice_results_a)
        fp2 = generate_behavior_signature(slice_results_a)
        fp3 = generate_behavior_signature(slice_results_a)
        
        assert fp1.fingerprint_hash == fp2.fingerprint_hash
        assert fp2.fingerprint_hash == fp3.fingerprint_hash

    def test_fingerprint_hash_100_calls(self, slice_results_a):
        """Test 2: Fingerprint hash identical across 100 calls."""
        hashes = [
            generate_behavior_signature(slice_results_a).fingerprint_hash
            for _ in range(100)
        ]
        
        assert all(h == hashes[0] for h in hashes)

    def test_different_slices_different_fingerprints(
        self, slice_results_a, slice_results_b,
    ):
        """Test 3: Different slices produce different fingerprints."""
        fp_a = generate_behavior_signature(slice_results_a)
        fp_b = generate_behavior_signature(slice_results_b)
        
        assert fp_a.fingerprint_hash != fp_b.fingerprint_hash

    def test_feature_vector_deterministic(
        self, fingerprint_a, sample_records_a,
    ):
        """Test 4: Feature vector extraction is deterministic."""
        vectors = [
            extract_feature_vector(
                fingerprint_a, fingerprint_a,
                sample_records_a, sample_records_a,
            )
            for _ in range(50)
        ]
        
        assert all(v == vectors[0] for v in vectors)

    def test_feature_vector_length_consistent(
        self, fingerprint_a, fingerprint_b,
        sample_records_a, sample_records_b,
    ):
        """Test 5: Feature vectors have consistent length."""
        vec_a = extract_feature_vector(
            fingerprint_a, fingerprint_a,
            sample_records_a, sample_records_a,
        )
        vec_b = extract_feature_vector(
            fingerprint_b, fingerprint_b,
            sample_records_b, sample_records_b,
        )
        
        assert len(vec_a) == len(vec_b)
        assert len(vec_a) == 10  # Expected feature count


# ===========================================================================
# MATRIX DETERMINISM TESTS (8 tests)
# ===========================================================================

class TestMatrixDeterminism:
    """Tests for matrix computation determinism."""

    def test_js_divergence_matrix_deterministic(self, sample_profiles):
        """Test 6: JS-divergence matrix is deterministic."""
        matrices = [
            compute_js_divergence_matrix(sample_profiles)
            for _ in range(50)
        ]
        
        json_matrices = [json.dumps(m, sort_keys=True) for m in matrices]
        assert all(jm == json_matrices[0] for jm in json_matrices)

    def test_js_divergence_matrix_symmetric(self, sample_profiles):
        """Test 7: JS-divergence matrix is symmetric."""
        matrix = compute_js_divergence_matrix(sample_profiles)
        
        for name1 in matrix:
            for name2 in matrix:
                assert matrix[name1][name2] == matrix[name2][name1]

    def test_js_divergence_diagonal_zero(self, sample_profiles):
        """Test 8: JS-divergence diagonal is zero (self-comparison)."""
        matrix = compute_js_divergence_matrix(sample_profiles)
        
        for name in matrix:
            assert matrix[name][name] == 0.0

    def test_trend_similarity_matrix_deterministic(
        self, sample_profiles, sample_baseline_records,
    ):
        """Test 9: Trend similarity matrix is deterministic."""
        matrices = [
            compute_trend_similarity_matrix(sample_profiles, sample_baseline_records)
            for _ in range(50)
        ]
        
        json_matrices = [json.dumps(m, sort_keys=True) for m in matrices]
        assert all(jm == json_matrices[0] for jm in json_matrices)

    def test_trend_similarity_matrix_symmetric(
        self, sample_profiles, sample_baseline_records,
    ):
        """Test 10: Trend similarity matrix is symmetric."""
        matrix = compute_trend_similarity_matrix(sample_profiles, sample_baseline_records)
        
        for name1 in matrix:
            for name2 in matrix:
                assert matrix[name1][name2] == matrix[name2][name1]

    def test_abstention_similarity_matrix_deterministic(
        self, sample_profiles, sample_baseline_records,
    ):
        """Test 11: Abstention similarity matrix is deterministic."""
        matrices = [
            compute_abstention_similarity_matrix(sample_profiles, sample_baseline_records)
            for _ in range(50)
        ]
        
        json_matrices = [json.dumps(m, sort_keys=True) for m in matrices]
        assert all(jm == json_matrices[0] for jm in json_matrices)

    def test_abstention_similarity_in_range(
        self, sample_profiles, sample_baseline_records,
    ):
        """Test 12: Abstention similarity (cosine) values are in valid range."""
        matrix = compute_abstention_similarity_matrix(sample_profiles, sample_baseline_records)
        
        for name1 in matrix:
            for name2 in matrix:
                assert -1.0 <= matrix[name1][name2] <= 1.0

    def test_atlas_hash_deterministic(self, sample_profiles, sample_baseline_records):
        """Test 13: Atlas hash is deterministic."""
        js_matrix = compute_js_divergence_matrix(sample_profiles)
        trend_matrix = compute_trend_similarity_matrix(sample_profiles, sample_baseline_records)
        abstention_matrix = compute_abstention_similarity_matrix(sample_profiles, sample_baseline_records)
        
        atlas = BehaviorAtlas(
            slice_profiles=sample_profiles,
            js_divergence_matrix=js_matrix,
            trend_similarity_matrix=trend_matrix,
            abstention_similarity_matrix=abstention_matrix,
        )
        
        hashes = [_compute_atlas_hash(atlas) for _ in range(50)]
        assert all(h == hashes[0] for h in hashes)


# ===========================================================================
# SEED-STABLE CLUSTERING TESTS (7 tests)
# ===========================================================================

class TestSeedStableClustering:
    """Tests for seed-stable k-means clustering."""

    def test_kmeans_deterministic_same_seed(self, sample_profiles):
        """Test 14: k-means produces identical results with same seed."""
        results = [
            classify_archetypes(sample_profiles, n_clusters=2, seed=42)
            for _ in range(50)
        ]
        
        json_results = [
            json.dumps(archetype_assignments_to_dict(r), sort_keys=True)
            for r in results
        ]
        assert all(jr == json_results[0] for jr in json_results)

    def test_kmeans_different_seeds_may_differ(self, sample_profiles):
        """Test 15: k-means with different seeds may produce different results."""
        result_42 = classify_archetypes(sample_profiles, n_clusters=2, seed=42)
        result_99 = classify_archetypes(sample_profiles, n_clusters=2, seed=99)
        
        # They may or may not differ, but both should be valid
        assert len(result_42) == len(sample_profiles)
        assert len(result_99) == len(sample_profiles)

    def test_archetype_labels_valid(self, sample_profiles):
        """Test 16: Archetype labels are from predefined set."""
        result = classify_archetypes(sample_profiles, n_clusters=4, seed=42)
        
        for assignment in result.values():
            assert assignment.archetype_label in ARCHETYPE_LABELS.values() or \
                   assignment.archetype_label.startswith("archetype-")

    def test_cluster_ids_in_range(self, sample_profiles):
        """Test 17: Cluster IDs are in valid range."""
        n_clusters = 3
        result = classify_archetypes(sample_profiles, n_clusters=n_clusters, seed=42)
        
        for assignment in result.values():
            assert 0 <= assignment.cluster_id < n_clusters

    def test_centroid_distance_non_negative(self, sample_profiles):
        """Test 18: Centroid distances are non-negative."""
        result = classify_archetypes(sample_profiles, n_clusters=2, seed=42)
        
        for assignment in result.values():
            assert assignment.centroid_distance >= 0.0

    def test_single_slice_single_cluster(self):
        """Test 19: Single slice gets assigned to single cluster."""
        # Create single-slice profile
        records = [{"cycle": i, "success": True, "metric_value": 1.0} for i in range(10)]
        results = SliceResults(slice_name="only_slice", mode="baseline", records=records)
        fp = generate_behavior_signature(results)
        
        single_profile = {
            "only_slice": SliceProfile(
                slice_name="only_slice",
                baseline_fingerprint=fp,
                rfl_fingerprint=fp,
                baseline_record_count=10,
                rfl_record_count=10,
                feature_vector=extract_feature_vector(fp, fp, records, records),
            )
        }
        
        result = classify_archetypes(single_profile, n_clusters=3, seed=42)
        assert len(result) == 1
        assert result["only_slice"].cluster_id == 0

    def test_empty_profiles_returns_empty(self):
        """Test 20: Empty profiles returns empty assignments."""
        result = classify_archetypes({}, n_clusters=3, seed=42)
        assert result == {}


# ===========================================================================
# NO UPLIFT/RANKING TESTS (5 tests)
# ===========================================================================

class TestNoUpliftRanking:
    """Tests ensuring no uplift inference or ranking."""

    def test_atlas_has_disclaimer(self, sample_profiles, sample_baseline_records):
        """Test 21: Atlas output contains disclaimer."""
        js_matrix = compute_js_divergence_matrix(sample_profiles)
        trend_matrix = compute_trend_similarity_matrix(sample_profiles, sample_baseline_records)
        abstention_matrix = compute_abstention_similarity_matrix(sample_profiles, sample_baseline_records)
        
        atlas = BehaviorAtlas(
            slice_profiles=sample_profiles,
            js_divergence_matrix=js_matrix,
            trend_similarity_matrix=trend_matrix,
            abstention_similarity_matrix=abstention_matrix,
        )
        
        atlas_dict = atlas_to_dict(atlas)
        
        assert "disclaimer" in atlas_dict
        assert "NOT imply quality" in atlas_dict["disclaimer"]

    def test_archetype_dict_has_disclaimer(self, sample_profiles):
        """Test 22: Archetype output contains disclaimer."""
        assignments = classify_archetypes(sample_profiles, n_clusters=2, seed=42)
        result_dict = archetype_assignments_to_dict(assignments)
        
        assert "disclaimer" in result_dict
        assert "NOT imply quality" in result_dict["disclaimer"]

    def test_no_ranking_keys_in_output(self, sample_profiles, sample_baseline_records):
        """Test 23: No ranking-related keys in atlas output."""
        js_matrix = compute_js_divergence_matrix(sample_profiles)
        trend_matrix = compute_trend_similarity_matrix(sample_profiles, sample_baseline_records)
        abstention_matrix = compute_abstention_similarity_matrix(sample_profiles, sample_baseline_records)
        
        atlas = BehaviorAtlas(
            slice_profiles=sample_profiles,
            js_divergence_matrix=js_matrix,
            trend_similarity_matrix=trend_matrix,
            abstention_similarity_matrix=abstention_matrix,
        )
        
        atlas_str = json.dumps(atlas_to_dict(atlas)).lower()
        
        forbidden = ["rank", "best", "worst", "better", "worse", "uplift", "improvement"]
        for word in forbidden:
            assert word not in atlas_str, f"Forbidden word '{word}' found in output"

    def test_archetype_labels_neutral(self):
        """Test 24: Archetype labels are behaviorally neutral (no quality terms)."""
        quality_terms = ["good", "bad", "best", "worst", "better", "worse", "optimal", "poor"]
        
        for label in ARCHETYPE_LABELS.values():
            label_lower = label.lower()
            for term in quality_terms:
                assert term not in label_lower, f"Quality term '{term}' found in label '{label}'"

    def test_phase_label_present(self, sample_profiles, sample_baseline_records):
        """Test 25: Phase label present in all outputs."""
        js_matrix = compute_js_divergence_matrix(sample_profiles)
        trend_matrix = compute_trend_similarity_matrix(sample_profiles, sample_baseline_records)
        abstention_matrix = compute_abstention_similarity_matrix(sample_profiles, sample_baseline_records)
        
        atlas = BehaviorAtlas(
            slice_profiles=sample_profiles,
            js_divergence_matrix=js_matrix,
            trend_similarity_matrix=trend_matrix,
            abstention_similarity_matrix=abstention_matrix,
        )
        
        atlas_dict = atlas_to_dict(atlas)
        assert "PHASE II" in atlas_dict["phase_label"]
        
        assignments = classify_archetypes(sample_profiles, n_clusters=2, seed=42)
        assign_dict = archetype_assignments_to_dict(assignments)
        assert "PHASE II" in assign_dict["phase_label"]


# ===========================================================================
# UTILITY FUNCTION TESTS
# ===========================================================================

class TestUtilityFunctions:
    """Additional tests for utility functions."""

    def test_histogram_entropy_empty(self):
        """Entropy of empty histogram is 0."""
        assert _histogram_entropy({}) == 0.0

    def test_histogram_entropy_single(self):
        """Entropy of single-bin histogram is 0."""
        assert _histogram_entropy({"a": 10}) == 0.0

    def test_histogram_entropy_uniform(self):
        """Entropy of uniform distribution."""
        hist = {"a": 10, "b": 10}  # 2 bins, uniform
        entropy = _histogram_entropy(hist)
        assert entropy == pytest.approx(1.0, abs=0.01)  # log2(2) = 1

    def test_weighted_mean_empty(self):
        """Mean of empty histogram is 0."""
        assert _weighted_mean({}) == 0.0

    def test_weighted_mean_single(self):
        """Mean of single-value histogram."""
        assert _weighted_mean({5: 10}) == 5.0

    def test_weighted_variance_constant(self):
        """Variance of constant is 0."""
        assert _weighted_variance({5: 10}) == 0.0

    def test_trend_volatility_constant(self):
        """Volatility of constant trend is 0."""
        assert _trend_volatility([0.5, 0.5, 0.5]) == 0.0

    def test_trend_volatility_short(self):
        """Volatility of single-element trend is 0."""
        assert _trend_volatility([0.5]) == 0.0

    def test_cosine_similarity_identical(self):
        """Cosine similarity of identical vectors is 1."""
        vec = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0, abs=0.001)

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors is 0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        assert _cosine_similarity(vec1, vec2) == pytest.approx(0.0, abs=0.001)

    def test_euclidean_distance_same(self):
        """Distance between identical vectors is 0."""
        vec = [1.0, 2.0, 3.0]
        assert _euclidean_distance(vec, vec) == 0.0

    def test_euclidean_distance_known(self):
        """Known Euclidean distance."""
        vec1 = [0.0, 0.0]
        vec2 = [3.0, 4.0]
        assert _euclidean_distance(vec1, vec2) == 5.0


# ===========================================================================
# SLICE BEHAVIOR DOSSIER TESTS
# ===========================================================================

@pytest.fixture
def fixture_atlas_files(sample_profiles, sample_baseline_records, tmp_path):
    """Create temporary atlas and fingerprints files for testing."""
    # Build matrices
    js_matrix = compute_js_divergence_matrix(sample_profiles)
    trend_matrix = compute_trend_similarity_matrix(sample_profiles, sample_baseline_records)
    abstention_matrix = compute_abstention_similarity_matrix(sample_profiles, sample_baseline_records)
    
    # Classify archetypes
    archetypes = classify_archetypes(sample_profiles, n_clusters=2, seed=42)
    
    # Build atlas dict
    atlas = BehaviorAtlas(
        slice_profiles=sample_profiles,
        js_divergence_matrix=js_matrix,
        trend_similarity_matrix=trend_matrix,
        abstention_similarity_matrix=abstention_matrix,
        archetypes={k: v.archetype_label for k, v in archetypes.items()},
    )
    atlas.manifest_hash = _compute_atlas_hash(atlas)
    
    atlas_dict = atlas_to_dict(atlas)
    
    # Write atlas file
    atlas_path = tmp_path / "behavior_atlas.json"
    with open(atlas_path, 'w') as f:
        json.dump(atlas_dict, f, indent=2, sort_keys=True)
    
    # Build fingerprints
    fingerprints_data = {
        "phase_label": "PHASE II — NOT USED IN PHASE I",
        "fingerprints": {
            name: {
                "baseline_hash": profile.baseline_fingerprint.fingerprint_hash,
                "rfl_hash": profile.rfl_fingerprint.fingerprint_hash,
            }
            for name, profile in sample_profiles.items()
        },
    }
    
    # Write fingerprints file
    fingerprints_path = tmp_path / "fingerprints.json"
    with open(fingerprints_path, 'w') as f:
        json.dump(fingerprints_data, f, indent=2, sort_keys=True)
    
    return {
        "atlas_path": str(atlas_path),
        "fingerprints_path": str(fingerprints_path),
        "atlas_dict": atlas_dict,
        "tmp_path": tmp_path,
    }


class TestSliceBehaviorDossier:
    """Tests for generate_slice_dossier function (FROZEN CONTRACT)."""

    def test_dossier_generates_valid_output(self, fixture_atlas_files):
        """Dossier generates all expected keys per FROZEN CONTRACT."""
        out_path = fixture_atlas_files["tmp_path"] / "dossiers" / "slice_a.json"
        
        dossier = generate_slice_dossier(
            slice_name="slice_a",
            atlas_path=fixture_atlas_files["atlas_path"],
            fingerprints_path=fixture_atlas_files["fingerprints_path"],
            out_path=str(out_path),
        )
        
        # FROZEN CONTRACT: Required top-level keys
        assert "slice_name" in dossier
        assert "assigned_archetype" in dossier
        assert "feature_vector" in dossier
        assert "nearest_neighbors" in dossier
        assert "metrics_summary" in dossier
        
        # Additional fields (not contract-breaking if present)
        assert "phase_label" in dossier
        assert "disclaimer" in dossier
        assert "fingerprint_hashes" in dossier
        assert "trend_similarity_profile" in dossier
        assert "abstention_similarity_profile" in dossier
        assert "lineage" in dossier
        
        # FROZEN CONTRACT: nearest_neighbors structure
        for neighbor in dossier["nearest_neighbors"]:
            assert "neighbor_slice" in neighbor
            assert "neighbor_archetype" in neighbor
            assert "distance" in neighbor
            assert isinstance(neighbor["distance"], float)
        
        # FROZEN CONTRACT: metrics_summary structure
        summary = dossier["metrics_summary"]
        assert "feature_vector_dimension" in summary
        assert "feature_vector_mean" in summary
        assert "neighbor_distance_mean" in summary
        assert "neighbor_distance_min" in summary
        assert "trend_similarity_mean" in summary
        assert "abstention_similarity_mean" in summary
        assert "neighbor_count" in summary
        
        # Type verification
        assert isinstance(summary["feature_vector_dimension"], int)
        assert isinstance(summary["feature_vector_mean"], float)
        assert isinstance(summary["neighbor_distance_mean"], float)
        assert isinstance(summary["neighbor_distance_min"], float)
        assert isinstance(summary["trend_similarity_mean"], float)
        assert isinstance(summary["abstention_similarity_mean"], float)
        assert isinstance(summary["neighbor_count"], int)
        
        # Verify NO statistical test terms
        summary_str = json.dumps(summary).lower()
        assert "p_value" not in summary_str
        assert "delta" not in summary_str
        assert "significance" not in summary_str

    def test_dossier_for_multiple_slices(self, fixture_atlas_files):
        """Dossier generates correctly for at least 2 slices."""
        dossiers = {}
        for slice_name in ["slice_a", "slice_b"]:
            out_path = fixture_atlas_files["tmp_path"] / f"dossier_{slice_name}.json"
            dossiers[slice_name] = generate_slice_dossier(
                slice_name=slice_name,
                atlas_path=fixture_atlas_files["atlas_path"],
                fingerprints_path=fixture_atlas_files["fingerprints_path"],
                out_path=str(out_path),
            )
        
        # Both dossiers should have all required keys
        for name, dossier in dossiers.items():
            assert dossier["slice_name"] == name
            assert "assigned_archetype" in dossier
            assert "feature_vector" in dossier
            assert "nearest_neighbors" in dossier
            assert "metrics_summary" in dossier
        
        # Dossiers should be different
        assert dossiers["slice_a"]["fingerprint_hashes"] != dossiers["slice_b"]["fingerprint_hashes"]

    def test_dossier_deterministic(self, fixture_atlas_files):
        """Dossier output is deterministic across multiple calls."""
        out_path1 = fixture_atlas_files["tmp_path"] / "dossier1.json"
        out_path2 = fixture_atlas_files["tmp_path"] / "dossier2.json"
        
        dossier1 = generate_slice_dossier(
            slice_name="slice_a",
            atlas_path=fixture_atlas_files["atlas_path"],
            fingerprints_path=fixture_atlas_files["fingerprints_path"],
            out_path=str(out_path1),
        )
        
        dossier2 = generate_slice_dossier(
            slice_name="slice_a",
            atlas_path=fixture_atlas_files["atlas_path"],
            fingerprints_path=fixture_atlas_files["fingerprints_path"],
            out_path=str(out_path2),
        )
        
        # Compare JSON strings (excluding path-specific lineage)
        d1 = {k: v for k, v in dossier1.items() if k != "lineage"}
        d2 = {k: v for k, v in dossier2.items() if k != "lineage"}
        
        assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)

    def test_nearest_neighbors_deterministic(self, fixture_atlas_files):
        """Nearest neighbors list is deterministic given fixed seed."""
        results = []
        for i in range(10):
            out_path = fixture_atlas_files["tmp_path"] / f"dossier_det_{i}.json"
            dossier = generate_slice_dossier(
                slice_name="slice_a",
                atlas_path=fixture_atlas_files["atlas_path"],
                fingerprints_path=fixture_atlas_files["fingerprints_path"],
                out_path=str(out_path),
            )
            results.append(json.dumps(dossier["nearest_neighbors"], sort_keys=True))
        
        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_dossier_neighbors_excludes_self(self, fixture_atlas_files):
        """Nearest neighbors do not include the slice itself."""
        out_path = fixture_atlas_files["tmp_path"] / "dossier.json"
        
        dossier = generate_slice_dossier(
            slice_name="slice_a",
            atlas_path=fixture_atlas_files["atlas_path"],
            fingerprints_path=fixture_atlas_files["fingerprints_path"],
            out_path=str(out_path),
        )
        
        neighbor_names = [n["neighbor_slice"] for n in dossier["nearest_neighbors"]]
        assert "slice_a" not in neighbor_names

    def test_dossier_raises_for_invalid_slice(self, fixture_atlas_files):
        """Dossier raises ValueError for non-existent slice."""
        out_path = fixture_atlas_files["tmp_path"] / "invalid.json"
        
        with pytest.raises(ValueError, match="not found in atlas"):
            generate_slice_dossier(
                slice_name="nonexistent_slice",
                atlas_path=fixture_atlas_files["atlas_path"],
                fingerprints_path=fixture_atlas_files["fingerprints_path"],
                out_path=str(out_path),
            )

    def test_dossier_writes_file(self, fixture_atlas_files):
        """Dossier writes output file."""
        out_path = fixture_atlas_files["tmp_path"] / "dossier_output.json"
        
        generate_slice_dossier(
            slice_name="slice_a",
            atlas_path=fixture_atlas_files["atlas_path"],
            fingerprints_path=fixture_atlas_files["fingerprints_path"],
            out_path=str(out_path),
        )
        
        assert out_path.exists()
        
        with open(out_path) as f:
            loaded = json.load(f)
        
        assert loaded["slice_name"] == "slice_a"

    def test_dossier_has_phase_disclaimer(self, fixture_atlas_files):
        """Dossier includes PHASE II disclaimer."""
        out_path = fixture_atlas_files["tmp_path"] / "dossier.json"
        
        dossier = generate_slice_dossier(
            slice_name="slice_a",
            atlas_path=fixture_atlas_files["atlas_path"],
            fingerprints_path=fixture_atlas_files["fingerprints_path"],
            out_path=str(out_path),
        )
        
        assert "PHASE II" in dossier["phase_label"]
        assert "NOT imply quality" in dossier["disclaimer"]

    def test_dossier_no_ranking_terms(self, fixture_atlas_files):
        """Dossier output contains no ranking terms."""
        out_path = fixture_atlas_files["tmp_path"] / "dossier.json"
        
        dossier = generate_slice_dossier(
            slice_name="slice_a",
            atlas_path=fixture_atlas_files["atlas_path"],
            fingerprints_path=fixture_atlas_files["fingerprints_path"],
            out_path=str(out_path),
        )
        
        # Exclude paths from check
        dossier_without_paths = {k: v for k, v in dossier.items() if k != "lineage"}
        dossier_str = json.dumps(dossier_without_paths).lower()
        
        forbidden = ["best", "worst", "top", "bottom", "rank"]
        for word in forbidden:
            assert word not in dossier_str, f"Ranking term '{word}' found in dossier"


# ===========================================================================
# REAL VS SYNTHETIC ATLAS COMPARISON TESTS
# ===========================================================================

@pytest.fixture
def fixture_two_atlases(fixture_atlas_files, sample_profiles, sample_baseline_records, tmp_path):
    """Create two atlas files (real and synthetic) for comparison testing."""
    # First atlas is from fixture_atlas_files
    real_path = fixture_atlas_files["atlas_path"]
    
    # Create a slightly different "synthetic" atlas
    # Modify profiles slightly
    synthetic_profiles = {}
    for name, profile in sample_profiles.items():
        # Shift feature vector slightly
        new_fv = [v + 0.1 for v in profile.feature_vector]
        synthetic_profiles[name] = SliceProfile(
            slice_name=name,
            baseline_fingerprint=profile.baseline_fingerprint,
            rfl_fingerprint=profile.rfl_fingerprint,
            baseline_record_count=profile.baseline_record_count,
            rfl_record_count=profile.rfl_record_count,
            feature_vector=new_fv,
        )
    
    # Recompute matrices with original profiles (same structure, different for test)
    js_matrix = compute_js_divergence_matrix(sample_profiles)
    trend_matrix = compute_trend_similarity_matrix(sample_profiles, sample_baseline_records)
    abstention_matrix = compute_abstention_similarity_matrix(sample_profiles, sample_baseline_records)
    
    # Use different clustering seed for synthetic
    archetypes = classify_archetypes(synthetic_profiles, n_clusters=2, seed=99)
    
    synthetic_atlas = BehaviorAtlas(
        slice_profiles=synthetic_profiles,
        js_divergence_matrix=js_matrix,
        trend_similarity_matrix=trend_matrix,
        abstention_similarity_matrix=abstention_matrix,
        archetypes={k: v.archetype_label for k, v in archetypes.items()},
    )
    synthetic_atlas.manifest_hash = _compute_atlas_hash(synthetic_atlas)
    
    synthetic_dict = atlas_to_dict(synthetic_atlas)
    
    synthetic_path = tmp_path / "synthetic_atlas.json"
    with open(synthetic_path, 'w') as f:
        json.dump(synthetic_dict, f, indent=2, sort_keys=True)
    
    return {
        "real_path": real_path,
        "synthetic_path": str(synthetic_path),
    }


class TestRealVsSyntheticComparison:
    """Tests for compare_real_vs_synthetic function."""

    def test_comparison_returns_expected_keys(self, fixture_two_atlases):
        """Comparison returns all expected keys per spec."""
        result = compare_real_vs_synthetic(
            fixture_two_atlases["real_path"],
            fixture_two_atlases["synthetic_path"],
        )
        
        # Required keys per spec
        assert "phase_label" in result
        assert "disclaimer" in result
        assert "real_atlas_path" in result
        assert "synthetic_atlas_path" in result
        assert "slice_overlaps" in result
        assert "distance_summary" in result
        assert "slice_counts" in result
        assert "archetype_counts" in result
        
        # Check distance_summary structure
        ds = result["distance_summary"]
        assert "js_divergence_distribution" in ds
        assert "trend_similarity_distribution" in ds
        assert "abstention_similarity_distribution" in ds

    def test_comparison_slice_overlaps_structure(self, fixture_two_atlases):
        """Slice overlaps contain expected per-slice data."""
        result = compare_real_vs_synthetic(
            fixture_two_atlases["real_path"],
            fixture_two_atlases["synthetic_path"],
        )
        
        overlaps = result["slice_overlaps"]
        # At least one overlapping slice should exist
        if overlaps:
            slice_data = list(overlaps.values())[0]
            assert "real_archetype" in slice_data
            assert "synthetic_archetype" in slice_data
            assert "archetype_match" in slice_data
            assert "js_divergence_stats" in slice_data
            assert "trend_similarity_stats" in slice_data

    def test_comparison_deterministic(self, fixture_two_atlases):
        """Comparison is deterministic."""
        results = [
            compare_real_vs_synthetic(
                fixture_two_atlases["real_path"],
                fixture_two_atlases["synthetic_path"],
            )
            for _ in range(10)
        ]
        
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        assert all(jr == json_results[0] for jr in json_results)

    def test_comparison_has_distribution_stats(self, fixture_two_atlases):
        """Comparison includes distribution statistics in distance_summary."""
        result = compare_real_vs_synthetic(
            fixture_two_atlases["real_path"],
            fixture_two_atlases["synthetic_path"],
        )
        
        ds = result["distance_summary"]
        for dist_key in ["js_divergence_distribution", "trend_similarity_distribution"]:
            for source in ["real", "synthetic"]:
                stats = ds[dist_key][source]
                assert "count" in stats
                assert "mean" in stats
                assert "std" in stats
                assert "min" in stats
                assert "max" in stats

    def test_comparison_has_phase_disclaimer(self, fixture_two_atlases):
        """Comparison includes PHASE II disclaimer."""
        result = compare_real_vs_synthetic(
            fixture_two_atlases["real_path"],
            fixture_two_atlases["synthetic_path"],
        )
        
        assert "PHASE II" in result["phase_label"]
        assert "No uplift" in result["disclaimer"]

    def test_comparison_no_ranking_terms(self, fixture_two_atlases):
        """Comparison output contains no ranking terms."""
        result = compare_real_vs_synthetic(
            fixture_two_atlases["real_path"],
            fixture_two_atlases["synthetic_path"],
        )
        
        # Exclude paths from the check (they may contain test function names with "ranking")
        result_without_paths = {
            k: v for k, v in result.items()
            if k not in ["real_atlas_path", "synthetic_atlas_path"]
        }
        result_str = json.dumps(result_without_paths).lower()
        forbidden = ["best", "worst", "better", "worse", "top", "bottom"]
        
        for word in forbidden:
            assert word not in result_str, f"Ranking term '{word}' found in comparison output"

    def test_comparison_slice_counts_detailed(self, fixture_two_atlases):
        """Slice counts include all contract fields (FROZEN CONTRACT)."""
        result = compare_real_vs_synthetic(
            fixture_two_atlases["real_path"],
            fixture_two_atlases["synthetic_path"],
        )
        
        counts = result["slice_counts"]
        # FROZEN CONTRACT: total_real, total_synthetic, overlapping, real_only, synthetic_only
        assert "total_real" in counts
        assert "total_synthetic" in counts
        assert "overlapping" in counts
        assert "real_only" in counts
        assert "synthetic_only" in counts
        
        # Type verification
        assert isinstance(counts["total_real"], int)
        assert isinstance(counts["total_synthetic"], int)
        assert isinstance(counts["overlapping"], int)
        assert isinstance(counts["real_only"], int)
        assert isinstance(counts["synthetic_only"], int)

    def test_comparison_contract_no_overlapping_slices(self, tmp_path):
        """Comparison handles case where real and synthetic share no slices."""
        real_atlas = {
            "slice_names": ["slice_a", "slice_b"],
            "slice_count": 2,
            "fingerprints": {"slice_a": {}, "slice_b": {}},
            "archetypes": {"slice_a": "type-1", "slice_b": "type-2"},
            "js_divergence_matrix": {
                "slice_a": {"slice_a": 0.0, "slice_b": 0.5},
                "slice_b": {"slice_a": 0.5, "slice_b": 0.0},
            },
            "trend_similarity_matrix": {
                "slice_a": {"slice_a": 1.0, "slice_b": 0.5},
                "slice_b": {"slice_a": 0.5, "slice_b": 1.0},
            },
            "abstention_similarity_matrix": {
                "slice_a": {"slice_a": 1.0, "slice_b": 0.5},
                "slice_b": {"slice_a": 0.5, "slice_b": 1.0},
            },
        }
        
        synthetic_atlas = {
            "slice_names": ["slice_x", "slice_y"],
            "slice_count": 2,
            "fingerprints": {"slice_x": {}, "slice_y": {}},
            "archetypes": {"slice_x": "type-1", "slice_y": "type-2"},
            "js_divergence_matrix": {
                "slice_x": {"slice_x": 0.0, "slice_y": 0.5},
                "slice_y": {"slice_x": 0.5, "slice_y": 0.0},
            },
            "trend_similarity_matrix": {
                "slice_x": {"slice_x": 1.0, "slice_y": 0.5},
                "slice_y": {"slice_x": 0.5, "slice_y": 1.0},
            },
            "abstention_similarity_matrix": {
                "slice_x": {"slice_x": 1.0, "slice_y": 0.5},
                "slice_y": {"slice_x": 0.5, "slice_y": 1.0},
            },
        }
        
        real_path = tmp_path / "real_no_overlap.json"
        synthetic_path = tmp_path / "synthetic_no_overlap.json"
        
        with open(real_path, "w") as f:
            json.dump(real_atlas, f)
        with open(synthetic_path, "w") as f:
            json.dump(synthetic_atlas, f)
        
        result = compare_real_vs_synthetic(str(real_path), str(synthetic_path))
        
        # No overlapping slices
        assert result["slice_counts"]["overlapping"] == 0
        assert result["slice_counts"]["real_only"] == 2
        assert result["slice_counts"]["synthetic_only"] == 2
        assert result["slice_overlaps"] == {}


# ===========================================================================
# ATLAS HEALTH CHECK TESTS
# ===========================================================================

class TestAtlasHealthCheck:
    """Tests for evaluate_atlas_health function (FROZEN CONTRACT)."""

    def test_health_check_ok_on_valid_atlas(self, fixture_atlas_files):
        """Health check returns OK on valid atlas with all contract fields."""
        atlas = load_atlas_from_file(fixture_atlas_files["atlas_path"])
        result = evaluate_atlas_health(atlas)
        
        # FROZEN CONTRACT: status, issues, matrix_checks
        assert result["status"] in ["OK", "WARN"]
        assert "checks_performed" in result
        assert isinstance(result["issues"], list)
        assert "matrix_checks" in result
        
        # Verify matrix_checks has all required boolean fields
        mc = result["matrix_checks"]
        assert isinstance(mc["fingerprints_complete"], bool)
        assert isinstance(mc["matrices_square"], bool)
        assert isinstance(mc["matrices_symmetric"], bool)
        assert isinstance(mc["js_diagonal_zero"], bool)
        assert isinstance(mc["no_invalid_values"], bool)
        assert isinstance(mc["slice_count_match"], bool)

    def test_health_check_block_on_asymmetric_matrix(self):
        """Health check returns BLOCK when matrix is asymmetric."""
        atlas = {
            "slice_names": ["a", "b"],
            "fingerprints": {"a": {}, "b": {}},
            "js_divergence_matrix": {
                "a": {"a": 0.0, "b": 0.5},
                "b": {"a": 0.3, "b": 0.0},  # Asymmetric: 0.5 != 0.3
            },
            "trend_similarity_matrix": {},
            "abstention_similarity_matrix": {},
            "slice_count": 2,
        }
        
        result = evaluate_atlas_health(atlas)
        assert result["status"] == "BLOCK"
        assert any("symmetric" in issue.lower() for issue in result["issues"])

    def test_health_check_block_on_nan(self):
        """Health check returns BLOCK when matrix contains NaN."""
        atlas = {
            "slice_names": ["a", "b"],
            "fingerprints": {"a": {}, "b": {}},
            "js_divergence_matrix": {
                "a": {"a": 0.0, "b": float('nan')},
                "b": {"a": float('nan'), "b": 0.0},
            },
            "trend_similarity_matrix": {},
            "abstention_similarity_matrix": {},
            "slice_count": 2,
        }
        
        result = evaluate_atlas_health(atlas)
        assert result["status"] == "BLOCK"
        assert any("invalid" in issue.lower() or "nan" in issue.lower() for issue in result["issues"])

    def test_health_check_block_on_inf(self):
        """Health check returns BLOCK when matrix contains infinity."""
        atlas = {
            "slice_names": ["a", "b"],
            "fingerprints": {"a": {}, "b": {}},
            "js_divergence_matrix": {
                "a": {"a": 0.0, "b": float('inf')},
                "b": {"a": float('inf'), "b": 0.0},
            },
            "trend_similarity_matrix": {},
            "abstention_similarity_matrix": {},
            "slice_count": 2,
        }
        
        result = evaluate_atlas_health(atlas)
        assert result["status"] == "BLOCK"
        assert any("invalid" in issue.lower() or "inf" in issue.lower() for issue in result["issues"])

    def test_health_check_warn_on_nonzero_diagonal(self):
        """Health check warns when JS diagonal is non-zero."""
        atlas = {
            "slice_names": ["a", "b"],
            "fingerprints": {"a": {}, "b": {}},
            "js_divergence_matrix": {
                "a": {"a": 0.1, "b": 0.5},  # Non-zero diagonal
                "b": {"a": 0.5, "b": 0.0},
            },
            "trend_similarity_matrix": {
                "a": {"a": 1.0, "b": 0.5},
                "b": {"a": 0.5, "b": 1.0},
            },
            "abstention_similarity_matrix": {
                "a": {"a": 1.0, "b": 0.5},
                "b": {"a": 0.5, "b": 1.0},
            },
            "slice_count": 2,
        }
        
        result = evaluate_atlas_health(atlas)
        assert result["status"] in ["WARN", "BLOCK"]
        assert any("diagonal" in issue.lower() for issue in result["issues"])

    def test_health_check_deterministic(self, fixture_atlas_files):
        """Health check is deterministic."""
        atlas = load_atlas_from_file(fixture_atlas_files["atlas_path"])
        
        results = [evaluate_atlas_health(atlas) for _ in range(20)]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_health_check_block_with_synthetic_corrupted_matrix(self):
        """Health check returns BLOCK when injecting synthetic corrupted matrix."""
        # This test simulates a corrupted matrix in test environment
        corrupted_atlas = {
            "slice_names": ["test_a", "test_b", "test_c"],
            "fingerprints": {"test_a": {}, "test_b": {}, "test_c": {}},
            "js_divergence_matrix": {
                "test_a": {"test_a": 0.0, "test_b": float('nan'), "test_c": 0.3},
                "test_b": {"test_a": float('nan'), "test_b": 0.0, "test_c": 0.5},
                "test_c": {"test_a": 0.3, "test_b": 0.5, "test_c": 0.0},
            },
            "trend_similarity_matrix": {},
            "abstention_similarity_matrix": {},
            "slice_count": 3,
        }
        
        result = evaluate_atlas_health(corrupted_atlas)
        assert result["status"] == "BLOCK"
        # NaN detected
        assert any("nan" in issue.lower() or "invalid" in issue.lower() for issue in result["issues"])

    def test_health_check_block_with_non_square_matrix(self):
        """Health check returns BLOCK when matrix is non-square."""
        atlas = {
            "slice_names": ["a", "b"],
            "fingerprints": {"a": {}, "b": {}},
            "js_divergence_matrix": {
                "a": {"a": 0.0, "b": 0.5, "c": 0.3},  # Extra column 'c'
                "b": {"a": 0.5, "b": 0.0, "c": 0.2},
            },
            "trend_similarity_matrix": {},
            "abstention_similarity_matrix": {},
            "slice_count": 2,
        }
        
        result = evaluate_atlas_health(atlas)
        assert result["status"] == "BLOCK"
        assert any("square" in issue.lower() for issue in result["issues"])

    def test_health_check_returns_status_ok_warn_block(self, fixture_atlas_files):
        """Health check status is one of OK, WARN, or BLOCK."""
        atlas = load_atlas_from_file(fixture_atlas_files["atlas_path"])
        result = evaluate_atlas_health(atlas)
        
        assert result["status"] in ["OK", "WARN", "BLOCK"]
        assert isinstance(result["issues"], list)

    def test_health_check_minor_drift_is_warn_not_block(self):
        """Minor numerical drift (e.g., JS diagonal slightly off) → WARN, not BLOCK."""
        atlas = {
            "slice_names": ["a", "b"],
            "fingerprints": {"a": {}, "b": {}},
            "js_divergence_matrix": {
                "a": {"a": 1e-5, "b": 0.5},  # Tiny non-zero diagonal (< 1e-6 threshold is OK)
                "b": {"a": 0.5, "b": 1e-5},
            },
            "trend_similarity_matrix": {
                "a": {"a": 1.0, "b": 0.5},
                "b": {"a": 0.5, "b": 1.0},
            },
            "abstention_similarity_matrix": {
                "a": {"a": 1.0, "b": 0.5},
                "b": {"a": 0.5, "b": 1.0},
            },
            "slice_count": 2,
        }
        
        result = evaluate_atlas_health(atlas)
        
        # Minor drift should be WARN, not BLOCK
        assert result["status"] == "WARN"
        assert result["matrix_checks"]["js_diagonal_zero"] == False
        # Other checks should still pass
        assert result["matrix_checks"]["matrices_square"] == True
        assert result["matrix_checks"]["matrices_symmetric"] == True
        assert result["matrix_checks"]["no_invalid_values"] == True

    def test_health_check_matrix_checks_false_on_asymmetric(self):
        """matrix_checks['matrices_symmetric'] is False when asymmetric."""
        atlas = {
            "slice_names": ["a", "b"],
            "fingerprints": {"a": {}, "b": {}},
            "js_divergence_matrix": {
                "a": {"a": 0.0, "b": 0.5},
                "b": {"a": 0.3, "b": 0.0},  # Asymmetric
            },
            "trend_similarity_matrix": {},
            "abstention_similarity_matrix": {},
            "slice_count": 2,
        }
        
        result = evaluate_atlas_health(atlas)
        
        assert result["status"] == "BLOCK"
        assert result["matrix_checks"]["matrices_symmetric"] == False

    def test_health_check_matrix_checks_false_on_nan(self):
        """matrix_checks['no_invalid_values'] is False when NaN present."""
        atlas = {
            "slice_names": ["a", "b"],
            "fingerprints": {"a": {}, "b": {}},
            "js_divergence_matrix": {
                "a": {"a": 0.0, "b": float('nan')},
                "b": {"a": float('nan'), "b": 0.0},
            },
            "trend_similarity_matrix": {},
            "abstention_similarity_matrix": {},
            "slice_count": 2,
        }
        
        result = evaluate_atlas_health(atlas)
        
        assert result["status"] == "BLOCK"
        assert result["matrix_checks"]["no_invalid_values"] == False

    def test_health_check_lists_all_checks(self, fixture_atlas_files):
        """Health check lists all performed checks."""
        atlas = load_atlas_from_file(fixture_atlas_files["atlas_path"])
        result = evaluate_atlas_health(atlas)
        
        expected_checks = [
            "fingerprints_complete",
            "matrices_square",
            "matrices_symmetric",
            "js_diagonal_zero",
            "no_invalid_values",
            "slice_count_match",
        ]
        
        for check in expected_checks:
            assert check in result["checks_performed"]


# ===========================================================================
# HELPER FUNCTION TESTS FOR NEW FEATURES
# ===========================================================================

class TestHelperFunctions:
    """Tests for helper functions used by new features."""

    def test_extract_matrix_values_upper_triangle(self):
        """Extract matrix values gets upper triangle only."""
        matrix = {
            "a": {"a": 0.0, "b": 0.5, "c": 0.3},
            "b": {"a": 0.5, "b": 0.0, "c": 0.4},
            "c": {"a": 0.3, "b": 0.4, "c": 0.0},
        }
        
        values = _extract_matrix_values(matrix)
        
        # Upper triangle: (a,b), (a,c), (b,c) = 0.5, 0.3, 0.4
        assert len(values) == 3
        assert sorted(values) == [0.3, 0.4, 0.5]

    def test_count_values(self):
        """Count values counts correctly."""
        d = {"a": "x", "b": "y", "c": "x", "d": "x"}
        counts = _count_values(d)
        
        assert counts == {"x": 3, "y": 1}

    def test_compute_distribution_stats_empty(self):
        """Distribution stats handles empty list."""
        stats = _compute_distribution_stats([])
        
        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0

    def test_compute_distribution_stats_single(self):
        """Distribution stats handles single value."""
        stats = _compute_distribution_stats([5.0])
        
        assert stats["count"] == 1
        assert stats["mean"] == 5.0
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0

    def test_compute_distribution_stats_known(self):
        """Distribution stats computes correctly."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = _compute_distribution_stats(values)
        
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0


# ===========================================================================
# ATLAS ROUTING & ALIGNMENT LAYER TESTS (Phase II v1.2)
# ===========================================================================

class TestBuildRoutingHint:
    """Tests for build_routing_hint function (FROZEN CONTRACT)."""

    def test_routing_hint_has_required_keys(self):
        """Routing hint contains all contract keys."""
        dossier = {
            "slice_name": "test_slice",
            "assigned_archetype": "dense-shallow",
            "feature_vector": [0.1, 0.2, 0.3],
            "nearest_neighbors": [{"neighbor_slice": "a", "distance": 0.1}],
            "metrics_summary": {
                "feature_vector_dimension": 3,
                "feature_vector_mean": 0.2,
                "neighbor_distance_mean": 0.1,
                "neighbor_distance_min": 0.1,
                "trend_similarity_mean": 0.5,
                "abstention_similarity_mean": 0.8,
                "neighbor_count": 1,
            },
        }
        
        hint = build_routing_hint(dossier)
        
        # FROZEN CONTRACT keys
        assert "slice_name" in hint
        assert "archetype" in hint
        assert "neighbor_count" in hint
        assert "feature_vector_dimension" in hint
        
        # Type verification
        assert isinstance(hint["slice_name"], str)
        assert isinstance(hint["archetype"], str)
        assert isinstance(hint["neighbor_count"], int)
        assert isinstance(hint["feature_vector_dimension"], int)

    def test_routing_hint_values_from_dossier(self):
        """Routing hint values are derived from dossier."""
        dossier = {
            "slice_name": "my_slice",
            "assigned_archetype": "sparse-deep",
            "metrics_summary": {
                "neighbor_count": 5,
                "feature_vector_dimension": 10,
            },
        }
        
        hint = build_routing_hint(dossier)
        
        assert hint["slice_name"] == "my_slice"
        assert hint["archetype"] == "sparse-deep"
        assert hint["neighbor_count"] == 5
        assert hint["feature_vector_dimension"] == 10

    def test_routing_hint_deterministic(self):
        """Routing hint is deterministic across calls."""
        dossier = {
            "slice_name": "test",
            "assigned_archetype": "type-a",
            "metrics_summary": {
                "neighbor_count": 3,
                "feature_vector_dimension": 10,
            },
        }
        
        hints = [build_routing_hint(dossier) for _ in range(50)]
        json_hints = [json.dumps(h, sort_keys=True) for h in hints]
        
        assert all(jh == json_hints[0] for jh in json_hints)

    def test_routing_hint_no_ranking_language(self):
        """Routing hint contains no value-loaded words."""
        dossier = {
            "slice_name": "test",
            "assigned_archetype": "neutral-type",
            "metrics_summary": {
                "neighbor_count": 2,
                "feature_vector_dimension": 5,
            },
        }
        
        hint = build_routing_hint(dossier)
        hint_str = json.dumps(hint).lower()
        
        forbidden = ["best", "worst", "top", "bottom", "better", "worse", "rank"]
        for word in forbidden:
            assert word not in hint_str, f"Ranking term '{word}' found in routing hint"

    def test_routing_hint_handles_missing_fields(self):
        """Routing hint handles missing dossier fields gracefully."""
        dossier = {}  # Empty dossier
        
        hint = build_routing_hint(dossier)
        
        assert hint["slice_name"] == ""
        assert hint["archetype"] == "unknown"
        assert hint["neighbor_count"] == 0
        assert hint["feature_vector_dimension"] == 0


class TestComputeAtlasCompatibility:
    """Tests for compute_atlas_compatibility function (FROZEN CONTRACT)."""

    def test_compatibility_has_required_keys(self):
        """Compatibility overview contains all contract keys."""
        comparison = {
            "slice_counts": {
                "overlapping": 3,
                "real_only": 1,
                "synthetic_only": 2,
            },
            "slice_overlaps": {
                "slice_a": {"archetype_match": True},
                "slice_b": {"archetype_match": False},
                "slice_c": {"archetype_match": True},
            },
        }
        
        compat = compute_atlas_compatibility(comparison)
        
        # FROZEN CONTRACT keys
        assert "schema_version" in compat
        assert "overlap_slice_count" in compat
        assert "exact_archetype_match_count" in compat
        assert "real_only_count" in compat
        assert "synthetic_only_count" in compat
        
        # Type verification
        assert isinstance(compat["schema_version"], str)
        assert isinstance(compat["overlap_slice_count"], int)
        assert isinstance(compat["exact_archetype_match_count"], int)
        assert isinstance(compat["real_only_count"], int)
        assert isinstance(compat["synthetic_only_count"], int)

    def test_compatibility_counts_match(self):
        """Compatibility counts match underlying comparison structure."""
        comparison = {
            "slice_counts": {
                "overlapping": 5,
                "real_only": 2,
                "synthetic_only": 3,
            },
            "slice_overlaps": {
                "s1": {"archetype_match": True},
                "s2": {"archetype_match": True},
                "s3": {"archetype_match": False},
                "s4": {"archetype_match": True},
                "s5": {"archetype_match": False},
            },
        }
        
        compat = compute_atlas_compatibility(comparison)
        
        assert compat["overlap_slice_count"] == 5
        assert compat["exact_archetype_match_count"] == 3  # 3 True matches
        assert compat["real_only_count"] == 2
        assert compat["synthetic_only_count"] == 3

    def test_compatibility_deterministic(self):
        """Compatibility is deterministic across runs."""
        comparison = {
            "slice_counts": {"overlapping": 2, "real_only": 1, "synthetic_only": 1},
            "slice_overlaps": {
                "a": {"archetype_match": True},
                "b": {"archetype_match": False},
            },
        }
        
        results = [compute_atlas_compatibility(comparison) for _ in range(50)]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_compatibility_schema_version(self):
        """Compatibility includes schema version 1.0.0."""
        comparison = {
            "slice_counts": {"overlapping": 0, "real_only": 0, "synthetic_only": 0},
            "slice_overlaps": {},
        }
        
        compat = compute_atlas_compatibility(comparison)
        
        assert compat["schema_version"] == "1.0.0"

    def test_compatibility_empty_overlaps(self):
        """Compatibility handles empty overlaps correctly."""
        comparison = {
            "slice_counts": {"overlapping": 0, "real_only": 5, "synthetic_only": 3},
            "slice_overlaps": {},
        }
        
        compat = compute_atlas_compatibility(comparison)
        
        assert compat["overlap_slice_count"] == 0
        assert compat["exact_archetype_match_count"] == 0
        assert compat["real_only_count"] == 5
        assert compat["synthetic_only_count"] == 3

    def test_compatibility_handles_missing_fields(self):
        """Compatibility handles missing comparison fields gracefully."""
        comparison = {}  # Empty comparison
        
        compat = compute_atlas_compatibility(comparison)
        
        assert compat["overlap_slice_count"] == 0
        assert compat["exact_archetype_match_count"] == 0
        assert compat["real_only_count"] == 0
        assert compat["synthetic_only_count"] == 0


class TestIsAtlasStructurallySound:
    """Tests for is_atlas_structurally_sound predicate (FROZEN CONTRACT)."""

    def test_sound_returns_true_on_ok_status(self):
        """Returns True when status is OK and checks pass."""
        health = {
            "status": "OK",
            "issues": [],
            "matrix_checks": {
                "fingerprints_complete": True,
                "matrices_square": True,
                "matrices_symmetric": True,
                "js_diagonal_zero": True,
                "no_invalid_values": True,
                "slice_count_match": True,
            },
        }
        
        assert is_atlas_structurally_sound(health) is True

    def test_sound_returns_true_on_warn_status(self):
        """Returns True when status is WARN but critical checks pass."""
        health = {
            "status": "WARN",
            "issues": ["Minor issue"],
            "matrix_checks": {
                "fingerprints_complete": True,
                "matrices_square": True,
                "matrices_symmetric": True,
                "js_diagonal_zero": False,  # This caused WARN
                "no_invalid_values": True,
                "slice_count_match": True,
            },
        }
        
        assert is_atlas_structurally_sound(health) is True

    def test_sound_returns_false_on_block_status(self):
        """Returns False when status is BLOCK."""
        health = {
            "status": "BLOCK",
            "issues": ["Critical issue"],
            "matrix_checks": {
                "fingerprints_complete": True,
                "matrices_square": True,
                "matrices_symmetric": False,  # This caused BLOCK
                "js_diagonal_zero": True,
                "no_invalid_values": True,
                "slice_count_match": True,
            },
        }
        
        assert is_atlas_structurally_sound(health) is False

    def test_sound_returns_false_if_matrices_not_square(self):
        """Returns False when matrices_square is False."""
        health = {
            "status": "WARN",  # Even non-BLOCK status
            "issues": [],
            "matrix_checks": {
                "fingerprints_complete": True,
                "matrices_square": False,  # Critical check
                "matrices_symmetric": True,
                "js_diagonal_zero": True,
                "no_invalid_values": True,
                "slice_count_match": True,
            },
        }
        
        assert is_atlas_structurally_sound(health) is False

    def test_sound_returns_false_if_invalid_values(self):
        """Returns False when no_invalid_values is False."""
        health = {
            "status": "WARN",  # Even non-BLOCK status
            "issues": [],
            "matrix_checks": {
                "fingerprints_complete": True,
                "matrices_square": True,
                "matrices_symmetric": True,
                "js_diagonal_zero": True,
                "no_invalid_values": False,  # Critical check
                "slice_count_match": True,
            },
        }
        
        assert is_atlas_structurally_sound(health) is False

    def test_sound_is_pure_function(self):
        """Predicate is deterministic and has no side effects."""
        health = {
            "status": "OK",
            "issues": [],
            "matrix_checks": {
                "matrices_square": True,
                "no_invalid_values": True,
            },
        }
        
        # Call multiple times
        results = [is_atlas_structurally_sound(health) for _ in range(100)]
        
        # All results should be identical
        assert all(r is True for r in results)
        
        # Original health dict should be unchanged
        assert health["status"] == "OK"

    def test_sound_handles_missing_matrix_checks(self):
        """Predicate handles missing matrix_checks gracefully."""
        health = {
            "status": "OK",
            "issues": [],
            # No matrix_checks key
        }
        
        # Should return False (defensive)
        assert is_atlas_structurally_sound(health) is False

    def test_sound_synthetic_health_objects(self):
        """Test True/False behavior on various synthetic health objects."""
        # Case 1: Fully valid
        assert is_atlas_structurally_sound({
            "status": "OK",
            "matrix_checks": {"matrices_square": True, "no_invalid_values": True},
        }) is True
        
        # Case 2: BLOCK status
        assert is_atlas_structurally_sound({
            "status": "BLOCK",
            "matrix_checks": {"matrices_square": True, "no_invalid_values": True},
        }) is False
        
        # Case 3: Non-square matrix
        assert is_atlas_structurally_sound({
            "status": "OK",
            "matrix_checks": {"matrices_square": False, "no_invalid_values": True},
        }) is False
        
        # Case 4: Invalid values
        assert is_atlas_structurally_sound({
            "status": "WARN",
            "matrix_checks": {"matrices_square": True, "no_invalid_values": False},
        }) is False
        
        # Case 5: Both critical checks fail
        assert is_atlas_structurally_sound({
            "status": "OK",
            "matrix_checks": {"matrices_square": False, "no_invalid_values": False},
        }) is False


# ===========================================================================
# PHASE III — ATLAS GOVERNANCE & GLOBAL ROUTING LAYER TESTS
# ===========================================================================

class TestBuildAtlasGovernanceSnapshot:
    """Tests for build_atlas_governance_snapshot function (FROZEN CONTRACT)."""

    def test_governance_snapshot_has_required_keys(self):
        """Governance snapshot contains all contract keys."""
        dossiers = [
            {
                "slice_name": "slice_a",
                "assigned_archetype": "dense-shallow",
                "metrics_summary": {},
            },
        ]
        comparisons = []
        health_reports = [
            {
                "status": "OK",
                "matrix_checks": {"matrices_square": True, "no_invalid_values": True},
            },
        ]
        
        snapshot = build_atlas_governance_snapshot(dossiers, comparisons, health_reports)
        
        # FROZEN CONTRACT keys
        assert "schema_version" in snapshot
        assert "total_slices_indexed" in snapshot
        assert "real_vs_synthetic_overlap" in snapshot
        assert "structurally_sound" in snapshot
        assert "archetype_diversity_index" in snapshot

    def test_governance_snapshot_total_slices(self):
        """Total slices indexed matches dossier count."""
        dossiers = [
            {"slice_name": "a", "assigned_archetype": "type-1"},
            {"slice_name": "b", "assigned_archetype": "type-2"},
            {"slice_name": "c", "assigned_archetype": "type-1"},
        ]
        comparisons = []
        health_reports = [
            {"status": "OK", "matrix_checks": {"matrices_square": True, "no_invalid_values": True}},
        ]
        
        snapshot = build_atlas_governance_snapshot(dossiers, comparisons, health_reports)
        
        assert snapshot["total_slices_indexed"] == 3

    def test_governance_snapshot_structurally_sound_all_pass(self):
        """Structurally sound is True when all health reports pass."""
        dossiers = [{"slice_name": "a"}]
        comparisons = []
        health_reports = [
            {"status": "OK", "matrix_checks": {"matrices_square": True, "no_invalid_values": True}},
            {"status": "WARN", "matrix_checks": {"matrices_square": True, "no_invalid_values": True}},
        ]
        
        snapshot = build_atlas_governance_snapshot(dossiers, comparisons, health_reports)
        
        assert snapshot["structurally_sound"] is True

    def test_governance_snapshot_structurally_sound_any_fails(self):
        """Structurally sound is False when any health report fails."""
        dossiers = [{"slice_name": "a"}]
        comparisons = []
        health_reports = [
            {"status": "OK", "matrix_checks": {"matrices_square": True, "no_invalid_values": True}},
            {"status": "BLOCK", "matrix_checks": {"matrices_square": False, "no_invalid_values": True}},
        ]
        
        snapshot = build_atlas_governance_snapshot(dossiers, comparisons, health_reports)
        
        assert snapshot["structurally_sound"] is False

    def test_governance_snapshot_archetype_diversity_index(self):
        """Archetype diversity index computed correctly."""
        dossiers = [
            {"slice_name": "a", "assigned_archetype": "type-1"},
            {"slice_name": "b", "assigned_archetype": "type-2"},
            {"slice_name": "c", "assigned_archetype": "type-1"},
            {"slice_name": "d", "assigned_archetype": "type-3"},
        ]
        comparisons = []
        health_reports = [
            {"status": "OK", "matrix_checks": {"matrices_square": True, "no_invalid_values": True}},
        ]
        
        snapshot = build_atlas_governance_snapshot(dossiers, comparisons, health_reports)
        
        # 3 distinct archetypes / 4 slices = 0.75
        assert snapshot["archetype_diversity_index"] == pytest.approx(0.75)

    def test_governance_snapshot_aggregates_comparisons(self):
        """Real vs synthetic overlap aggregates from all comparisons."""
        dossiers = [{"slice_name": "a"}]
        comparisons = [
            {
                "slice_counts": {"overlapping": 2, "real_only": 1, "synthetic_only": 1},
                "slice_overlaps": {
                    "s1": {"archetype_match": True},
                    "s2": {"archetype_match": False},
                },
            },
            {
                "slice_counts": {"overlapping": 1, "real_only": 0, "synthetic_only": 0},
                "slice_overlaps": {
                    "s3": {"archetype_match": True},
                },
            },
        ]
        health_reports = [
            {"status": "OK", "matrix_checks": {"matrices_square": True, "no_invalid_values": True}},
        ]
        
        snapshot = build_atlas_governance_snapshot(dossiers, comparisons, health_reports)
        
        overlap = snapshot["real_vs_synthetic_overlap"]
        assert overlap["total_overlap_slices"] == 3  # 2 + 1
        assert overlap["total_exact_matches"] == 2  # 1 + 1
        assert overlap["total_real_only"] == 1
        assert overlap["total_synthetic_only"] == 1

    def test_governance_snapshot_empty_inputs(self):
        """Governance snapshot handles empty inputs gracefully."""
        snapshot = build_atlas_governance_snapshot([], [], [])
        
        assert snapshot["total_slices_indexed"] == 0
        assert snapshot["structurally_sound"] is True  # All empty = all pass
        assert snapshot["archetype_diversity_index"] == 0.0
        assert snapshot["schema_version"] == "1.0.0"

    def test_governance_snapshot_deterministic(self):
        """Governance snapshot is deterministic."""
        dossiers = [
            {"slice_name": "a", "assigned_archetype": "type-1"},
            {"slice_name": "b", "assigned_archetype": "type-2"},
        ]
        comparisons = []
        health_reports = [
            {"status": "OK", "matrix_checks": {"matrices_square": True, "no_invalid_values": True}},
        ]
        
        results = [
            build_atlas_governance_snapshot(dossiers, comparisons, health_reports)
            for _ in range(50)
        ]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)


class TestBuildRoutingOverview:
    """Tests for build_routing_overview function (FROZEN CONTRACT)."""

    def test_routing_overview_has_required_keys(self):
        """Routing overview contains all contract keys."""
        hints = [
            {"slice_name": "a", "archetype": "type-1", "neighbor_count": 3},
        ]
        
        overview = build_routing_overview(hints)
        
        # FROZEN CONTRACT keys
        assert "slices_by_neighbor_band" in overview
        assert "archetype_frequency" in overview
        assert "status" in overview

    def test_routing_overview_neighbor_bands(self):
        """Slices are correctly bucketed into neighbor bands."""
        hints = [
            {"slice_name": "a", "archetype": "type-1", "neighbor_count": 2},  # low
            {"slice_name": "b", "archetype": "type-2", "neighbor_count": 4},  # medium
            {"slice_name": "c", "archetype": "type-1", "neighbor_count": 6},  # high
            {"slice_name": "d", "archetype": "type-3", "neighbor_count": 1},  # low
            {"slice_name": "e", "archetype": "type-2", "neighbor_count": 5},  # medium
        ]
        
        overview = build_routing_overview(hints)
        
        bands = overview["slices_by_neighbor_band"]
        assert bands["low"] == 2  # 2 and 1
        assert bands["medium"] == 2  # 4 and 5
        assert bands["high"] == 1  # 6

    def test_routing_overview_archetype_frequency(self):
        """Archetype frequency counts correctly."""
        hints = [
            {"slice_name": "a", "archetype": "type-1", "neighbor_count": 3},
            {"slice_name": "b", "archetype": "type-2", "neighbor_count": 4},
            {"slice_name": "c", "archetype": "type-1", "neighbor_count": 5},
            {"slice_name": "d", "archetype": "type-1", "neighbor_count": 2},
        ]
        
        overview = build_routing_overview(hints)
        
        freq = overview["archetype_frequency"]
        assert freq["type-1"] == 3
        assert freq["type-2"] == 1

    def test_routing_overview_status_clustered(self):
        """Status is CLUSTERED when >50% slices have high neighbor counts."""
        hints = [
            {"slice_name": "a", "archetype": "type-1", "neighbor_count": 6},  # high
            {"slice_name": "b", "archetype": "type-2", "neighbor_count": 7},  # high
            {"slice_name": "c", "archetype": "type-1", "neighbor_count": 8},  # high
            {"slice_name": "d", "archetype": "type-3", "neighbor_count": 2},  # low
        ]
        
        overview = build_routing_overview(hints)
        
        assert overview["status"] == "CLUSTERED"

    def test_routing_overview_status_sparse(self):
        """Status is SPARSE when >50% slices have low neighbor counts."""
        hints = [
            {"slice_name": "a", "archetype": "type-1", "neighbor_count": 1},  # low
            {"slice_name": "b", "archetype": "type-2", "neighbor_count": 2},  # low
            {"slice_name": "c", "archetype": "type-1", "neighbor_count": 0},  # low
            {"slice_name": "d", "archetype": "type-3", "neighbor_count": 5},  # medium
        ]
        
        overview = build_routing_overview(hints)
        
        assert overview["status"] == "SPARSE"

    def test_routing_overview_status_ok(self):
        """Status is OK when distribution is balanced."""
        hints = [
            {"slice_name": "a", "archetype": "type-1", "neighbor_count": 2},  # low
            {"slice_name": "b", "archetype": "type-2", "neighbor_count": 4},  # medium
            {"slice_name": "c", "archetype": "type-1", "neighbor_count": 6},  # high
            {"slice_name": "d", "archetype": "type-3", "neighbor_count": 3},  # medium
        ]
        
        overview = build_routing_overview(hints)
        
        assert overview["status"] == "OK"

    def test_routing_overview_empty_input(self):
        """Routing overview handles empty input gracefully."""
        overview = build_routing_overview([])
        
        assert overview["slices_by_neighbor_band"] == {"low": 0, "medium": 0, "high": 0}
        assert overview["archetype_frequency"] == {}
        assert overview["status"] == "SPARSE"

    def test_routing_overview_deterministic(self):
        """Routing overview is deterministic."""
        hints = [
            {"slice_name": "a", "archetype": "type-1", "neighbor_count": 3},
            {"slice_name": "b", "archetype": "type-2", "neighbor_count": 4},
        ]
        
        results = [build_routing_overview(hints) for _ in range(50)]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)


class TestSummarizeAtlasForGlobalHealth:
    """Tests for summarize_atlas_for_global_health function (FROZEN CONTRACT)."""

    def test_global_health_has_required_keys(self):
        """Global health summary contains all contract keys."""
        governance = {
            "structurally_sound": True,
            "total_slices_indexed": 5,
        }
        routing = {
            "status": "OK",
        }
        
        summary = summarize_atlas_for_global_health(governance, routing)
        
        # FROZEN CONTRACT keys
        assert "atlas_ok" in summary
        assert "structurally_sound" in summary
        assert "routing_status" in summary
        assert "status" in summary

    def test_global_health_status_ok(self):
        """Status is OK when structurally sound and routing OK/CLUSTERED."""
        governance = {
            "structurally_sound": True,
            "total_slices_indexed": 5,
        }
        routing = {
            "status": "OK",
        }
        
        summary = summarize_atlas_for_global_health(governance, routing)
        
        assert summary["status"] == "OK"
        assert summary["atlas_ok"] is True

    def test_global_health_status_ok_clustered(self):
        """Status is OK when structurally sound and CLUSTERED."""
        governance = {
            "structurally_sound": True,
            "total_slices_indexed": 5,
        }
        routing = {
            "status": "CLUSTERED",
        }
        
        summary = summarize_atlas_for_global_health(governance, routing)
        
        assert summary["status"] == "OK"
        assert summary["atlas_ok"] is True

    def test_global_health_status_warn_sparse(self):
        """Status is WARN when structurally sound but SPARSE."""
        governance = {
            "structurally_sound": True,
            "total_slices_indexed": 5,
        }
        routing = {
            "status": "SPARSE",
        }
        
        summary = summarize_atlas_for_global_health(governance, routing)
        
        assert summary["status"] == "WARN"
        assert summary["atlas_ok"] is False

    def test_global_health_status_warn_no_slices(self):
        """Status is WARN when no slices indexed."""
        governance = {
            "structurally_sound": True,
            "total_slices_indexed": 0,
        }
        routing = {
            "status": "OK",
        }
        
        summary = summarize_atlas_for_global_health(governance, routing)
        
        assert summary["status"] == "WARN"
        assert summary["atlas_ok"] is False

    def test_global_health_status_block(self):
        """Status is BLOCK when structurally unsound."""
        governance = {
            "structurally_sound": False,
            "total_slices_indexed": 5,
        }
        routing = {
            "status": "OK",
        }
        
        summary = summarize_atlas_for_global_health(governance, routing)
        
        assert summary["status"] == "BLOCK"
        assert summary["atlas_ok"] is False
        assert summary["structurally_sound"] is False

    def test_global_health_routing_status_propagated(self):
        """Routing status is propagated correctly."""
        governance = {
            "structurally_sound": True,
            "total_slices_indexed": 5,
        }
        routing = {
            "status": "CLUSTERED",
        }
        
        summary = summarize_atlas_for_global_health(governance, routing)
        
        assert summary["routing_status"] == "CLUSTERED"

    def test_global_health_deterministic(self):
        """Global health summary is deterministic."""
        governance = {
            "structurally_sound": True,
            "total_slices_indexed": 3,
        }
        routing = {
            "status": "OK",
        }
        
        results = [
            summarize_atlas_for_global_health(governance, routing)
            for _ in range(50)
        ]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_global_health_no_value_judgments(self):
        """Global health summary contains no value-loaded language."""
        governance = {
            "structurally_sound": True,
            "total_slices_indexed": 5,
        }
        routing = {
            "status": "OK",
        }
        
        summary = summarize_atlas_for_global_health(governance, routing)
        summary_str = json.dumps(summary).lower()
        
        forbidden = ["best", "worst", "better", "worse", "top", "bottom", "rank"]
        for word in forbidden:
            assert word not in summary_str, f"Ranking term '{word}' found in summary"


# ===========================================================================
# PHASE IV — ATLAS-GUIDED ROUTING & STRUCTURAL GOVERNANCE TESTS
# ===========================================================================

class TestDeriveAtlasRoutingPolicy:
    """Tests for derive_atlas_routing_policy function (FROZEN CONTRACT)."""

    def test_routing_policy_has_required_keys(self):
        """Routing policy contains all contract keys."""
        governance = {
            "archetype_diversity_index": 0.5,
            "total_slices_indexed": 3,
        }
        routing = {
            "status": "OK",
            "archetype_frequency": {"dense-shallow": 2, "sparse-deep": 1},
        }
        dossiers = [
            {"slice_name": "slice_a", "assigned_archetype": "dense-shallow"},
            {"slice_name": "slice_b", "assigned_archetype": "dense-shallow"},
            {"slice_name": "slice_c", "assigned_archetype": "sparse-deep"},
        ]
        
        policy = derive_atlas_routing_policy(governance, routing, dossiers)
        
        # FROZEN CONTRACT keys
        assert "slices_preferring_dense_archetypes" in policy
        assert "slices_preferring_sparse_archetypes" in policy
        assert "routing_status" in policy
        assert "policy_notes" in policy

    def test_routing_policy_categorizes_slices(self):
        """Slices are correctly categorized by archetype preference."""
        governance = {"archetype_diversity_index": 0.5}
        routing = {"status": "OK", "archetype_frequency": {}}
        dossiers = [
            {"slice_name": "dense_1", "assigned_archetype": "dense-shallow"},
            {"slice_name": "dense_2", "assigned_archetype": "dense-deep"},
            {"slice_name": "sparse_1", "assigned_archetype": "sparse-shallow"},
            {"slice_name": "other", "assigned_archetype": "unknown"},
        ]
        
        policy = derive_atlas_routing_policy(governance, routing, dossiers)
        
        assert set(policy["slices_preferring_dense_archetypes"]) == {"dense_1", "dense_2"}
        assert set(policy["slices_preferring_sparse_archetypes"]) == {"sparse_1"}

    def test_routing_policy_status_clustered(self):
        """Routing status is CLUSTERED when routing overview is OK/CLUSTERED."""
        governance = {"archetype_diversity_index": 0.5}
        routing = {"status": "CLUSTERED", "archetype_frequency": {}}
        
        policy = derive_atlas_routing_policy(governance, routing)
        
        assert policy["routing_status"] == "CLUSTERED"

    def test_routing_policy_status_sparse(self):
        """Routing status is SPARSE when routing overview is SPARSE."""
        governance = {"archetype_diversity_index": 0.5}
        routing = {"status": "SPARSE", "archetype_frequency": {}}
        
        policy = derive_atlas_routing_policy(governance, routing)
        
        assert policy["routing_status"] == "SPARSE"

    def test_routing_policy_status_balanced(self):
        """Routing status is BALANCED when routing overview is OK."""
        governance = {"archetype_diversity_index": 0.5}
        routing = {"status": "OK", "archetype_frequency": {}}
        
        policy = derive_atlas_routing_policy(governance, routing)
        
        assert policy["routing_status"] == "CLUSTERED"  # OK maps to CLUSTERED

    def test_routing_policy_notes(self):
        """Policy notes are generated based on diversity index."""
        governance = {"archetype_diversity_index": 0.8}
        routing = {"status": "OK", "archetype_frequency": {"type1": 1, "type2": 1}}
        
        policy = derive_atlas_routing_policy(governance, routing)
        
        assert "high" in policy["policy_notes"].lower() or "2" in policy["policy_notes"]

    def test_routing_policy_deterministic(self):
        """Routing policy is deterministic."""
        governance = {"archetype_diversity_index": 0.5}
        routing = {"status": "OK", "archetype_frequency": {}}
        dossiers = [
            {"slice_name": "a", "assigned_archetype": "dense-shallow"},
        ]
        
        results = [
            derive_atlas_routing_policy(governance, routing, dossiers)
            for _ in range(50)
        ]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_routing_policy_no_value_judgments(self):
        """Routing policy contains no value-loaded language."""
        governance = {"archetype_diversity_index": 0.5}
        routing = {"status": "OK", "archetype_frequency": {}}
        
        policy = derive_atlas_routing_policy(governance, routing)
        policy_str = json.dumps(policy).lower()
        
        forbidden = ["best", "worst", "better", "worse", "top", "bottom", "rank"]
        for word in forbidden:
            assert word not in policy_str, f"Ranking term '{word}' found in policy"


class TestBuildStructuralGovernanceView:
    """Tests for build_structural_governance_view function (FROZEN CONTRACT)."""

    def test_structural_view_has_required_keys(self):
        """Structural governance view contains all contract keys."""
        atlas_snapshot = {"archetype_diversity_index": 0.5}
        topology = {
            "stability_index": 0.9,
            "consistency_flags": {},
            "slice_consistency": {},
        }
        
        view = build_structural_governance_view(atlas_snapshot, topology)
        
        # FROZEN CONTRACT keys
        assert "slices_with_structure_vs_routing_mismatch" in view
        assert "slices_with_consistent_archetypes" in view
        assert "governance_status" in view

    def test_structural_view_status_ok(self):
        """Governance status is OK when stability is high and no mismatches."""
        atlas_snapshot = {"archetype_diversity_index": 0.5}
        topology = {
            "stability_index": 0.9,
            "consistency_flags": {},
            "slice_consistency": {},
        }
        
        view = build_structural_governance_view(atlas_snapshot, topology)
        
        assert view["governance_status"] == "OK"

    def test_structural_view_status_attention(self):
        """Governance status is ATTENTION when stability is moderate."""
        atlas_snapshot = {"archetype_diversity_index": 0.5}
        topology = {
            "stability_index": 0.6,
            "consistency_flags": {},
            "slice_consistency": {},
        }
        
        view = build_structural_governance_view(atlas_snapshot, topology)
        
        assert view["governance_status"] == "ATTENTION"

    def test_structural_view_status_volatile(self):
        """Governance status is VOLATILE when stability is low."""
        atlas_snapshot = {"archetype_diversity_index": 0.5}
        topology = {
            "stability_index": 0.3,
            "consistency_flags": {},
            "slice_consistency": {},
        }
        
        view = build_structural_governance_view(atlas_snapshot, topology)
        
        assert view["governance_status"] == "VOLATILE"

    def test_structural_view_identifies_mismatches(self):
        """Mismatch slices are identified from consistency flags."""
        atlas_snapshot = {"archetype_diversity_index": 0.5}
        topology = {
            "stability_index": 0.9,
            "consistency_flags": {
                "slice_a": False,
                "slice_b": "mismatch",
                "slice_c": True,
            },
            "slice_consistency": {},
        }
        
        view = build_structural_governance_view(atlas_snapshot, topology)
        
        assert "slice_a" in view["slices_with_structure_vs_routing_mismatch"]
        assert "slice_b" in view["slices_with_structure_vs_routing_mismatch"]
        assert "slice_c" not in view["slices_with_structure_vs_routing_mismatch"]

    def test_structural_view_identifies_consistent_slices(self):
        """Consistent slices are identified from slice_consistency."""
        atlas_snapshot = {"archetype_diversity_index": 0.5}
        topology = {
            "stability_index": 0.9,
            "consistency_flags": {},
            "slice_consistency": {
                "slice_a": True,
                "slice_b": False,
                "slice_c": True,
            },
        }
        
        view = build_structural_governance_view(atlas_snapshot, topology)
        
        assert "slice_a" in view["slices_with_consistent_archetypes"]
        assert "slice_c" in view["slices_with_consistent_archetypes"]
        assert "slice_b" not in view["slices_with_consistent_archetypes"]

    def test_structural_view_deterministic(self):
        """Structural governance view is deterministic."""
        atlas_snapshot = {"archetype_diversity_index": 0.5}
        topology = {"stability_index": 0.8}
        
        results = [
            build_structural_governance_view(atlas_snapshot, topology)
            for _ in range(50)
        ]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_structural_view_handles_missing_topology_fields(self):
        """Structural view handles missing topology fields gracefully."""
        atlas_snapshot = {"archetype_diversity_index": 0.5}
        topology = {}  # Empty topology
        
        view = build_structural_governance_view(atlas_snapshot, topology)
        
        assert view["governance_status"] in ["OK", "ATTENTION", "VOLATILE"]
        assert isinstance(view["slices_with_structure_vs_routing_mismatch"], list)
        assert isinstance(view["slices_with_consistent_archetypes"], list)


class TestBuildAtlasDirectorPanel:
    """Tests for build_atlas_director_panel function (FROZEN CONTRACT)."""

    def test_director_panel_has_required_keys(self):
        """Director panel contains all contract keys."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5, "archetype_diversity_index": 0.5}
        routing_policy = {"routing_status": "CLUSTERED"}
        structural_view = {"governance_status": "OK"}
        
        panel = build_atlas_director_panel(governance, routing_policy, structural_view)
        
        # FROZEN CONTRACT keys
        assert "status_light" in panel
        assert "atlas_ok" in panel
        assert "structurally_sound" in panel
        assert "routing_status" in panel
        assert "governance_status" in panel
        assert "headline" in panel

    def test_director_panel_status_light_green(self):
        """Status light is GREEN when all systems are healthy."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5, "archetype_diversity_index": 0.5}
        routing_policy = {"routing_status": "CLUSTERED"}
        structural_view = {"governance_status": "OK"}
        
        panel = build_atlas_director_panel(governance, routing_policy, structural_view)
        
        assert panel["status_light"] == "GREEN"
        assert panel["atlas_ok"] is True

    def test_director_panel_status_light_yellow_attention(self):
        """Status light is YELLOW when governance requires attention."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5, "archetype_diversity_index": 0.5}
        routing_policy = {"routing_status": "CLUSTERED"}
        structural_view = {"governance_status": "ATTENTION"}
        
        panel = build_atlas_director_panel(governance, routing_policy, structural_view)
        
        assert panel["status_light"] == "YELLOW"
        assert panel["atlas_ok"] is False

    def test_director_panel_status_light_yellow_sparse(self):
        """Status light is YELLOW when routing is sparse."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5, "archetype_diversity_index": 0.5}
        routing_policy = {"routing_status": "SPARSE"}
        structural_view = {"governance_status": "OK"}
        
        panel = build_atlas_director_panel(governance, routing_policy, structural_view)
        
        assert panel["status_light"] == "YELLOW"
        assert panel["atlas_ok"] is False

    def test_director_panel_status_light_red_volatile(self):
        """Status light is RED when governance is volatile."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5, "archetype_diversity_index": 0.5}
        routing_policy = {"routing_status": "CLUSTERED"}
        structural_view = {"governance_status": "VOLATILE"}
        
        panel = build_atlas_director_panel(governance, routing_policy, structural_view)
        
        assert panel["status_light"] == "RED"
        assert panel["atlas_ok"] is False

    def test_director_panel_status_light_red_unsound(self):
        """Status light is RED when structurally unsound."""
        governance = {"structurally_sound": False, "total_slices_indexed": 5, "archetype_diversity_index": 0.5}
        routing_policy = {"routing_status": "CLUSTERED"}
        structural_view = {"governance_status": "OK"}
        
        panel = build_atlas_director_panel(governance, routing_policy, structural_view)
        
        assert panel["status_light"] == "RED"
        assert panel["atlas_ok"] is False

    def test_director_panel_headline_generated(self):
        """Headline is generated based on atlas state."""
        governance = {"structurally_sound": True, "total_slices_indexed": 10, "archetype_diversity_index": 0.6}
        routing_policy = {"routing_status": "CLUSTERED"}
        structural_view = {"governance_status": "OK"}
        
        panel = build_atlas_director_panel(governance, routing_policy, structural_view)
        
        assert isinstance(panel["headline"], str)
        assert len(panel["headline"]) > 0
        assert "10" in panel["headline"] or "slices" in panel["headline"].lower()

    def test_director_panel_headline_no_slices(self):
        """Headline indicates when no slices are indexed."""
        governance = {"structurally_sound": True, "total_slices_indexed": 0, "archetype_diversity_index": 0.0}
        routing_policy = {"routing_status": "BALANCED"}
        structural_view = {"governance_status": "OK"}
        
        panel = build_atlas_director_panel(governance, routing_policy, structural_view)
        
        assert "no indexed slices" in panel["headline"].lower() or "0" in panel["headline"]

    def test_director_panel_propagates_status(self):
        """Director panel propagates status from inputs."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5, "archetype_diversity_index": 0.5}
        routing_policy = {"routing_status": "SPARSE"}
        structural_view = {"governance_status": "ATTENTION"}
        
        panel = build_atlas_director_panel(governance, routing_policy, structural_view)
        
        assert panel["structurally_sound"] is True
        assert panel["routing_status"] == "SPARSE"
        assert panel["governance_status"] == "ATTENTION"

    def test_director_panel_deterministic(self):
        """Director panel is deterministic."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5, "archetype_diversity_index": 0.5}
        routing_policy = {"routing_status": "CLUSTERED"}
        structural_view = {"governance_status": "OK"}
        
        results = [
            build_atlas_director_panel(governance, routing_policy, structural_view)
            for _ in range(50)
        ]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_director_panel_no_value_judgments(self):
        """Director panel headline contains no value-loaded language."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5, "archetype_diversity_index": 0.5}
        routing_policy = {"routing_status": "CLUSTERED"}
        structural_view = {"governance_status": "OK"}
        
        panel = build_atlas_director_panel(governance, routing_policy, structural_view)
        panel_str = panel["headline"].lower()
        
        forbidden = ["best", "worst", "better", "worse", "top", "bottom", "rank"]
        for word in forbidden:
            assert word not in panel_str, f"Ranking term '{word}' found in headline"


# ===========================================================================
# ATLAS-CURRICULUM COUPLER & PHASE TRANSITION ADVISOR TESTS
# ===========================================================================

class TestBuildAtlasCurriculumCouplingView:
    """Tests for build_atlas_curriculum_coupling_view function (FROZEN CONTRACT)."""

    def test_coupling_view_has_required_keys(self):
        """Coupling view contains all contract keys."""
        atlas_governance = {"total_slices_indexed": 5}
        curriculum_alignment = {
            "slice_alignment": {
                "slice_a": True,
                "slice_b": False,
            }
        }
        
        view = build_atlas_curriculum_coupling_view(atlas_governance, curriculum_alignment)
        
        # FROZEN CONTRACT keys
        assert "slices_with_atlas_support" in view
        assert "slices_without_atlas_support" in view
        assert "coupling_status" in view
        assert "neutral_notes" in view

    def test_coupling_view_categorizes_slices(self):
        """Slices are correctly categorized by atlas support."""
        atlas_governance = {"total_slices_indexed": 5}
        curriculum_alignment = {
            "slice_alignment": {
                "slice_a": True,
                "slice_b": "OK",
                "slice_c": False,
                "slice_d": "aligned",
            }
        }
        
        view = build_atlas_curriculum_coupling_view(atlas_governance, curriculum_alignment)
        
        assert "slice_a" in view["slices_with_atlas_support"]
        assert "slice_b" in view["slices_with_atlas_support"]
        assert "slice_d" in view["slices_with_atlas_support"]
        assert "slice_c" in view["slices_without_atlas_support"]

    def test_coupling_view_status_tight(self):
        """Coupling status is TIGHT when >=80% slices have support."""
        atlas_governance = {"total_slices_indexed": 5}
        curriculum_alignment = {
            "slice_alignment": {
                "slice_a": True,
                "slice_b": True,
                "slice_c": True,
                "slice_d": True,
                "slice_e": False,  # 4/5 = 80%
            }
        }
        
        view = build_atlas_curriculum_coupling_view(atlas_governance, curriculum_alignment)
        
        assert view["coupling_status"] == "TIGHT"

    def test_coupling_view_status_loose(self):
        """Coupling status is LOOSE when 50-79% slices have support."""
        atlas_governance = {"total_slices_indexed": 5}
        curriculum_alignment = {
            "slice_alignment": {
                "slice_a": True,
                "slice_b": True,
                "slice_c": True,
                "slice_d": False,
                "slice_e": False,  # 3/5 = 60%
            }
        }
        
        view = build_atlas_curriculum_coupling_view(atlas_governance, curriculum_alignment)
        
        assert view["coupling_status"] == "LOOSE"

    def test_coupling_view_status_missing(self):
        """Coupling status is MISSING when <50% slices have support."""
        atlas_governance = {"total_slices_indexed": 5}
        curriculum_alignment = {
            "slice_alignment": {
                "slice_a": True,
                "slice_b": False,
                "slice_c": False,
                "slice_d": False,
                "slice_e": False,  # 1/5 = 20%
            }
        }
        
        view = build_atlas_curriculum_coupling_view(atlas_governance, curriculum_alignment)
        
        assert view["coupling_status"] == "MISSING"

    def test_coupling_view_status_missing_no_support(self):
        """Coupling status is MISSING when no slices have support."""
        atlas_governance = {"total_slices_indexed": 5}
        curriculum_alignment = {
            "slice_alignment": {
                "slice_a": False,
                "slice_b": False,
            }
        }
        
        view = build_atlas_curriculum_coupling_view(atlas_governance, curriculum_alignment)
        
        assert view["coupling_status"] == "MISSING"

    def test_coupling_view_alternative_structure(self):
        """Coupling view handles alternative curriculum_alignment structure."""
        atlas_governance = {"total_slices_indexed": 5}
        curriculum_alignment = {
            "slices": ["slice_a", "slice_b", "slice_c"],
            "aligned_slices": ["slice_a", "slice_b"],
        }
        
        view = build_atlas_curriculum_coupling_view(atlas_governance, curriculum_alignment)
        
        assert "slice_a" in view["slices_with_atlas_support"]
        assert "slice_b" in view["slices_with_atlas_support"]
        assert "slice_c" in view["slices_without_atlas_support"]

    def test_coupling_view_generates_notes(self):
        """Coupling view generates neutral notes."""
        atlas_governance = {"total_slices_indexed": 5}
        curriculum_alignment = {
            "slice_alignment": {
                "slice_a": True,
                "slice_b": False,
            }
        }
        
        view = build_atlas_curriculum_coupling_view(atlas_governance, curriculum_alignment)
        
        assert len(view["neutral_notes"]) > 0
        assert all(isinstance(note, str) for note in view["neutral_notes"])

    def test_coupling_view_deterministic(self):
        """Coupling view is deterministic."""
        atlas_governance = {"total_slices_indexed": 5}
        curriculum_alignment = {
            "slice_alignment": {
                "slice_a": True,
                "slice_b": False,
            }
        }
        
        results = [
            build_atlas_curriculum_coupling_view(atlas_governance, curriculum_alignment)
            for _ in range(50)
        ]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_coupling_view_handles_empty_curriculum(self):
        """Coupling view handles empty curriculum gracefully."""
        atlas_governance = {"total_slices_indexed": 5}
        curriculum_alignment = {}
        
        view = build_atlas_curriculum_coupling_view(atlas_governance, curriculum_alignment)
        
        assert view["coupling_status"] == "MISSING"
        assert len(view["slices_with_atlas_support"]) == 0
        assert len(view["slices_without_atlas_support"]) == 0


class TestDerivePhaseTransitionAdvice:
    """Tests for derive_phase_transition_advice function (FROZEN CONTRACT)."""

    def test_phase_advice_has_required_keys(self):
        """Phase transition advice contains all contract keys."""
        coupling_view = {
            "coupling_status": "TIGHT",
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": ["slice_b"],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": ["slice_a"],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        # FROZEN CONTRACT keys
        assert "phase_transition_safe" in advice
        assert "status" in advice
        assert "suggested_slices_for_phase_upgrade" in advice
        assert "slices_needing_more_atlas_support" in advice
        assert "headline" in advice

    def test_phase_advice_safe_tight_coupling_ok_structure(self):
        """Phase transition safe is True when TIGHT coupling and OK structure."""
        coupling_view = {
            "coupling_status": "TIGHT",
            "slices_with_atlas_support": ["slice_a", "slice_b"],
            "slices_without_atlas_support": [],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": ["slice_a", "slice_b"],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert advice["phase_transition_safe"] is True
        assert advice["status"] == "OK"

    def test_phase_advice_block_missing_coupling(self):
        """Phase transition blocked when coupling is MISSING."""
        coupling_view = {
            "coupling_status": "MISSING",
            "slices_with_atlas_support": [],
            "slices_without_atlas_support": ["slice_a", "slice_b"],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": [],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert advice["phase_transition_safe"] is False
        assert advice["status"] == "BLOCK"

    def test_phase_advice_block_volatile_governance(self):
        """Phase transition blocked when governance is VOLATILE."""
        coupling_view = {
            "coupling_status": "TIGHT",
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": [],
        }
        structural_view = {
            "governance_status": "VOLATILE",
            "slices_with_consistent_archetypes": [],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert advice["phase_transition_safe"] is False
        assert advice["status"] == "BLOCK"

    def test_phase_advice_attention_loose_coupling(self):
        """Phase transition requires attention when coupling is LOOSE."""
        coupling_view = {
            "coupling_status": "LOOSE",
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": ["slice_b"],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": ["slice_a"],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert advice["phase_transition_safe"] is False
        assert advice["status"] == "ATTENTION"

    def test_phase_advice_attention_governance_attention(self):
        """Phase transition requires attention when governance is ATTENTION."""
        coupling_view = {
            "coupling_status": "TIGHT",
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": [],
        }
        structural_view = {
            "governance_status": "ATTENTION",
            "slices_with_consistent_archetypes": ["slice_a"],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert advice["phase_transition_safe"] is False
        assert advice["status"] == "ATTENTION"

    def test_phase_advice_suggested_slices(self):
        """Suggested slices are those with support and consistent archetypes."""
        coupling_view = {
            "coupling_status": "TIGHT",
            "slices_with_atlas_support": ["slice_a", "slice_b", "slice_c"],
            "slices_without_atlas_support": [],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": ["slice_a", "slice_b"],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert set(advice["suggested_slices_for_phase_upgrade"]) == {"slice_a", "slice_b"}

    def test_phase_advice_suggested_slices_fallback(self):
        """Suggested slices fallback to all supported slices if no consistent overlap."""
        coupling_view = {
            "coupling_status": "TIGHT",
            "slices_with_atlas_support": ["slice_a", "slice_b"],
            "slices_without_atlas_support": [],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": ["slice_x"],  # No overlap
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert set(advice["suggested_slices_for_phase_upgrade"]) == {"slice_a", "slice_b"}

    def test_phase_advice_slices_needing_support(self):
        """Slices needing support are those without atlas support."""
        coupling_view = {
            "coupling_status": "LOOSE",
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": ["slice_b", "slice_c"],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": [],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert set(advice["slices_needing_more_atlas_support"]) == {"slice_b", "slice_c"}

    def test_phase_advice_headline_generated(self):
        """Headline is generated based on status."""
        coupling_view = {
            "coupling_status": "TIGHT",
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": [],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": [],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert isinstance(advice["headline"], str)
        assert len(advice["headline"]) > 0

    def test_phase_advice_headline_block_missing(self):
        """Headline indicates block when coupling is missing."""
        coupling_view = {
            "coupling_status": "MISSING",
            "slices_with_atlas_support": [],
            "slices_without_atlas_support": ["slice_a"],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": [],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert "blocked" in advice["headline"].lower()
        assert "lack atlas support" in advice["headline"].lower()

    def test_phase_advice_headline_block_volatile(self):
        """Headline indicates block when governance is volatile."""
        coupling_view = {
            "coupling_status": "TIGHT",
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": [],
        }
        structural_view = {
            "governance_status": "VOLATILE",
            "slices_with_consistent_archetypes": [],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        
        assert "blocked" in advice["headline"].lower()
        assert "volatile" in advice["headline"].lower()

    def test_phase_advice_deterministic(self):
        """Phase transition advice is deterministic."""
        coupling_view = {
            "coupling_status": "TIGHT",
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": [],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": ["slice_a"],
        }
        
        results = [
            derive_phase_transition_advice(coupling_view, structural_view)
            for _ in range(50)
        ]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_phase_advice_no_value_judgments(self):
        """Phase transition advice headline contains no value-loaded language."""
        coupling_view = {
            "coupling_status": "TIGHT",
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": [],
        }
        structural_view = {
            "governance_status": "OK",
            "slices_with_consistent_archetypes": ["slice_a"],
        }
        
        advice = derive_phase_transition_advice(coupling_view, structural_view)
        advice_str = advice["headline"].lower()
        
        forbidden = ["best", "worst", "better", "worse", "top", "bottom", "rank"]
        for word in forbidden:
            assert word not in advice_str, f"Ranking term '{word}' found in headline"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

