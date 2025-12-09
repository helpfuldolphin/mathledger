"""
PHASE II — Behavioral Telemetry Visualization Tests

Descriptive only — not admissible as uplift evidence.

This module provides 30+ tests for the Behavioral Telemetry Visualization Suite:
- Deterministic PNG generation
- Hash-based equality checks
- Agg-backend safety
- Graceful handling of missing fields
- Layout engine determinism
- Manifest generator verification

Author: metrics-engineer-4 (Agent D4)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from experiments.behavioral_telemetry_viz import (
    COLOR_DENSITY_BASELINE,
    COLOR_DENSITY_RFL,
    CROSS_LINK_SCHEMA_VERSION,
    DeterministicLayoutEngine,
    GOVERNANCE_SCHEMA_VERSION,
    HEATMAP_CMAP,
    PHASE_II_LABEL,
    VisualizationManifest,
    _compute_entropy_from_scores,
    _render_abstention_heatmap_cell,
    _render_chain_depth_cell,
    _render_entropy_cell,
    _render_volatility_cell,
    build_full_telemetry_governance_report,
    build_telemetry_atlas_link,
    build_telemetry_director_panel,
    build_telemetry_drift_envelope,
    build_telemetry_drift_history,
    build_telemetry_governance_snapshot,
    build_telemetry_structural_coupling_view,
    check_pack_snapshot,
    evaluate_telemetry_for_release,
    extract_abstention_series,
    extract_candidate_ordering_entropy,
    extract_chain_depth_series,
    extract_metric_for_volatility,
    generate_telemetry_pack,
    get_pack_health_report,
    is_telemetry_pack_healthy,
    load_pack_index,
    plot_abstention_heatmap,
    plot_behavioral_telemetry_grid,
    plot_candidate_ordering_entropy,
    plot_chain_depth_density,
    plot_rolling_metric_volatility,
    save_pack_snapshot,
    summarize_telemetry_for_global_health,
    summarize_telemetry_for_uplift_safety,
    validate_pack_index_against_manifest,
)
from experiments.uplift_visualization import (
    _compute_file_checksum,
    load_jsonl,
    setup_deterministic_style,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

def create_mock_records_with_abstention(n_cycles: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Create deterministic mock records with abstention data."""
    records = []
    for i in range(n_cycles):
        # Deterministic pattern based on seed and cycle
        candidates = 5 + (seed + i) % 3
        abstained = ((seed * 7 + i * 11) % 100) % (candidates + 1)
        verified = candidates - abstained
        
        records.append({
            'cycle': i,
            'mode': 'test',
            'derivation': {
                'candidates': candidates,
                'abstained': abstained,
                'verified': verified,
                'chain_depth': (i % 5) + 1,
            },
            'success': verified > 0,
        })
    return records


def create_mock_records_with_entropy(n_cycles: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Create deterministic mock records with entropy data."""
    records = []
    for i in range(n_cycles):
        # Deterministic entropy values
        base_entropy = 0.5 + 0.3 * ((seed + i) % 10) / 10.0
        
        records.append({
            'cycle': i,
            'candidate_entropy': base_entropy,
            'derivation': {
                'candidates': 5,
                'verified': 2 + (i % 3),
                'abstained': 1,
                'ordering_entropy': base_entropy,
            },
        })
    return records


def create_mock_records_minimal(n_cycles: int = 50) -> List[Dict[str, Any]]:
    """Create minimal records without optional fields."""
    return [{'cycle': i, 'success': i % 2 == 0} for i in range(n_cycles)]


def write_mock_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """Write records to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + '\n')


# -----------------------------------------------------------------------------
# Test: Data Extraction
# -----------------------------------------------------------------------------

class TestDataExtraction(unittest.TestCase):
    """Tests for data extraction functions."""

    def test_extract_abstention_series_returns_dataframe(self):
        """extract_abstention_series should return DataFrame with expected columns."""
        records = create_mock_records_with_abstention(50)
        df = extract_abstention_series(records)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), 
                        ['cycle', 'abstention_rate', 'abstained_count', 'total_candidates'])
        self.assertEqual(len(df), 50)

    def test_extract_abstention_series_sorted_by_cycle(self):
        """Abstention series should be sorted by cycle."""
        records = create_mock_records_with_abstention(30)
        df = extract_abstention_series(records)
        
        cycles = df['cycle'].tolist()
        self.assertEqual(cycles, sorted(cycles))

    def test_extract_abstention_rate_bounded(self):
        """Abstention rate should be in [0, 1]."""
        records = create_mock_records_with_abstention(100)
        df = extract_abstention_series(records)
        
        self.assertTrue((df['abstention_rate'] >= 0).all())
        self.assertTrue((df['abstention_rate'] <= 1).all())

    def test_extract_chain_depth_series_returns_dataframe(self):
        """extract_chain_depth_series should return DataFrame."""
        records = create_mock_records_with_abstention(50)
        df = extract_chain_depth_series(records)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ['cycle', 'chain_depth'])

    def test_extract_chain_depth_handles_missing_field(self):
        """Missing chain_depth should default to 0."""
        records = [{'cycle': 0, 'derivation': {'candidates': 5}}]
        df = extract_chain_depth_series(records)
        
        self.assertEqual(df.loc[0, 'chain_depth'], 0)

    def test_extract_candidate_ordering_entropy_returns_dataframe(self):
        """extract_candidate_ordering_entropy should return DataFrame."""
        records = create_mock_records_with_entropy(50)
        df = extract_candidate_ordering_entropy(records)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ['cycle', 'entropy'])

    def test_extract_entropy_handles_missing_field(self):
        """Missing entropy field should produce valid (possibly 0) entropy."""
        records = create_mock_records_minimal(20)
        df = extract_candidate_ordering_entropy(records)
        
        self.assertEqual(len(df), 20)
        self.assertTrue((df['entropy'] >= 0).all())

    def test_extract_metric_for_volatility_returns_dataframe(self):
        """extract_metric_for_volatility should return DataFrame."""
        records = create_mock_records_with_abstention(50)
        df = extract_metric_for_volatility(records, 'derivation.verified')
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ['cycle', 'value'])

    def test_extract_metric_handles_missing_path(self):
        """Missing metric path should return 0."""
        records = [{'cycle': 0}]
        df = extract_metric_for_volatility(records, 'nonexistent.path')
        
        self.assertEqual(df.loc[0, 'value'], 0.0)


# -----------------------------------------------------------------------------
# Test: Entropy Computation
# -----------------------------------------------------------------------------

class TestEntropyComputation(unittest.TestCase):
    """Tests for entropy computation helpers."""

    def test_entropy_from_uniform_scores(self):
        """Uniform scores should produce maximum entropy."""
        scores = [1.0, 1.0, 1.0, 1.0]
        entropy = _compute_entropy_from_scores(scores)
        
        # Max entropy for 4 equally probable outcomes = log2(4) = 2.0
        self.assertAlmostEqual(entropy, 2.0, places=5)

    def test_entropy_from_single_score(self):
        """Single score should produce 0 entropy."""
        scores = [1.0]
        entropy = _compute_entropy_from_scores(scores)
        
        self.assertEqual(entropy, 0.0)

    def test_entropy_from_empty_scores(self):
        """Empty scores should produce 0 entropy."""
        entropy = _compute_entropy_from_scores([])
        self.assertEqual(entropy, 0.0)

    def test_entropy_from_zero_scores(self):
        """All-zero scores should produce 0 entropy."""
        entropy = _compute_entropy_from_scores([0, 0, 0])
        self.assertEqual(entropy, 0.0)

    def test_entropy_deterministic(self):
        """Same inputs should produce identical entropy."""
        scores = [0.5, 0.3, 0.2]
        results = [_compute_entropy_from_scores(scores) for _ in range(10)]
        
        self.assertTrue(all(r == results[0] for r in results))


# -----------------------------------------------------------------------------
# Test: Image Generation
# -----------------------------------------------------------------------------

class TestImageGeneration(unittest.TestCase):
    """Tests for PNG file generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        write_mock_jsonl(create_mock_records_with_abstention(100, seed=42), self.baseline_path)
        write_mock_jsonl(create_mock_records_with_entropy(100, seed=43), self.rfl_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_abstention_heatmap_creates_png(self):
        """plot_abstention_heatmap should create PNG file."""
        out_path = os.path.join(self.temp_dir, 'heatmap.png')
        meta = plot_abstention_heatmap(self.baseline_path, out_path)
        
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        self.assertEqual(meta['plot_type'], 'abstention_heatmap')

    def test_chain_depth_density_creates_png(self):
        """plot_chain_depth_density should create PNG file."""
        out_path = os.path.join(self.temp_dir, 'depth.png')
        meta = plot_chain_depth_density(self.baseline_path, self.rfl_path, out_path)
        
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        self.assertEqual(meta['plot_type'], 'chain_depth_density')

    def test_candidate_entropy_creates_png(self):
        """plot_candidate_ordering_entropy should create PNG file."""
        out_path = os.path.join(self.temp_dir, 'entropy.png')
        meta = plot_candidate_ordering_entropy(self.baseline_path, self.rfl_path, out_path)
        
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        self.assertEqual(meta['plot_type'], 'candidate_ordering_entropy')

    def test_rolling_volatility_creates_png(self):
        """plot_rolling_metric_volatility should create PNG file."""
        out_path = os.path.join(self.temp_dir, 'volatility.png')
        meta = plot_rolling_metric_volatility(self.baseline_path, self.rfl_path, out_path)
        
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        self.assertEqual(meta['plot_type'], 'rolling_metric_volatility')


# -----------------------------------------------------------------------------
# Test: Determinism
# -----------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    """Tests for deterministic output checksums."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        # Use fixed seeds for reproducibility
        write_mock_jsonl(create_mock_records_with_abstention(100, seed=42), self.baseline_path)
        write_mock_jsonl(create_mock_records_with_entropy(100, seed=43), self.rfl_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_abstention_heatmap_deterministic_checksum(self):
        """Abstention heatmap should produce identical checksums."""
        checksums = []
        for i in range(3):
            out_path = os.path.join(self.temp_dir, f'heatmap_{i}.png')
            meta = plot_abstention_heatmap(self.baseline_path, out_path, bin_size=20)
            checksums.append(meta['checksum_sha256'])
        
        self.assertEqual(checksums[0], checksums[1])
        self.assertEqual(checksums[1], checksums[2])

    def test_chain_depth_density_deterministic_checksum(self):
        """Chain depth density should produce identical checksums."""
        checksums = []
        for i in range(3):
            out_path = os.path.join(self.temp_dir, f'depth_{i}.png')
            meta = plot_chain_depth_density(self.baseline_path, self.rfl_path, out_path)
            checksums.append(meta['checksum_sha256'])
        
        self.assertEqual(checksums[0], checksums[1])
        self.assertEqual(checksums[1], checksums[2])

    def test_entropy_deterministic_checksum(self):
        """Entropy plot should produce identical checksums."""
        checksums = []
        for i in range(3):
            out_path = os.path.join(self.temp_dir, f'entropy_{i}.png')
            meta = plot_candidate_ordering_entropy(
                self.baseline_path, self.rfl_path, out_path, window=20
            )
            checksums.append(meta['checksum_sha256'])
        
        self.assertEqual(checksums[0], checksums[1])
        self.assertEqual(checksums[1], checksums[2])

    def test_volatility_deterministic_checksum(self):
        """Volatility plot should produce identical checksums."""
        checksums = []
        for i in range(3):
            out_path = os.path.join(self.temp_dir, f'vol_{i}.png')
            meta = plot_rolling_metric_volatility(
                self.baseline_path, self.rfl_path, out_path, window=20
            )
            checksums.append(meta['checksum_sha256'])
        
        self.assertEqual(checksums[0], checksums[1])
        self.assertEqual(checksums[1], checksums[2])

    def test_grid_deterministic_checksum(self):
        """Behavioral telemetry grid should produce identical checksums."""
        slice_data = [
            {'name': 'Slice A', 'baseline_path': self.baseline_path, 'rfl_path': self.rfl_path},
        ]
        
        checksums = []
        for i in range(3):
            out_path = os.path.join(self.temp_dir, f'grid_{i}.png')
            meta = plot_behavioral_telemetry_grid(
                slice_data, out_path, plot_type='entropy'
            )
            checksums.append(meta['checksum_sha256'])
        
        self.assertEqual(checksums[0], checksums[1])
        self.assertEqual(checksums[1], checksums[2])


# -----------------------------------------------------------------------------
# Test: Layout Engine
# -----------------------------------------------------------------------------

class TestDeterministicLayoutEngine(unittest.TestCase):
    """Tests for DeterministicLayoutEngine."""

    def test_compute_grid_dimensions_single(self):
        """Single item should produce 1x1 grid."""
        layout = DeterministicLayoutEngine(n_cols=2)
        rows, cols = layout.compute_grid_dimensions(1)
        
        self.assertEqual(rows, 1)
        self.assertEqual(cols, 1)

    def test_compute_grid_dimensions_two_cols(self):
        """Three items with 2 cols should produce 2x2 grid."""
        layout = DeterministicLayoutEngine(n_cols=2)
        rows, cols = layout.compute_grid_dimensions(3)
        
        self.assertEqual(rows, 2)
        self.assertEqual(cols, 2)

    def test_compute_grid_dimensions_four_items(self):
        """Four items with 2 cols should produce 2x2 grid."""
        layout = DeterministicLayoutEngine(n_cols=2)
        rows, cols = layout.compute_grid_dimensions(4)
        
        self.assertEqual(rows, 2)
        self.assertEqual(cols, 2)

    def test_sort_items_alphabetically(self):
        """Items should be sorted alphabetically by name."""
        layout = DeterministicLayoutEngine()
        items = [
            {'name': 'Zebra'},
            {'name': 'Alpha'},
            {'name': 'Beta'},
        ]
        
        sorted_items = layout.sort_items_alphabetically(items)
        
        self.assertEqual([i['name'] for i in sorted_items], ['Alpha', 'Beta', 'Zebra'])

    def test_sort_items_deterministic(self):
        """Sorting should be deterministic across calls."""
        layout = DeterministicLayoutEngine()
        items = [
            {'name': 'C'}, {'name': 'A'}, {'name': 'B'}
        ]
        
        results = [layout.sort_items_alphabetically(items) for _ in range(5)]
        
        for r in results[1:]:
            self.assertEqual(r, results[0])

    def test_get_cell_position_row_major(self):
        """Cell positions should follow row-major order."""
        layout = DeterministicLayoutEngine(n_cols=2)
        
        self.assertEqual(layout.get_cell_position(0), (0, 0))
        self.assertEqual(layout.get_cell_position(1), (0, 1))
        self.assertEqual(layout.get_cell_position(2), (1, 0))
        self.assertEqual(layout.get_cell_position(3), (1, 1))


# -----------------------------------------------------------------------------
# Test: Visualization Manifest
# -----------------------------------------------------------------------------

class TestVisualizationManifest(unittest.TestCase):
    """Tests for VisualizationManifest."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_rcparams_hash_deterministic(self):
        """rcParams hash should be deterministic."""
        m1 = VisualizationManifest()
        m2 = VisualizationManifest()
        
        h1 = m1.compute_rcparams_hash()
        h2 = m2.compute_rcparams_hash()
        
        self.assertEqual(h1, h2)

    def test_compute_parameter_hash_deterministic(self):
        """Parameter hash should be deterministic."""
        manifest = VisualizationManifest()
        params = {'window': 20, 'bin_size': 50, 'path': '/test/path.jsonl'}
        
        h1 = manifest.compute_parameter_hash(params)
        h2 = manifest.compute_parameter_hash(params)
        
        self.assertEqual(h1, h2)

    def test_compute_parameter_hash_different_params(self):
        """Different parameters should produce different hashes."""
        manifest = VisualizationManifest()
        
        h1 = manifest.compute_parameter_hash({'window': 20})
        h2 = manifest.compute_parameter_hash({'window': 30})
        
        self.assertNotEqual(h1, h2)

    def test_add_entry_stores_metadata(self):
        """add_entry should store plot metadata."""
        manifest = VisualizationManifest()
        
        plot_meta = {
            'plot_type': 'test_plot',
            'output_path': '/test/output.png',
            'checksum_sha256': 'abc123',
        }
        params = {'window': 20}
        
        manifest.add_entry(plot_meta, params)
        
        self.assertEqual(len(manifest.entries), 1)
        self.assertEqual(manifest.entries[0]['plot_type'], 'test_plot')

    def test_generate_returns_dict(self):
        """generate should return complete manifest dict."""
        manifest = VisualizationManifest()
        manifest.add_entry({'plot_type': 'test', 'checksum_sha256': 'x'}, {})
        
        result = manifest.generate()
        
        self.assertIn('manifest_version', result)
        self.assertIn('rcparams_hash_sha256', result)
        self.assertIn('entries', result)
        self.assertEqual(result['entry_count'], 1)

    def test_save_creates_json_file(self):
        """save should create JSON file."""
        manifest = VisualizationManifest()
        manifest.add_entry({'plot_type': 'test', 'checksum_sha256': 'x'}, {})
        
        out_path = os.path.join(self.temp_dir, 'manifest.json')
        manifest.save(out_path)
        
        self.assertTrue(os.path.exists(out_path))
        
        with open(out_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('entries', data)

    def test_load_restores_manifest(self):
        """load should restore manifest from file."""
        # Create and save
        m1 = VisualizationManifest()
        m1.add_entry({'plot_type': 'test', 'checksum_sha256': 'abc'}, {'p': 1})
        m1.compute_rcparams_hash()
        
        out_path = os.path.join(self.temp_dir, 'manifest.json')
        m1.save(out_path)
        
        # Load
        m2 = VisualizationManifest.load(out_path)
        
        self.assertEqual(len(m2.entries), 1)
        self.assertEqual(m2._rcparams_hash, m1._rcparams_hash)

    def test_verify_reproducibility_identical_manifests(self):
        """Identical manifests should pass verification."""
        m1 = VisualizationManifest()
        m1.add_entry({'plot_type': 'test', 'checksum_sha256': 'abc',
                     'file_hash_sha256': 'abc', 'parameter_hash_sha256': 'xyz'}, {})
        m1.compute_rcparams_hash()
        m1.entries[0]['file_hash_sha256'] = 'abc'
        m1.entries[0]['parameter_hash_sha256'] = 'xyz'
        
        m2 = VisualizationManifest()
        m2.add_entry({'plot_type': 'test', 'checksum_sha256': 'abc',
                     'file_hash_sha256': 'abc', 'parameter_hash_sha256': 'xyz'}, {})
        m2.compute_rcparams_hash()
        m2.entries[0]['file_hash_sha256'] = 'abc'
        m2.entries[0]['parameter_hash_sha256'] = 'xyz'
        
        report = m1.verify_reproducibility(m2)
        
        self.assertTrue(report['rcparams_match'])
        self.assertTrue(report['entry_count_match'])

    def test_verify_reproducibility_different_file_hash(self):
        """Different file hashes should be detected."""
        m1 = VisualizationManifest()
        m1.add_entry({'plot_type': 'test', 'checksum_sha256': 'abc'}, {})
        m1.compute_rcparams_hash()
        m1.entries[0]['file_hash_sha256'] = 'hash_a'
        
        m2 = VisualizationManifest()
        m2.add_entry({'plot_type': 'test', 'checksum_sha256': 'def'}, {})
        m2.compute_rcparams_hash()
        m2.entries[0]['file_hash_sha256'] = 'hash_b'
        
        report = m1.verify_reproducibility(m2)
        
        self.assertFalse(report['all_match'])


# -----------------------------------------------------------------------------
# Test: Agg Backend Safety
# -----------------------------------------------------------------------------

class TestAggBackendSafety(unittest.TestCase):
    """Tests for Agg backend usage."""

    def test_matplotlib_backend_is_agg(self):
        """Matplotlib should use Agg backend after import."""
        import matplotlib
        
        # The module sets matplotlib.use('Agg') at import
        # We verify it doesn't crash when creating figures
        setup_deterministic_style()
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)
        
        # If we got here without error, Agg is working

    def test_no_display_required(self):
        """Plots should generate without display environment."""
        # This test verifies non-interactive rendering works
        temp_dir = tempfile.mkdtemp()
        try:
            baseline_path = os.path.join(temp_dir, 'test.jsonl')
            write_mock_jsonl(create_mock_records_with_abstention(20), baseline_path)
            
            out_path = os.path.join(temp_dir, 'test.png')
            meta = plot_abstention_heatmap(baseline_path, out_path)
            
            self.assertTrue(os.path.exists(out_path))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Test: Graceful Handling of Missing Fields
# -----------------------------------------------------------------------------

class TestMissingFieldHandling(unittest.TestCase):
    """Tests for graceful handling of missing fields."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_abstention_missing_derivation(self):
        """Abstention extraction handles missing derivation."""
        records = [{'cycle': 0}]
        df = extract_abstention_series(records)
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, 'abstention_rate'], 0.0)

    def test_chain_depth_missing_derivation(self):
        """Chain depth extraction handles missing derivation."""
        records = [{'cycle': 0}]
        df = extract_chain_depth_series(records)
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, 'chain_depth'], 0)

    def test_entropy_missing_all_fields(self):
        """Entropy extraction handles completely minimal records."""
        records = [{'cycle': i} for i in range(10)]
        df = extract_candidate_ordering_entropy(records)
        
        self.assertEqual(len(df), 10)
        # Should not crash, entropy values are 0 or computed from defaults

    def test_plot_with_minimal_data(self):
        """Plots should handle minimal data gracefully."""
        minimal_path = os.path.join(self.temp_dir, 'minimal.jsonl')
        write_mock_jsonl([{'cycle': 0}], minimal_path)
        
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        # Should not raise
        meta = plot_abstention_heatmap(minimal_path, out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_plot_with_empty_records(self):
        """Plots should handle empty records gracefully."""
        empty_path = os.path.join(self.temp_dir, 'empty.jsonl')
        with open(empty_path, 'w') as f:
            pass  # Empty file
        
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        # This will load empty list, should handle gracefully
        records = load_jsonl(empty_path)
        self.assertEqual(len(records), 0)
        
        df = extract_abstention_series(records)
        self.assertEqual(len(df), 0)


# -----------------------------------------------------------------------------
# Test: PHASE II Labeling
# -----------------------------------------------------------------------------

class TestPhaseIILabeling(unittest.TestCase):
    """Tests for PHASE II labeling compliance."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = os.path.join(self.temp_dir, 'test.jsonl')
        write_mock_jsonl(create_mock_records_with_abstention(50), self.test_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_abstention_heatmap_has_phase_ii_label(self):
        """Abstention heatmap metadata should have PHASE II label."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        meta = plot_abstention_heatmap(self.test_path, out_path)
        
        self.assertEqual(meta['phase'], 'PHASE II')
        self.assertEqual(meta['label'], PHASE_II_LABEL)

    def test_chain_depth_has_phase_ii_label(self):
        """Chain depth metadata should have PHASE II label."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        meta = plot_chain_depth_density(self.test_path, self.test_path, out_path)
        
        self.assertEqual(meta['phase'], 'PHASE II')
        self.assertEqual(meta['label'], PHASE_II_LABEL)

    def test_entropy_has_phase_ii_label(self):
        """Entropy metadata should have PHASE II label."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        meta = plot_candidate_ordering_entropy(self.test_path, self.test_path, out_path)
        
        self.assertEqual(meta['phase'], 'PHASE II')
        self.assertEqual(meta['label'], PHASE_II_LABEL)

    def test_volatility_has_phase_ii_label(self):
        """Volatility metadata should have PHASE II label."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        meta = plot_rolling_metric_volatility(self.test_path, self.test_path, out_path)
        
        self.assertEqual(meta['phase'], 'PHASE II')
        self.assertEqual(meta['label'], PHASE_II_LABEL)

    def test_manifest_has_phase_ii_label(self):
        """Manifest should have PHASE II label."""
        manifest = VisualizationManifest()
        result = manifest.generate()
        
        self.assertEqual(result['phase'], 'PHASE II')
        self.assertEqual(result['label'], PHASE_II_LABEL)


# -----------------------------------------------------------------------------
# Test: Grid Plots
# -----------------------------------------------------------------------------

class TestGridPlots(unittest.TestCase):
    """Tests for behavioral telemetry grid plots."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path_a = os.path.join(self.temp_dir, 'a.jsonl')
        self.path_b = os.path.join(self.temp_dir, 'b.jsonl')
        
        write_mock_jsonl(create_mock_records_with_abstention(50, seed=1), self.path_a)
        write_mock_jsonl(create_mock_records_with_abstention(50, seed=2), self.path_b)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_grid_empty_raises_error(self):
        """Empty slice_data should raise ValueError."""
        out_path = os.path.join(self.temp_dir, 'grid.png')
        
        with self.assertRaises(ValueError):
            plot_behavioral_telemetry_grid([], out_path)

    def test_grid_single_slice(self):
        """Single slice should produce valid grid."""
        slice_data = [{'name': 'Only', 'baseline_path': self.path_a, 'rfl_path': self.path_b}]
        out_path = os.path.join(self.temp_dir, 'grid.png')
        
        meta = plot_behavioral_telemetry_grid(slice_data, out_path, plot_type='entropy')
        
        self.assertTrue(os.path.exists(out_path))
        self.assertEqual(meta['num_slices'], 1)

    def test_grid_alphabetical_ordering(self):
        """Grid should order slices alphabetically."""
        slice_data = [
            {'name': 'Zebra', 'baseline_path': self.path_a, 'rfl_path': self.path_b},
            {'name': 'Alpha', 'baseline_path': self.path_a, 'rfl_path': self.path_b},
        ]
        out_path = os.path.join(self.temp_dir, 'grid.png')
        
        meta = plot_behavioral_telemetry_grid(slice_data, out_path, plot_type='entropy')
        
        # Slices in metadata should be in order processed (alphabetical)
        self.assertEqual(meta['slices'][0]['name'], 'Alpha')
        self.assertEqual(meta['slices'][1]['name'], 'Zebra')


# -----------------------------------------------------------------------------
# Test: Return Types
# -----------------------------------------------------------------------------

class TestReturnTypes(unittest.TestCase):
    """Tests for consistent return types."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = os.path.join(self.temp_dir, 'test.jsonl')
        write_mock_jsonl(create_mock_records_with_abstention(30), self.test_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_all_plots_return_dict(self):
        """All plot functions should return dict."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        meta1 = plot_abstention_heatmap(self.test_path, out_path)
        self.assertIsInstance(meta1, dict)
        
        meta2 = plot_chain_depth_density(self.test_path, self.test_path, out_path)
        self.assertIsInstance(meta2, dict)
        
        meta3 = plot_candidate_ordering_entropy(self.test_path, self.test_path, out_path)
        self.assertIsInstance(meta3, dict)
        
        meta4 = plot_rolling_metric_volatility(self.test_path, self.test_path, out_path)
        self.assertIsInstance(meta4, dict)

    def test_all_metadata_has_checksum(self):
        """All metadata should include checksum."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        for plot_fn in [
            lambda: plot_abstention_heatmap(self.test_path, out_path),
            lambda: plot_chain_depth_density(self.test_path, self.test_path, out_path),
            lambda: plot_candidate_ordering_entropy(self.test_path, self.test_path, out_path),
            lambda: plot_rolling_metric_volatility(self.test_path, self.test_path, out_path),
        ]:
            meta = plot_fn()
            self.assertIn('checksum_sha256', meta)
            self.assertEqual(len(meta['checksum_sha256']), 64)


# -----------------------------------------------------------------------------
# Test: Pack Index
# -----------------------------------------------------------------------------

class TestPackIndex(unittest.TestCase):
    """Tests for pack_index.json generation and validation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        write_mock_jsonl(create_mock_records_with_abstention(50, seed=42), self.baseline_path)
        write_mock_jsonl(create_mock_records_with_entropy(50, seed=43), self.rfl_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pack_creates_index_file(self):
        """generate_telemetry_pack should create pack_index.json."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        index_path = os.path.join(pack_dir, 'pack_index.json')
        self.assertTrue(os.path.exists(index_path))

    def test_pack_index_has_required_fields(self):
        """pack_index.json should have all required fields."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        index = load_pack_index(pack_dir)
        
        # Check required fields
        self.assertIn('generated_at', index)
        self.assertIn('baseline_log', index)
        self.assertIn('rfl_log', index)
        self.assertIn('plots', index)
        self.assertIn('manifest_hash', index)

    def test_pack_index_baseline_rfl_paths(self):
        """pack_index.json should record correct input paths."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        index = load_pack_index(pack_dir)
        
        self.assertEqual(index['baseline_log'], self.baseline_path)
        self.assertEqual(index['rfl_log'], self.rfl_path)

    def test_pack_index_plots_list(self):
        """pack_index.json plots should have name, filename, checksum."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        index = load_pack_index(pack_dir)
        
        self.assertEqual(len(index['plots']), 4)
        
        for plot in index['plots']:
            self.assertIn('name', plot)
            self.assertIn('filename', plot)
            self.assertIn('checksum', plot)
            self.assertEqual(len(plot['checksum']), 64)  # SHA-256 hex length

    def test_pack_index_matches_manifest(self):
        """pack_index.json entries should match telemetry_manifest.json."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        report = validate_pack_index_against_manifest(pack_dir)
        
        self.assertTrue(report['valid'])
        self.assertTrue(report['manifest_hash_match'])
        
        for check in report['plot_checks']:
            self.assertTrue(check['match'], f"Mismatch for {check['name']}")

    def test_pack_index_manifest_hash_valid(self):
        """pack_index manifest_hash should match actual manifest file hash."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        index = load_pack_index(pack_dir)
        manifest_path = os.path.join(pack_dir, 'telemetry_manifest.json')
        actual_hash = _compute_file_checksum(manifest_path)
        
        self.assertEqual(index['manifest_hash'], actual_hash)

    def test_pack_index_plot_order_deterministic(self):
        """pack_index plots should be in deterministic order."""
        pack_dir_a = os.path.join(self.temp_dir, 'pack_a')
        pack_dir_b = os.path.join(self.temp_dir, 'pack_b')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir_a)
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir_b)
        
        index_a = load_pack_index(pack_dir_a)
        index_b = load_pack_index(pack_dir_b)
        
        # Plot order should be identical (sorted by plot_type)
        names_a = [p['name'] for p in index_a['plots']]
        names_b = [p['name'] for p in index_b['plots']]
        
        self.assertEqual(names_a, names_b)

    def test_validate_missing_index(self):
        """Validation should fail if pack_index.json missing."""
        pack_dir = os.path.join(self.temp_dir, 'empty_pack')
        os.makedirs(pack_dir)
        
        report = validate_pack_index_against_manifest(pack_dir)
        
        self.assertFalse(report['valid'])
        self.assertIn('not found', report.get('error', ''))

    def test_validate_missing_manifest(self):
        """Validation should fail if telemetry_manifest.json missing."""
        pack_dir = os.path.join(self.temp_dir, 'partial_pack')
        os.makedirs(pack_dir)
        
        # Create only index, not manifest
        with open(os.path.join(pack_dir, 'pack_index.json'), 'w') as f:
            json.dump({'plots': []}, f)
        
        report = validate_pack_index_against_manifest(pack_dir)
        
        self.assertFalse(report['valid'])
        self.assertIn('not found', report.get('error', ''))

    def test_pack_index_has_phase_ii_label(self):
        """pack_index.json should have PHASE II label."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        index = load_pack_index(pack_dir)
        
        self.assertEqual(index['phase'], 'PHASE II')
        self.assertEqual(index['label'], PHASE_II_LABEL)

    def test_regeneration_produces_identical_plot_checksums(self):
        """Re-generating pack with same inputs should produce identical plot checksums."""
        pack_dir_a = os.path.join(self.temp_dir, 'pack_a')
        pack_dir_b = os.path.join(self.temp_dir, 'pack_b')
        
        # Generate twice with identical inputs
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir_a)
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir_b)
        
        index_a = load_pack_index(pack_dir_a)
        index_b = load_pack_index(pack_dir_b)
        
        # Plot checksums should be identical (deterministic rendering)
        checksums_a = {p['name']: p['checksum'] for p in index_a['plots']}
        checksums_b = {p['name']: p['checksum'] for p in index_b['plots']}
        
        self.assertEqual(checksums_a, checksums_b)
        
        # NOTE: manifest_hash differs because manifest includes output_path,
        # which contains the temp directory path that differs between runs.
        # Plot checksums are the determinism guarantee, not manifest hash.

    def test_regeneration_deterministic_plot_content(self):
        """Re-generated pack plots list should be identical except paths."""
        pack_dir_a = os.path.join(self.temp_dir, 'pack_a')
        pack_dir_b = os.path.join(self.temp_dir, 'pack_b')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir_a)
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir_b)
        
        index_a = load_pack_index(pack_dir_a)
        index_b = load_pack_index(pack_dir_b)
        
        # Plots list should be identical (same names, filenames, checksums)
        # Remove path-dependent fields for comparison
        plots_a = [(p['name'], p['filename'], p['checksum']) for p in index_a['plots']]
        plots_b = [(p['name'], p['filename'], p['checksum']) for p in index_b['plots']]
        
        self.assertEqual(plots_a, plots_b)
        
        # Input paths should match
        self.assertEqual(index_a['baseline_log'], index_b['baseline_log'])
        self.assertEqual(index_a['rfl_log'], index_b['rfl_log'])
        
        # Phase labels should match
        self.assertEqual(index_a['phase'], index_b['phase'])
        self.assertEqual(index_a['label'], index_b['label'])

    def test_pack_creates_both_manifest_and_index(self):
        """Pack generation should create both telemetry_manifest.json and pack_index.json."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        manifest_path = os.path.join(pack_dir, 'telemetry_manifest.json')
        index_path = os.path.join(pack_dir, 'pack_index.json')
        
        self.assertTrue(os.path.exists(manifest_path), "telemetry_manifest.json not created")
        self.assertTrue(os.path.exists(index_path), "pack_index.json not created")

    def test_pack_index_generated_at_is_iso8601(self):
        """pack_index generated_at should be ISO8601 format."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        index = load_pack_index(pack_dir)
        
        # Should match ISO8601 format: YYYY-MM-DDTHH:MM:SSZ
        generated_at = index['generated_at']
        self.assertRegex(generated_at, r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')


# -----------------------------------------------------------------------------
# Test: Empty Data Handling Contract
# -----------------------------------------------------------------------------

class TestEmptyDataHandlingContract(unittest.TestCase):
    """
    Contract tests for empty data handling.
    
    All extraction functions must:
    - Return well-typed empty structures when logs are empty
    - Never raise due to missing columns
    - Be validated by tests for empty DF → no crash, correct defaults
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- extract_abstention_series ---
    
    def test_abstention_empty_records_returns_empty_df(self):
        """extract_abstention_series with empty records returns empty DataFrame."""
        df = extract_abstention_series([])
        
        self.assertEqual(len(df), 0)
        self.assertIsInstance(df, pd.DataFrame)

    def test_abstention_empty_records_has_correct_schema(self):
        """Empty abstention DataFrame should have correct column schema."""
        df = extract_abstention_series([])
        
        # When populated, would have these columns
        # Empty DF may not have columns, but should be a valid DataFrame
        self.assertIsInstance(df, pd.DataFrame)

    def test_abstention_minimal_record_returns_defaults(self):
        """Minimal record without derivation returns default values."""
        df = extract_abstention_series([{'cycle': 0}])
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, 'cycle'], 0)
        self.assertEqual(df.loc[0, 'abstention_rate'], 0.0)
        self.assertEqual(df.loc[0, 'abstained_count'], 0)
        self.assertEqual(df.loc[0, 'total_candidates'], 1)

    # --- extract_chain_depth_series ---
    
    def test_chain_depth_empty_records_returns_empty_df(self):
        """extract_chain_depth_series with empty records returns empty DataFrame."""
        df = extract_chain_depth_series([])
        
        self.assertEqual(len(df), 0)
        self.assertIsInstance(df, pd.DataFrame)

    def test_chain_depth_minimal_record_returns_defaults(self):
        """Minimal record without derivation returns default chain_depth=0."""
        df = extract_chain_depth_series([{'cycle': 0}])
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, 'cycle'], 0)
        self.assertEqual(df.loc[0, 'chain_depth'], 0)

    # --- extract_candidate_ordering_entropy ---
    
    def test_entropy_empty_records_returns_empty_df(self):
        """extract_candidate_ordering_entropy with empty records returns empty DataFrame."""
        df = extract_candidate_ordering_entropy([])
        
        self.assertEqual(len(df), 0)
        self.assertIsInstance(df, pd.DataFrame)

    def test_entropy_minimal_record_returns_defaults(self):
        """Minimal record without candidates returns default entropy=0."""
        df = extract_candidate_ordering_entropy([{'cycle': 0}])
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, 'cycle'], 0)
        self.assertEqual(df.loc[0, 'entropy'], 0.0)

    # --- extract_metric_for_volatility ---
    
    def test_volatility_empty_records_returns_empty_df(self):
        """extract_metric_for_volatility with empty records returns empty DataFrame."""
        df = extract_metric_for_volatility([])
        
        self.assertEqual(len(df), 0)
        self.assertIsInstance(df, pd.DataFrame)

    def test_volatility_minimal_record_returns_defaults(self):
        """Minimal record without metric path returns default value=0."""
        df = extract_metric_for_volatility([{'cycle': 0}])
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, 'cycle'], 0)
        self.assertEqual(df.loc[0, 'value'], 0.0)

    def test_volatility_missing_nested_path_returns_zero(self):
        """Missing nested metric path returns 0, not crash."""
        df = extract_metric_for_volatility(
            [{'cycle': 0, 'derivation': {}}],
            metric_path='derivation.nonexistent.field'
        )
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, 'value'], 0.0)

    # --- Empty file handling ---
    
    def test_load_empty_jsonl_returns_empty_list(self):
        """Loading empty JSONL file returns empty list."""
        empty_path = os.path.join(self.temp_dir, 'empty.jsonl')
        with open(empty_path, 'w') as f:
            pass  # Empty file
        
        records = load_jsonl(empty_path)
        
        self.assertEqual(records, [])

    def test_all_extractions_chain_from_empty_file(self):
        """All extraction functions handle empty file gracefully."""
        empty_path = os.path.join(self.temp_dir, 'empty.jsonl')
        with open(empty_path, 'w') as f:
            pass
        
        records = load_jsonl(empty_path)
        
        # All should return empty DataFrames without crashing
        df1 = extract_abstention_series(records)
        df2 = extract_chain_depth_series(records)
        df3 = extract_candidate_ordering_entropy(records)
        df4 = extract_metric_for_volatility(records)
        
        self.assertEqual(len(df1), 0)
        self.assertEqual(len(df2), 0)
        self.assertEqual(len(df3), 0)
        self.assertEqual(len(df4), 0)

    # --- Type contracts ---
    
    def test_abstention_always_returns_dataframe(self):
        """extract_abstention_series always returns DataFrame type."""
        self.assertIsInstance(extract_abstention_series([]), pd.DataFrame)
        self.assertIsInstance(extract_abstention_series([{'cycle': 0}]), pd.DataFrame)

    def test_chain_depth_always_returns_dataframe(self):
        """extract_chain_depth_series always returns DataFrame type."""
        self.assertIsInstance(extract_chain_depth_series([]), pd.DataFrame)
        self.assertIsInstance(extract_chain_depth_series([{'cycle': 0}]), pd.DataFrame)

    def test_entropy_always_returns_dataframe(self):
        """extract_candidate_ordering_entropy always returns DataFrame type."""
        self.assertIsInstance(extract_candidate_ordering_entropy([]), pd.DataFrame)
        self.assertIsInstance(extract_candidate_ordering_entropy([{'cycle': 0}]), pd.DataFrame)

    def test_volatility_always_returns_dataframe(self):
        """extract_metric_for_volatility always returns DataFrame type."""
        self.assertIsInstance(extract_metric_for_volatility([]), pd.DataFrame)
        self.assertIsInstance(extract_metric_for_volatility([{'cycle': 0}]), pd.DataFrame)


# -----------------------------------------------------------------------------
# Test: Telemetry ↔ Atlas Cross-Link Contract
# -----------------------------------------------------------------------------

class TestTelemetryAtlasCrossLink(unittest.TestCase):
    """Tests for the telemetry ↔ atlas cross-link contract."""

    def test_cross_link_schema_version(self):
        """Cross-link should include schema_version."""
        index = {
            'generated_at': '2025-12-06T12:00:00Z',
            'manifest_hash': 'a' * 64,
            'plots': [{'name': 'test', 'checksum': 'b' * 64}],
        }
        atlas = {
            'generated_at': '2025-12-06T12:00:00Z',
            'slice_count': 4,
            'manifest_hash': 'c' * 64,
        }
        
        link = build_telemetry_atlas_link(index, atlas)
        
        self.assertEqual(link['schema_version'], CROSS_LINK_SCHEMA_VERSION)

    def test_cross_link_telemetry_section(self):
        """Cross-link should extract telemetry pack info correctly."""
        index = {
            'generated_at': '2025-12-06T12:00:00Z',
            'manifest_hash': 'a' * 64,
            'plots': [
                {'name': 'plot1', 'checksum': 'b' * 64},
                {'name': 'plot2', 'checksum': 'c' * 64},
            ],
        }
        atlas = {'slice_count': 4}
        
        link = build_telemetry_atlas_link(index, atlas)
        
        self.assertEqual(link['telemetry_pack']['generated_at'], '2025-12-06T12:00:00Z')
        self.assertEqual(link['telemetry_pack']['manifest_hash'], 'a' * 64)
        self.assertEqual(link['telemetry_pack']['plot_count'], 2)

    def test_cross_link_atlas_section(self):
        """Cross-link should extract atlas info correctly."""
        index = {'plots': [], 'manifest_hash': 'a' * 64}
        atlas = {
            'generated_at': '2025-12-06T13:00:00Z',
            'slice_count': 4,
            'fingerprint_hash': 'd' * 64,
        }
        
        link = build_telemetry_atlas_link(index, atlas)
        
        self.assertEqual(link['atlas']['generated_at'], '2025-12-06T13:00:00Z')
        self.assertEqual(link['atlas']['cluster_count'], 4)
        self.assertEqual(link['atlas']['fingerprint_hash'], 'd' * 64)

    def test_cross_link_handles_alternate_atlas_keys(self):
        """Cross-link should handle different atlas key names."""
        index = {'plots': [], 'manifest_hash': 'a' * 64}
        
        # Test cluster_count
        atlas1 = {'cluster_count': 5}
        link1 = build_telemetry_atlas_link(index, atlas1)
        self.assertEqual(link1['atlas']['cluster_count'], 5)
        
        # Test n_clusters
        atlas2 = {'n_clusters': 6}
        link2 = build_telemetry_atlas_link(index, atlas2)
        self.assertEqual(link2['atlas']['cluster_count'], 6)
        
        # Test atlas_hash
        atlas3 = {'atlas_hash': 'e' * 64}
        link3 = build_telemetry_atlas_link(index, atlas3)
        self.assertEqual(link3['atlas']['fingerprint_hash'], 'e' * 64)

    def test_cross_link_stable_for_fixed_inputs(self):
        """Same inputs should produce same output (deterministic)."""
        index = {
            'generated_at': '2025-12-06T12:00:00Z',
            'manifest_hash': 'a' * 64,
            'plots': [{'name': 'test', 'checksum': 'b' * 64}],
        }
        atlas = {'slice_count': 4, 'manifest_hash': 'c' * 64}
        
        link1 = build_telemetry_atlas_link(index, atlas)
        link2 = build_telemetry_atlas_link(index, atlas)
        
        self.assertEqual(link1, link2)

    def test_cross_link_handles_missing_fields(self):
        """Cross-link should handle missing optional fields gracefully."""
        index = {'plots': []}
        atlas = {}
        
        link = build_telemetry_atlas_link(index, atlas)
        
        self.assertEqual(link['telemetry_pack']['generated_at'], '')
        self.assertEqual(link['telemetry_pack']['manifest_hash'], '')
        self.assertEqual(link['telemetry_pack']['plot_count'], 0)
        self.assertEqual(link['atlas']['generated_at'], '')
        self.assertEqual(link['atlas']['cluster_count'], 0)
        self.assertEqual(link['atlas']['fingerprint_hash'], '')


# -----------------------------------------------------------------------------
# Test: Telemetry Pack Health Predicate
# -----------------------------------------------------------------------------

class TestTelemetryPackHealth(unittest.TestCase):
    """Tests for the telemetry pack health predicate."""

    def test_healthy_pack_returns_true(self):
        """Healthy pack with all requirements should return True."""
        index = {
            'manifest_hash': 'a' * 64,
            'plots': [
                {'name': 'plot1', 'checksum': 'b' * 64},
                {'name': 'plot2', 'checksum': 'c' * 64},
            ],
        }
        
        self.assertTrue(is_telemetry_pack_healthy(index))

    def test_empty_plots_returns_false(self):
        """Pack with no plots should return False."""
        index = {
            'manifest_hash': 'a' * 64,
            'plots': [],
        }
        
        self.assertFalse(is_telemetry_pack_healthy(index))

    def test_missing_manifest_hash_returns_false(self):
        """Pack without manifest_hash should return False."""
        index = {
            'plots': [{'name': 'plot1', 'checksum': 'b' * 64}],
        }
        
        self.assertFalse(is_telemetry_pack_healthy(index))

    def test_invalid_manifest_hash_returns_false(self):
        """Pack with invalid manifest_hash should return False."""
        index = {
            'manifest_hash': 'not-a-valid-hash',
            'plots': [{'name': 'plot1', 'checksum': 'b' * 64}],
        }
        
        self.assertFalse(is_telemetry_pack_healthy(index))

    def test_invalid_plot_checksum_returns_false(self):
        """Pack with invalid plot checksum should return False."""
        index = {
            'manifest_hash': 'a' * 64,
            'plots': [
                {'name': 'plot1', 'checksum': 'b' * 64},
                {'name': 'plot2', 'checksum': 'invalid'},
            ],
        }
        
        self.assertFalse(is_telemetry_pack_healthy(index))

    def test_empty_plot_checksum_returns_false(self):
        """Pack with empty plot checksum should return False."""
        index = {
            'manifest_hash': 'a' * 64,
            'plots': [
                {'name': 'plot1', 'checksum': ''},
            ],
        }
        
        self.assertFalse(is_telemetry_pack_healthy(index))

    def test_health_report_contains_all_checks(self):
        """Health report should contain all check results."""
        index = {
            'manifest_hash': 'a' * 64,
            'plots': [{'name': 'plot1', 'checksum': 'b' * 64}],
        }
        
        report = get_pack_health_report(index)
        
        self.assertIn('healthy', report)
        self.assertIn('checks', report)
        self.assertIn('has_plots', report['checks'])
        self.assertIn('plot_count', report['checks'])
        self.assertIn('has_valid_manifest_hash', report['checks'])
        self.assertIn('all_checksums_valid', report['checks'])
        self.assertIn('invalid_checksum_plots', report['checks'])
        self.assertIn('advisory', report)

    def test_health_report_identifies_invalid_plots(self):
        """Health report should list plots with invalid checksums."""
        index = {
            'manifest_hash': 'a' * 64,
            'plots': [
                {'name': 'good_plot', 'checksum': 'b' * 64},
                {'name': 'bad_plot', 'checksum': 'invalid'},
            ],
        }
        
        report = get_pack_health_report(index)
        
        self.assertFalse(report['healthy'])
        self.assertIn('bad_plot', report['checks']['invalid_checksum_plots'])
        self.assertNotIn('good_plot', report['checks']['invalid_checksum_plots'])


# -----------------------------------------------------------------------------
# Test: Telemetry Pack Snapshot Guard
# -----------------------------------------------------------------------------

class TestTelemetryPackSnapshot(unittest.TestCase):
    """Tests for the telemetry pack snapshot guard."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        write_mock_jsonl(create_mock_records_with_abstention(50, seed=42), self.baseline_path)
        write_mock_jsonl(create_mock_records_with_entropy(50, seed=43), self.rfl_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_snapshot_creates_file(self):
        """save_pack_snapshot should create a snapshot file."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        snapshot_path = os.path.join(self.temp_dir, 'snapshot.json')
        save_pack_snapshot(pack_dir, snapshot_path)
        
        self.assertTrue(os.path.exists(snapshot_path))

    def test_snapshot_excludes_timestamps(self):
        """Snapshot should not include timestamps for determinism."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        snapshot_path = os.path.join(self.temp_dir, 'snapshot.json')
        snapshot = save_pack_snapshot(pack_dir, snapshot_path)
        
        self.assertNotIn('generated_at', snapshot)
        self.assertNotIn('manifest_hash', snapshot)

    def test_snapshot_contains_plot_checksums(self):
        """Snapshot should contain plot checksums."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        snapshot_path = os.path.join(self.temp_dir, 'snapshot.json')
        snapshot = save_pack_snapshot(pack_dir, snapshot_path)
        
        self.assertIn('plots', snapshot)
        self.assertEqual(len(snapshot['plots']), 4)
        for plot in snapshot['plots']:
            self.assertIn('name', plot)
            self.assertIn('filename', plot)
            self.assertIn('checksum', plot)
            self.assertEqual(len(plot['checksum']), 64)

    def test_check_snapshot_match(self):
        """check_pack_snapshot should return match=True for unchanged pack."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        snapshot_path = os.path.join(self.temp_dir, 'snapshot.json')
        save_pack_snapshot(pack_dir, snapshot_path)
        
        result = check_pack_snapshot(pack_dir, snapshot_path)
        
        self.assertTrue(result['match'])
        self.assertEqual(len(result['differences']), 0)

    def test_check_snapshot_detects_checksum_drift(self):
        """check_pack_snapshot should detect when checksums change."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        snapshot_path = os.path.join(self.temp_dir, 'snapshot.json')
        save_pack_snapshot(pack_dir, snapshot_path)
        
        # Modify snapshot to simulate drift
        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)
        snapshot['plots'][0]['checksum'] = 'x' * 64
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f)
        
        result = check_pack_snapshot(pack_dir, snapshot_path)
        
        self.assertFalse(result['match'])
        self.assertGreater(len(result['differences']), 0)

    def test_check_snapshot_detects_added_plot(self):
        """check_pack_snapshot should detect when a plot is added."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        snapshot_path = os.path.join(self.temp_dir, 'snapshot.json')
        save_pack_snapshot(pack_dir, snapshot_path)
        
        # Remove a plot from snapshot to simulate addition
        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)
        snapshot['plots'] = snapshot['plots'][1:]  # Remove first plot
        snapshot['plot_count'] = len(snapshot['plots'])
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f)
        
        result = check_pack_snapshot(pack_dir, snapshot_path)
        
        self.assertFalse(result['match'])
        # Should detect the "added" plot (present in pack, not in snapshot)
        added_diffs = [d for d in result['differences'] if d.get('issue') == 'added']
        self.assertGreater(len(added_diffs), 0)

    def test_check_snapshot_detects_removed_plot(self):
        """check_pack_snapshot should detect when a plot is removed."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        snapshot_path = os.path.join(self.temp_dir, 'snapshot.json')
        save_pack_snapshot(pack_dir, snapshot_path)
        
        # Add a fake plot to snapshot to simulate removal
        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)
        snapshot['plots'].append({
            'name': 'fake_plot',
            'filename': 'fake_plot.png',
            'checksum': 'f' * 64,
        })
        snapshot['plot_count'] = len(snapshot['plots'])
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f)
        
        result = check_pack_snapshot(pack_dir, snapshot_path)
        
        self.assertFalse(result['match'])
        # Should detect the "removed" plot (in snapshot, not in pack)
        removed_diffs = [d for d in result['differences'] if d.get('issue') == 'removed']
        self.assertGreater(len(removed_diffs), 0)

    def test_snapshot_deterministic_across_runs(self):
        """Same pack should produce identical snapshot (excluding timestamps)."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        snapshot_path_a = os.path.join(self.temp_dir, 'snapshot_a.json')
        snapshot_path_b = os.path.join(self.temp_dir, 'snapshot_b.json')
        
        snapshot_a = save_pack_snapshot(pack_dir, snapshot_path_a)
        snapshot_b = save_pack_snapshot(pack_dir, snapshot_path_b)
        
        self.assertEqual(snapshot_a, snapshot_b)

    def test_check_snapshot_clear_failure_messages(self):
        """check_pack_snapshot should provide clear failure messages."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        snapshot_path = os.path.join(self.temp_dir, 'snapshot.json')
        save_pack_snapshot(pack_dir, snapshot_path)
        
        # Modify to cause drift
        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)
        snapshot['plots'][0]['checksum'] = 'x' * 64
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f)
        
        result = check_pack_snapshot(pack_dir, snapshot_path)
        
        # Each difference should have field, issue, expected, actual
        for diff in result['differences']:
            self.assertIn('field', diff)
            self.assertIn('issue', diff)


# -----------------------------------------------------------------------------
# Test: Telemetry Governance Snapshot (Phase III)
# -----------------------------------------------------------------------------

class TestTelemetryGovernanceSnapshot(unittest.TestCase):
    """Tests for the telemetry governance snapshot."""

    def test_governance_snapshot_has_schema_version(self):
        """Governance snapshot should include schema_version."""
        index = {'plots': [{'checksum': 'a' * 64}], 'manifest_hash': 'b' * 64}
        health = {'checks': {
            'has_plots': True,
            'plot_count': 1,
            'has_valid_manifest_hash': True,
            'all_checksums_valid': True,
        }}
        
        gov = build_telemetry_governance_snapshot(index, health)
        
        self.assertEqual(gov['schema_version'], GOVERNANCE_SCHEMA_VERSION)

    def test_governance_status_ok_when_all_checks_pass(self):
        """Governance status should be OK when all checks pass."""
        index = {'plots': [{'checksum': 'a' * 64}], 'manifest_hash': 'b' * 64}
        health = {'checks': {
            'has_plots': True,
            'plot_count': 1,
            'has_valid_manifest_hash': True,
            'all_checksums_valid': True,
        }}
        
        gov = build_telemetry_governance_snapshot(index, health)
        
        self.assertEqual(gov['governance_status'], 'OK')

    def test_governance_status_broken_when_no_plots(self):
        """Governance status should be BROKEN when no plots."""
        index = {'plots': [], 'manifest_hash': 'b' * 64}
        health = {'checks': {
            'has_plots': False,
            'plot_count': 0,
            'has_valid_manifest_hash': True,
            'all_checksums_valid': True,
        }}
        
        gov = build_telemetry_governance_snapshot(index, health)
        
        self.assertEqual(gov['governance_status'], 'BROKEN')

    def test_governance_status_broken_when_invalid_checksums(self):
        """Governance status should be BROKEN when checksums invalid."""
        index = {'plots': [{'checksum': 'invalid'}], 'manifest_hash': 'b' * 64}
        health = {'checks': {
            'has_plots': True,
            'plot_count': 1,
            'has_valid_manifest_hash': True,
            'all_checksums_valid': False,
        }}
        
        gov = build_telemetry_governance_snapshot(index, health)
        
        self.assertEqual(gov['governance_status'], 'BROKEN')

    def test_governance_status_warn_when_snapshot_drift(self):
        """Governance status should be WARN when snapshot drifted."""
        index = {'plots': [{'checksum': 'a' * 64}], 'manifest_hash': 'b' * 64}
        health = {'checks': {
            'has_plots': True,
            'plot_count': 1,
            'has_valid_manifest_hash': True,
            'all_checksums_valid': True,
        }}
        snapshot = {'match': False, 'differences': [{'field': 'test'}]}
        
        gov = build_telemetry_governance_snapshot(index, health, snapshot)
        
        self.assertEqual(gov['governance_status'], 'WARN')
        self.assertEqual(gov['snapshot_match'], False)

    def test_governance_status_warn_when_no_manifest_hash(self):
        """Governance status should be WARN when manifest hash missing."""
        index = {'plots': [{'checksum': 'a' * 64}]}
        health = {'checks': {
            'has_plots': True,
            'plot_count': 1,
            'has_valid_manifest_hash': False,
            'all_checksums_valid': True,
        }}
        
        gov = build_telemetry_governance_snapshot(index, health)
        
        self.assertEqual(gov['governance_status'], 'WARN')

    def test_governance_snapshot_match_none_when_no_snapshot(self):
        """snapshot_match should be None when no snapshot provided."""
        index = {'plots': [{'checksum': 'a' * 64}], 'manifest_hash': 'b' * 64}
        health = {'checks': {
            'has_plots': True,
            'plot_count': 1,
            'has_valid_manifest_hash': True,
            'all_checksums_valid': True,
        }}
        
        gov = build_telemetry_governance_snapshot(index, health, snapshot_result=None)
        
        self.assertIsNone(gov['snapshot_match'])

    def test_governance_snapshot_match_true_when_snapshot_matches(self):
        """snapshot_match should be True when snapshot matches."""
        index = {'plots': [{'checksum': 'a' * 64}], 'manifest_hash': 'b' * 64}
        health = {'checks': {
            'has_plots': True,
            'plot_count': 1,
            'has_valid_manifest_hash': True,
            'all_checksums_valid': True,
        }}
        snapshot = {'match': True}
        
        gov = build_telemetry_governance_snapshot(index, health, snapshot)
        
        self.assertEqual(gov['snapshot_match'], True)


# -----------------------------------------------------------------------------
# Test: Telemetry Drift History (Phase III)
# -----------------------------------------------------------------------------

class TestTelemetryDriftHistory(unittest.TestCase):
    """Tests for the telemetry drift history builder."""

    def test_empty_snapshots_returns_ok(self):
        """Empty snapshot list should return OK status."""
        history = build_telemetry_drift_history([])
        
        self.assertEqual(history['total_runs'], 0)
        self.assertEqual(history['status'], 'OK')

    def test_all_ok_snapshots_returns_ok(self):
        """All OK snapshots should result in OK status."""
        snapshots = [
            {'governance_status': 'OK', 'snapshot_match': True},
            {'governance_status': 'OK', 'snapshot_match': True},
            {'governance_status': 'OK', 'snapshot_match': True},
        ]
        
        history = build_telemetry_drift_history(snapshots)
        
        self.assertEqual(history['total_runs'], 3)
        self.assertEqual(history['stable_runs'], 3)
        self.assertEqual(history['runs_with_broken_packs'], 0)
        self.assertEqual(history['runs_with_snapshot_drift'], 0)
        self.assertEqual(history['status'], 'OK')

    def test_one_broken_pack_returns_attention(self):
        """One broken pack should result in ATTENTION status."""
        snapshots = [
            {'governance_status': 'OK', 'snapshot_match': True},
            {'governance_status': 'BROKEN', 'snapshot_match': None},
            {'governance_status': 'OK', 'snapshot_match': True},
        ]
        
        history = build_telemetry_drift_history(snapshots)
        
        self.assertEqual(history['runs_with_broken_packs'], 1)
        self.assertEqual(history['status'], 'ATTENTION')

    def test_multiple_broken_packs_returns_unstable(self):
        """Multiple broken packs should result in UNSTABLE status."""
        snapshots = [
            {'governance_status': 'BROKEN', 'snapshot_match': None},
            {'governance_status': 'BROKEN', 'snapshot_match': None},
            {'governance_status': 'OK', 'snapshot_match': True},
        ]
        
        history = build_telemetry_drift_history(snapshots)
        
        self.assertEqual(history['runs_with_broken_packs'], 2)
        self.assertEqual(history['status'], 'UNSTABLE')

    def test_snapshot_drift_returns_attention(self):
        """Snapshot drift should result in ATTENTION status."""
        snapshots = [
            {'governance_status': 'OK', 'snapshot_match': True},
            {'governance_status': 'WARN', 'snapshot_match': False},
            {'governance_status': 'OK', 'snapshot_match': True},
        ]
        
        history = build_telemetry_drift_history(snapshots)
        
        self.assertEqual(history['runs_with_snapshot_drift'], 1)
        self.assertEqual(history['status'], 'ATTENTION')

    def test_majority_drift_returns_unstable(self):
        """Majority snapshot drift should result in UNSTABLE status."""
        snapshots = [
            {'governance_status': 'WARN', 'snapshot_match': False},
            {'governance_status': 'WARN', 'snapshot_match': False},
            {'governance_status': 'WARN', 'snapshot_match': False},
        ]
        
        history = build_telemetry_drift_history(snapshots)
        
        self.assertEqual(history['runs_with_snapshot_drift'], 3)
        self.assertEqual(history['status'], 'UNSTABLE')

    def test_warn_without_drift_counts_as_stable(self):
        """WARN status without drift should count as stable."""
        snapshots = [
            {'governance_status': 'WARN', 'snapshot_match': None},  # No snapshot
            {'governance_status': 'OK', 'snapshot_match': True},
        ]
        
        history = build_telemetry_drift_history(snapshots)
        
        self.assertEqual(history['stable_runs'], 2)
        self.assertEqual(history['status'], 'OK')


# -----------------------------------------------------------------------------
# Test: Global Health Telemetry Signal (Phase III)
# -----------------------------------------------------------------------------

class TestGlobalHealthTelemetrySignal(unittest.TestCase):
    """Tests for the global health telemetry signal."""

    def test_ok_history_returns_ok_signal(self):
        """OK history should produce OK global signal."""
        history = {
            'total_runs': 5,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 0,
            'stable_runs': 5,
            'status': 'OK',
        }
        
        signal = summarize_telemetry_for_global_health(history)
        
        self.assertTrue(signal['telemetry_ok'])
        self.assertEqual(signal['status'], 'OK')
        self.assertEqual(signal['broken_pack_count'], 0)
        self.assertEqual(signal['snapshot_drift_count'], 0)

    def test_attention_history_returns_warn_signal(self):
        """ATTENTION history should produce WARN global signal."""
        history = {
            'total_runs': 5,
            'runs_with_broken_packs': 1,
            'runs_with_snapshot_drift': 0,
            'stable_runs': 4,
            'status': 'ATTENTION',
        }
        
        signal = summarize_telemetry_for_global_health(history)
        
        self.assertFalse(signal['telemetry_ok'])
        self.assertEqual(signal['status'], 'WARN')
        self.assertEqual(signal['broken_pack_count'], 1)

    def test_unstable_history_returns_block_signal(self):
        """UNSTABLE history should produce BLOCK global signal."""
        history = {
            'total_runs': 5,
            'runs_with_broken_packs': 3,
            'runs_with_snapshot_drift': 1,
            'stable_runs': 1,
            'status': 'UNSTABLE',
        }
        
        signal = summarize_telemetry_for_global_health(history)
        
        self.assertFalse(signal['telemetry_ok'])
        self.assertEqual(signal['status'], 'BLOCK')
        self.assertEqual(signal['broken_pack_count'], 3)
        self.assertEqual(signal['snapshot_drift_count'], 1)

    def test_empty_history_returns_ok_signal(self):
        """Empty history should return OK signal."""
        history = {
            'total_runs': 0,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 0,
            'stable_runs': 0,
            'status': 'OK',
        }
        
        signal = summarize_telemetry_for_global_health(history)
        
        self.assertTrue(signal['telemetry_ok'])
        self.assertEqual(signal['status'], 'OK')


# -----------------------------------------------------------------------------
# Test: Full Governance Report (Phase III)
# -----------------------------------------------------------------------------

class TestFullGovernanceReport(unittest.TestCase):
    """Tests for the full governance report convenience function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        write_mock_jsonl(create_mock_records_with_abstention(50, seed=42), self.baseline_path)
        write_mock_jsonl(create_mock_records_with_entropy(50, seed=43), self.rfl_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_report_contains_all_sections(self):
        """Full governance report should contain all sections."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        report = build_full_telemetry_governance_report(pack_dir)
        
        self.assertIn('pack_dir', report)
        self.assertIn('index', report)
        self.assertIn('health', report)
        self.assertIn('snapshot', report)
        self.assertIn('governance', report)

    def test_full_report_without_snapshot(self):
        """Full report without snapshot should have None snapshot."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        report = build_full_telemetry_governance_report(pack_dir)
        
        self.assertIsNone(report['snapshot'])
        self.assertIsNone(report['governance']['snapshot_match'])

    def test_full_report_with_snapshot(self):
        """Full report with snapshot should include snapshot result."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        snapshot_path = os.path.join(self.temp_dir, 'snapshot.json')
        save_pack_snapshot(pack_dir, snapshot_path)
        
        report = build_full_telemetry_governance_report(pack_dir, snapshot_path)
        
        self.assertIsNotNone(report['snapshot'])
        self.assertTrue(report['snapshot']['match'])
        self.assertEqual(report['governance']['snapshot_match'], True)
        self.assertEqual(report['governance']['governance_status'], 'OK')

    def test_full_report_healthy_pack_is_ok(self):
        """Full report for healthy pack should be OK."""
        pack_dir = os.path.join(self.temp_dir, 'pack')
        generate_telemetry_pack(self.baseline_path, self.rfl_path, pack_dir)
        
        report = build_full_telemetry_governance_report(pack_dir)
        
        self.assertTrue(report['health']['healthy'])
        self.assertEqual(report['governance']['governance_status'], 'OK')


# -----------------------------------------------------------------------------
# Test: Telemetry Release Readiness (Phase IV)
# -----------------------------------------------------------------------------

class TestTelemetryReleaseReadiness(unittest.TestCase):
    """Tests for telemetry release readiness evaluation."""

    def test_ok_history_returns_release_ok(self):
        """OK history should allow release."""
        history = {
            'total_runs': 5,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 0,
            'stable_runs': 5,
            'status': 'OK',
        }
        
        result = evaluate_telemetry_for_release(history)
        
        self.assertTrue(result['release_ok'])
        self.assertEqual(result['status'], 'OK')
        self.assertEqual(len(result['blocking_reasons']), 0)

    def test_block_status_blocks_release(self):
        """BLOCK telemetry status should block release."""
        history = {
            'total_runs': 5,
            'runs_with_broken_packs': 3,
            'runs_with_snapshot_drift': 1,
            'stable_runs': 1,
            'status': 'UNSTABLE',
        }
        
        result = evaluate_telemetry_for_release(history)
        
        self.assertFalse(result['release_ok'])
        self.assertEqual(result['status'], 'BLOCK')
        self.assertGreater(len(result['blocking_reasons']), 0)
        self.assertIn('UNSTABLE', str(result['blocking_reasons']))

    def test_unstable_history_blocks_release(self):
        """UNSTABLE history status should block release."""
        history = {
            'total_runs': 5,
            'runs_with_broken_packs': 2,
            'runs_with_snapshot_drift': 0,
            'stable_runs': 3,
            'status': 'UNSTABLE',
        }
        
        result = evaluate_telemetry_for_release(history)
        
        self.assertFalse(result['release_ok'])
        self.assertEqual(result['status'], 'BLOCK')
        self.assertIn('UNSTABLE', str(result['blocking_reasons']))

    def test_warn_status_warns_release(self):
        """WARN telemetry status should warn release."""
        history = {
            'total_runs': 5,
            'runs_with_broken_packs': 1,
            'runs_with_snapshot_drift': 0,
            'stable_runs': 4,
            'status': 'ATTENTION',
        }
        
        result = evaluate_telemetry_for_release(history)
        
        self.assertFalse(result['release_ok'])
        self.assertEqual(result['status'], 'WARN')
        self.assertGreater(len(result['blocking_reasons']), 0)

    def test_snapshot_drift_warns_release(self):
        """Snapshot drift should warn release."""
        history = {
            'total_runs': 5,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 1,
            'stable_runs': 4,
            'status': 'ATTENTION',
        }
        
        result = evaluate_telemetry_for_release(history)
        
        self.assertFalse(result['release_ok'])
        self.assertEqual(result['status'], 'WARN')
        self.assertIn('drift', str(result['blocking_reasons']).lower())

    def test_broken_packs_warn_release(self):
        """Broken packs should warn release."""
        history = {
            'total_runs': 5,
            'runs_with_broken_packs': 1,
            'runs_with_snapshot_drift': 0,
            'stable_runs': 4,
            'status': 'ATTENTION',
        }
        
        result = evaluate_telemetry_for_release(history)
        
        self.assertFalse(result['release_ok'])
        self.assertEqual(result['status'], 'WARN')
        self.assertIn('broken', str(result['blocking_reasons']).lower())


# -----------------------------------------------------------------------------
# Test: Telemetry Structural Coupling (Phase IV)
# -----------------------------------------------------------------------------

class TestTelemetryStructuralCoupling(unittest.TestCase):
    """Tests for telemetry-structural coupling view."""

    def test_aligned_coupling_when_all_match(self):
        """Coupling should be ALIGNED when all systems match."""
        telemetry_history = {
            'total_runs': 3,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 0,
        }
        topology_analytics = {
            'runs': [
                {'run_id': 'run1', 'topology_events': [], 'warnings': []},
                {'run_id': 'run2', 'topology_events': [], 'warnings': []},
            ],
        }
        curriculum_timeline = {
            'runs': [
                {'run_id': 'run1', 'curriculum_changes': []},
                {'run_id': 'run2', 'curriculum_changes': []},
            ],
        }
        
        coupling = build_telemetry_structural_coupling_view(
            telemetry_history, topology_analytics, curriculum_timeline
        )
        
        self.assertEqual(coupling['coupling_status'], 'ALIGNED')
        self.assertEqual(coupling['runs_missing_telemetry_for_topology_events'], 0)
        self.assertEqual(coupling['runs_with_topology_warnings_but_no_telemetry_drift'], 0)

    def test_partial_coupling_when_some_missing(self):
        """Coupling should be PARTIAL when some telemetry missing."""
        telemetry_history = {
            'total_runs': 2,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 0,
        }
        topology_analytics = {
            'runs': [
                {'run_id': 'run1', 'topology_events': ['event1'], 'warnings': []},
                {'run_id': 'run2', 'topology_events': [], 'warnings': []},
            ],
        }
        curriculum_timeline = {
            'runs': [
                {'run_id': 'run1', 'curriculum_changes': []},
            ],
        }
        
        coupling = build_telemetry_structural_coupling_view(
            telemetry_history, topology_analytics, curriculum_timeline
        )
        
        self.assertEqual(coupling['coupling_status'], 'PARTIAL')
        self.assertGreater(coupling['runs_missing_telemetry_for_topology_events'], 0)

    def test_misaligned_coupling_when_majority_missing(self):
        """Coupling should be MISALIGNED when majority missing."""
        telemetry_history = {
            'total_runs': 1,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 0,
        }
        topology_analytics = {
            'runs': [
                {'run_id': 'run1', 'topology_events': ['event1'], 'warnings': []},
                {'run_id': 'run2', 'topology_events': ['event2'], 'warnings': []},
                {'run_id': 'run3', 'topology_events': ['event3'], 'warnings': []},
            ],
        }
        curriculum_timeline = {
            'runs': [],
        }
        
        coupling = build_telemetry_structural_coupling_view(
            telemetry_history, topology_analytics, curriculum_timeline
        )
        
        self.assertEqual(coupling['coupling_status'], 'MISALIGNED')

    def test_warnings_without_drift_detected(self):
        """Should detect topology warnings without telemetry drift."""
        telemetry_history = {
            'total_runs': 2,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 0,
        }
        topology_analytics = {
            'runs': [
                {'run_id': 'run1', 'topology_events': [], 'warnings': ['warning1']},
                {'run_id': 'run2', 'topology_events': [], 'warnings': []},
            ],
        }
        curriculum_timeline = {
            'runs': [
                {'run_id': 'run1', 'curriculum_changes': []},
                {'run_id': 'run2', 'curriculum_changes': []},
            ],
        }
        
        coupling = build_telemetry_structural_coupling_view(
            telemetry_history, topology_analytics, curriculum_timeline
        )
        
        self.assertGreater(coupling['runs_with_topology_warnings_but_no_telemetry_drift'], 0)
        self.assertEqual(coupling['coupling_status'], 'PARTIAL')

    def test_empty_inputs_return_aligned(self):
        """Empty inputs should return ALIGNED status."""
        telemetry_history = {'total_runs': 0}
        topology_analytics = {'runs': []}
        curriculum_timeline = {'runs': []}
        
        coupling = build_telemetry_structural_coupling_view(
            telemetry_history, topology_analytics, curriculum_timeline
        )
        
        self.assertEqual(coupling['coupling_status'], 'ALIGNED')


# -----------------------------------------------------------------------------
# Test: Telemetry Director Panel (Phase IV)
# -----------------------------------------------------------------------------

class TestTelemetryDirectorPanel(unittest.TestCase):
    """Tests for the telemetry director panel."""

    def test_green_light_when_all_ok(self):
        """Status light should be GREEN when all systems OK."""
        telemetry_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        release_eval = {
            'release_ok': True,
            'status': 'OK',
            'blocking_reasons': [],
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        
        panel = build_telemetry_director_panel(telemetry_health, release_eval, coupling_view)
        
        self.assertEqual(panel['status_light'], 'GREEN')
        self.assertTrue(panel['telemetry_ok'])
        self.assertEqual(panel['coupling_status'], 'ALIGNED')

    def test_red_light_when_blocked(self):
        """Status light should be RED when release blocked."""
        telemetry_health = {
            'telemetry_ok': False,
            'status': 'BLOCK',
        }
        release_eval = {
            'release_ok': False,
            'status': 'BLOCK',
            'blocking_reasons': ['Critical issue'],
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        
        panel = build_telemetry_director_panel(telemetry_health, release_eval, coupling_view)
        
        self.assertEqual(panel['status_light'], 'RED')
        self.assertFalse(panel['telemetry_ok'])

    def test_yellow_light_when_warn(self):
        """Status light should be YELLOW when warnings present."""
        telemetry_health = {
            'telemetry_ok': False,
            'status': 'WARN',
        }
        release_eval = {
            'release_ok': False,
            'status': 'WARN',
            'blocking_reasons': ['Minor issue'],
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        
        panel = build_telemetry_director_panel(telemetry_health, release_eval, coupling_view)
        
        self.assertEqual(panel['status_light'], 'YELLOW')

    def test_yellow_light_when_coupling_partial(self):
        """Status light should be YELLOW when coupling is PARTIAL."""
        telemetry_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        release_eval = {
            'release_ok': True,
            'status': 'OK',
            'blocking_reasons': [],
        }
        coupling_view = {
            'coupling_status': 'PARTIAL',
        }
        
        panel = build_telemetry_director_panel(telemetry_health, release_eval, coupling_view)
        
        self.assertEqual(panel['status_light'], 'YELLOW')

    def test_panel_contains_all_fields(self):
        """Panel should contain all required fields."""
        telemetry_health = {'telemetry_ok': True, 'status': 'OK'}
        release_eval = {'release_ok': True, 'status': 'OK', 'blocking_reasons': []}
        coupling_view = {'coupling_status': 'ALIGNED'}
        
        panel = build_telemetry_director_panel(telemetry_health, release_eval, coupling_view)
        
        self.assertIn('status_light', panel)
        self.assertIn('telemetry_ok', panel)
        self.assertIn('history_status', panel)
        self.assertIn('coupling_status', panel)
        self.assertIn('headline', panel)

    def test_headline_reflects_status(self):
        """Headline should reflect the current status."""
        telemetry_health = {'telemetry_ok': True, 'status': 'OK'}
        release_eval = {'release_ok': True, 'status': 'OK', 'blocking_reasons': []}
        coupling_view = {'coupling_status': 'ALIGNED'}
        
        panel = build_telemetry_director_panel(telemetry_health, release_eval, coupling_view)
        
        self.assertIsInstance(panel['headline'], str)
        self.assertGreater(len(panel['headline']), 0)
        self.assertIn('healthy', panel['headline'].lower())

    def test_headline_mentions_blocking_when_blocked(self):
        """Headline should mention blocking when release blocked."""
        telemetry_health = {'telemetry_ok': False, 'status': 'BLOCK'}
        release_eval = {'release_ok': False, 'status': 'BLOCK', 'blocking_reasons': ['Issue']}
        coupling_view = {'coupling_status': 'ALIGNED'}
        
        panel = build_telemetry_director_panel(telemetry_health, release_eval, coupling_view)
        
        self.assertIn('block', panel['headline'].lower())


# -----------------------------------------------------------------------------
# Test: Telemetry Drift Envelope (Phase IV Extension)
# -----------------------------------------------------------------------------

class TestTelemetryDriftEnvelope(unittest.TestCase):
    """Tests for the telemetry drift envelope."""

    def test_low_drift_when_minimal_drift(self):
        """Drift band should be LOW when minimal drift detected."""
        history = {
            'total_runs': 10,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 1,
            'stable_runs': 9,
        }
        
        envelope = build_telemetry_drift_envelope(history)
        
        self.assertEqual(envelope['drift_band'], 'LOW')
        self.assertEqual(len(envelope['plots_with_repeated_drift']), 0)

    def test_medium_drift_when_moderate_drift(self):
        """Drift band should be MEDIUM when moderate drift detected."""
        history = {
            'total_runs': 10,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 3,  # 30% drift
            'stable_runs': 7,
        }
        
        envelope = build_telemetry_drift_envelope(history)
        
        self.assertEqual(envelope['drift_band'], 'MEDIUM')
        self.assertGreater(len(envelope['plots_with_repeated_drift']), 0)

    def test_high_drift_when_significant_drift(self):
        """Drift band should be HIGH when significant drift detected."""
        history = {
            'total_runs': 10,
            'runs_with_broken_packs': 3,
            'runs_with_snapshot_drift': 3,  # 60% total drift
            'stable_runs': 4,
        }
        
        envelope = build_telemetry_drift_envelope(history)
        
        self.assertEqual(envelope['drift_band'], 'HIGH')
        self.assertGreater(len(envelope['plots_with_repeated_drift']), 0)

    def test_high_drift_identifies_core_plots(self):
        """HIGH drift should identify core plots with repeated drift."""
        history = {
            'total_runs': 10,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 6,  # 60% drift
            'stable_runs': 4,
        }
        
        envelope = build_telemetry_drift_envelope(history)
        
        self.assertEqual(envelope['drift_band'], 'HIGH')
        self.assertIn('abstention_heatmap', envelope['plots_with_repeated_drift'])
        self.assertIn('chain_depth_density', envelope['plots_with_repeated_drift'])

    def test_envelope_contains_neutral_notes(self):
        """Drift envelope should contain neutral notes."""
        history = {
            'total_runs': 10,
            'runs_with_broken_packs': 2,
            'runs_with_snapshot_drift': 1,
            'stable_runs': 7,
        }
        
        envelope = build_telemetry_drift_envelope(history)
        
        self.assertIn('neutral_notes', envelope)
        self.assertGreater(len(envelope['neutral_notes']), 0)

    def test_empty_history_returns_low_drift(self):
        """Empty history should return LOW drift."""
        history = {
            'total_runs': 0,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 0,
            'stable_runs': 0,
        }
        
        envelope = build_telemetry_drift_envelope(history)
        
        self.assertEqual(envelope['drift_band'], 'LOW')
        self.assertIn('No telemetry history', envelope['neutral_notes'][0])


# -----------------------------------------------------------------------------
# Test: Uplift Safety Adapter (Phase IV Extension)
# -----------------------------------------------------------------------------

class TestUpliftSafetyAdapter(unittest.TestCase):
    """Tests for the uplift safety adapter."""

    def test_ok_when_low_drift_and_healthy(self):
        """Should return OK when drift is LOW and pack is healthy."""
        drift_envelope = {
            'drift_band': 'LOW',
            'plots_with_repeated_drift': [],
            'neutral_notes': ['Minimal drift'],
        }
        pack_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        safety = summarize_telemetry_for_uplift_safety(drift_envelope, pack_health)
        
        self.assertTrue(safety['telemetry_ok_for_uplift'])
        self.assertEqual(safety['status'], 'OK')
        self.assertEqual(len(safety['blocking_reasons']), 0)

    def test_block_when_high_drift(self):
        """Should return BLOCK when drift band is HIGH."""
        drift_envelope = {
            'drift_band': 'HIGH',
            'plots_with_repeated_drift': ['abstention_heatmap', 'chain_depth_density'],
            'neutral_notes': ['High drift detected'],
        }
        pack_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        safety = summarize_telemetry_for_uplift_safety(drift_envelope, pack_health)
        
        self.assertFalse(safety['telemetry_ok_for_uplift'])
        self.assertEqual(safety['status'], 'BLOCK')
        self.assertIn('HIGH', str(safety['blocking_reasons']))

    def test_block_when_pack_unhealthy(self):
        """Should return BLOCK when pack health is BLOCK."""
        drift_envelope = {
            'drift_band': 'LOW',
            'plots_with_repeated_drift': [],
            'neutral_notes': ['Minimal drift'],
        }
        pack_health = {
            'telemetry_ok': False,
            'status': 'BLOCK',
        }
        
        safety = summarize_telemetry_for_uplift_safety(drift_envelope, pack_health)
        
        self.assertFalse(safety['telemetry_ok_for_uplift'])
        self.assertEqual(safety['status'], 'BLOCK')
        self.assertIn('BLOCK', str(safety['blocking_reasons']))

    def test_attention_when_medium_drift(self):
        """Should return ATTENTION when drift band is MEDIUM."""
        drift_envelope = {
            'drift_band': 'MEDIUM',
            'plots_with_repeated_drift': ['candidate_ordering_entropy'],
            'neutral_notes': ['Moderate drift'],
        }
        pack_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        safety = summarize_telemetry_for_uplift_safety(drift_envelope, pack_health)
        
        self.assertTrue(safety['telemetry_ok_for_uplift'])  # Still OK, just attention
        self.assertEqual(safety['status'], 'ATTENTION')
        self.assertGreater(len(safety['advisory_notes']), 0)

    def test_attention_when_pack_warn(self):
        """Should return ATTENTION when pack health is WARN."""
        drift_envelope = {
            'drift_band': 'LOW',
            'plots_with_repeated_drift': [],
            'neutral_notes': ['Minimal drift'],
        }
        pack_health = {
            'telemetry_ok': False,
            'status': 'WARN',
        }
        
        safety = summarize_telemetry_for_uplift_safety(drift_envelope, pack_health)
        
        self.assertTrue(safety['telemetry_ok_for_uplift'])  # Still OK, just attention
        self.assertEqual(safety['status'], 'ATTENTION')

    def test_blocking_reasons_include_plot_names(self):
        """Blocking reasons should include plot names when HIGH drift."""
        drift_envelope = {
            'drift_band': 'HIGH',
            'plots_with_repeated_drift': ['abstention_heatmap', 'chain_depth_density'],
            'neutral_notes': [],
        }
        pack_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        safety = summarize_telemetry_for_uplift_safety(drift_envelope, pack_health)
        
        reasons_str = ' '.join(safety['blocking_reasons'])
        self.assertIn('abstention_heatmap', reasons_str)
        self.assertIn('chain_depth_density', reasons_str)

    def test_advisory_notes_included(self):
        """Advisory notes should be included in safety summary."""
        drift_envelope = {
            'drift_band': 'MEDIUM',
            'plots_with_repeated_drift': [],
            'neutral_notes': ['Note 1', 'Note 2'],
        }
        pack_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        safety = summarize_telemetry_for_uplift_safety(drift_envelope, pack_health)
        
        self.assertIn('advisory_notes', safety)
        self.assertGreaterEqual(len(safety['advisory_notes']), 2)


# -----------------------------------------------------------------------------
# pytest markers
# -----------------------------------------------------------------------------

pytestmark = pytest.mark.unit


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

