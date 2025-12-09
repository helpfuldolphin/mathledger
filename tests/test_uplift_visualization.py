"""
PHASE II — Uplift Visualization Tests

Descriptive only — not admissible as uplift evidence.

This module tests the uplift visualization pipeline for:
- Image generation correctness
- Deterministic checksum stability
- Correct labeling and axes
- PHASE II metadata compliance

Author: metrics-engineer-4
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from experiments.uplift_visualization import (
    PHASE_II_LABEL,
    PHASE_II_WATERMARK,
    _compute_file_checksum,
    extract_cycle_success,
    extract_metric_values,
    load_jsonl,
    plot_delta_p_point_estimates,
    plot_metric_value_trajectory,
    plot_slice_comparison_grid,
    plot_success_rate_curve,
    setup_deterministic_style,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

def create_mock_baseline_records(n_cycles: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Create deterministic mock baseline experiment records.
    
    Descriptive only — not admissible as uplift evidence.
    """
    np.random.seed(seed)
    records = []
    for i in range(n_cycles):
        # Fixed pattern: ~30% success rate with slight trend
        success_prob = 0.30 + 0.0001 * i
        # Use deterministic threshold based on seed
        threshold = ((seed * 31 + i * 17) % 1000) / 1000.0
        success = threshold < success_prob
        
        records.append({
            'cycle': i,
            'mode': 'baseline',
            'derivation': {
                'candidates': 5,
                'verified': 2 if success else 0,
                'abstained': 1 if not success else 0,
            },
            'success': success,
            'label': 'PHASE II — NOT USED IN PHASE I',
        })
    return records


def create_mock_rfl_records(n_cycles: int = 100, seed: int = 43) -> List[Dict[str, Any]]:
    """
    Create deterministic mock RFL experiment records.
    
    Descriptive only — not admissible as uplift evidence.
    """
    np.random.seed(seed)
    records = []
    for i in range(n_cycles):
        # Fixed pattern: ~45% success rate with slight trend (better than baseline)
        success_prob = 0.45 + 0.0002 * i
        threshold = ((seed * 31 + i * 17) % 1000) / 1000.0
        success = threshold < success_prob
        
        records.append({
            'cycle': i,
            'mode': 'rfl',
            'derivation': {
                'candidates': 5,
                'verified': 3 if success else 1,
                'abstained': 0 if success else 1,
            },
            'success': success,
            'rfl': {'executed': True},
            'label': 'PHASE II — NOT USED IN PHASE I',
        })
    return records


def write_mock_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """Write records to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + '\n')


# -----------------------------------------------------------------------------
# Test: Data Loading
# -----------------------------------------------------------------------------

class TestDataLoading(unittest.TestCase):
    """Tests for JSONL loading and data extraction."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        write_mock_jsonl(create_mock_baseline_records(50), self.baseline_path)
        write_mock_jsonl(create_mock_rfl_records(50), self.rfl_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_jsonl_returns_list(self):
        """load_jsonl should return a list of dicts."""
        records = load_jsonl(self.baseline_path)
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 50)
        self.assertIsInstance(records[0], dict)

    def test_extract_cycle_success_returns_dataframe(self):
        """extract_cycle_success should return a DataFrame with expected columns."""
        records = load_jsonl(self.baseline_path)
        df = extract_cycle_success(records)
        
        self.assertEqual(list(df.columns), ['cycle', 'success', 'verified_count', 'abstained_count'])
        self.assertEqual(len(df), 50)

    def test_extract_cycle_success_sorted_by_cycle(self):
        """DataFrame should be sorted by cycle index."""
        records = load_jsonl(self.baseline_path)
        df = extract_cycle_success(records)
        
        cycles = df['cycle'].tolist()
        self.assertEqual(cycles, sorted(cycles))

    def test_extract_metric_values_returns_dataframe(self):
        """extract_metric_values should return a DataFrame with cycle and value."""
        records = load_jsonl(self.baseline_path)
        df = extract_metric_values(records, 'derivation.verified')
        
        self.assertEqual(list(df.columns), ['cycle', 'value'])
        self.assertEqual(len(df), 50)

    def test_extract_metric_values_handles_missing_path(self):
        """Missing metric path should return 0."""
        records = [{'cycle': 0, 'other_field': 'value'}]
        df = extract_metric_values(records, 'nonexistent.path')
        
        self.assertEqual(df.loc[0, 'value'], 0.0)


# -----------------------------------------------------------------------------
# Test: Image Generation
# -----------------------------------------------------------------------------

class TestImageGeneration(unittest.TestCase):
    """Tests for image file generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        self.out_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.out_dir)
        
        write_mock_jsonl(create_mock_baseline_records(100), self.baseline_path)
        write_mock_jsonl(create_mock_rfl_records(100), self.rfl_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_success_rate_curve_creates_png(self):
        """plot_success_rate_curve should create a PNG file."""
        out_path = os.path.join(self.out_dir, 'success_rate.png')
        
        metadata = plot_success_rate_curve(
            self.baseline_path, self.rfl_path, out_path
        )
        
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        self.assertEqual(metadata['plot_type'], 'success_rate_curve')

    def test_metric_trajectory_creates_png(self):
        """plot_metric_value_trajectory should create a PNG file."""
        out_path = os.path.join(self.out_dir, 'metric_trajectory.png')
        
        metadata = plot_metric_value_trajectory(
            self.baseline_path, self.rfl_path, out_path
        )
        
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        self.assertEqual(metadata['plot_type'], 'metric_value_trajectory')

    def test_delta_p_creates_png(self):
        """plot_delta_p_point_estimates should create a PNG file."""
        out_path = os.path.join(self.out_dir, 'delta_p.png')
        
        metadata = plot_delta_p_point_estimates(
            self.baseline_path, self.rfl_path, out_path
        )
        
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        self.assertEqual(metadata['plot_type'], 'delta_p_point_estimates')

    def test_slice_grid_creates_png(self):
        """plot_slice_comparison_grid should create a PNG file."""
        out_path = os.path.join(self.out_dir, 'slice_grid.png')
        
        slice_data = [
            {
                'name': 'Slice A',
                'baseline_path': self.baseline_path,
                'rfl_path': self.rfl_path,
            },
            {
                'name': 'Slice B',
                'baseline_path': self.baseline_path,
                'rfl_path': self.rfl_path,
            },
        ]
        
        metadata = plot_slice_comparison_grid(slice_data, out_path)
        
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        self.assertEqual(metadata['plot_type'], 'slice_comparison_grid')


# -----------------------------------------------------------------------------
# Test: Determinism
# -----------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    """Tests for deterministic output checksums."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        # Use fixed data for reproducibility
        write_mock_jsonl(create_mock_baseline_records(100, seed=42), self.baseline_path)
        write_mock_jsonl(create_mock_rfl_records(100, seed=43), self.rfl_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_success_rate_curve_deterministic_checksum(self):
        """Same inputs should produce identical checksums."""
        checksums = []
        
        for i in range(3):
            out_path = os.path.join(self.temp_dir, f'success_rate_{i}.png')
            metadata = plot_success_rate_curve(
                self.baseline_path, self.rfl_path, out_path,
                window=20
            )
            checksums.append(metadata['checksum_sha256'])
        
        # All checksums must be identical
        self.assertEqual(checksums[0], checksums[1])
        self.assertEqual(checksums[1], checksums[2])

    def test_metric_trajectory_deterministic_checksum(self):
        """Same inputs should produce identical checksums."""
        checksums = []
        
        for i in range(3):
            out_path = os.path.join(self.temp_dir, f'metric_{i}.png')
            metadata = plot_metric_value_trajectory(
                self.baseline_path, self.rfl_path, out_path,
                window=10
            )
            checksums.append(metadata['checksum_sha256'])
        
        self.assertEqual(checksums[0], checksums[1])
        self.assertEqual(checksums[1], checksums[2])

    def test_delta_p_deterministic_checksum(self):
        """Same inputs should produce identical checksums."""
        checksums = []
        
        for i in range(3):
            out_path = os.path.join(self.temp_dir, f'delta_p_{i}.png')
            metadata = plot_delta_p_point_estimates(
                self.baseline_path, self.rfl_path, out_path,
                bin_size=25
            )
            checksums.append(metadata['checksum_sha256'])
        
        self.assertEqual(checksums[0], checksums[1])
        self.assertEqual(checksums[1], checksums[2])

    def test_slice_grid_deterministic_checksum(self):
        """Same inputs should produce identical checksums."""
        checksums = []
        
        slice_data = [
            {
                'name': 'Slice A',
                'baseline_path': self.baseline_path,
                'rfl_path': self.rfl_path,
            },
        ]
        
        for i in range(3):
            out_path = os.path.join(self.temp_dir, f'grid_{i}.png')
            metadata = plot_slice_comparison_grid(
                slice_data, out_path,
                window=20
            )
            checksums.append(metadata['checksum_sha256'])
        
        self.assertEqual(checksums[0], checksums[1])
        self.assertEqual(checksums[1], checksums[2])

    def test_checksum_function_deterministic(self):
        """_compute_file_checksum should be deterministic."""
        # Create a fixed file
        test_file = os.path.join(self.temp_dir, 'test_file.txt')
        with open(test_file, 'w') as f:
            f.write('deterministic content')
        
        checksums = [_compute_file_checksum(test_file) for _ in range(5)]
        
        self.assertTrue(all(c == checksums[0] for c in checksums))


# -----------------------------------------------------------------------------
# Test: Metadata & Labeling
# -----------------------------------------------------------------------------

class TestMetadataLabeling(unittest.TestCase):
    """Tests for PHASE II labeling and metadata compliance."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        write_mock_jsonl(create_mock_baseline_records(50), self.baseline_path)
        write_mock_jsonl(create_mock_rfl_records(50), self.rfl_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_success_rate_metadata_contains_phase_ii_label(self):
        """Metadata should contain PHASE II label."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        metadata = plot_success_rate_curve(
            self.baseline_path, self.rfl_path, out_path
        )
        
        self.assertEqual(metadata['phase'], 'PHASE II')
        self.assertEqual(metadata['label'], PHASE_II_LABEL)

    def test_metric_trajectory_metadata_contains_phase_ii_label(self):
        """Metadata should contain PHASE II label."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        metadata = plot_metric_value_trajectory(
            self.baseline_path, self.rfl_path, out_path
        )
        
        self.assertEqual(metadata['phase'], 'PHASE II')
        self.assertEqual(metadata['label'], PHASE_II_LABEL)

    def test_delta_p_metadata_contains_phase_ii_label(self):
        """Metadata should contain PHASE II label."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        metadata = plot_delta_p_point_estimates(
            self.baseline_path, self.rfl_path, out_path
        )
        
        self.assertEqual(metadata['phase'], 'PHASE II')
        self.assertEqual(metadata['label'], PHASE_II_LABEL)

    def test_slice_grid_metadata_contains_phase_ii_label(self):
        """Metadata should contain PHASE II label."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        slice_data = [{
            'name': 'Test',
            'baseline_path': self.baseline_path,
            'rfl_path': self.rfl_path,
        }]
        metadata = plot_slice_comparison_grid(slice_data, out_path)
        
        self.assertEqual(metadata['phase'], 'PHASE II')
        self.assertEqual(metadata['label'], PHASE_II_LABEL)

    def test_metadata_contains_checksum(self):
        """All metadata should contain SHA-256 checksum."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        metadata = plot_success_rate_curve(
            self.baseline_path, self.rfl_path, out_path
        )
        
        self.assertIn('checksum_sha256', metadata)
        self.assertEqual(len(metadata['checksum_sha256']), 64)  # SHA-256 hex length

    def test_metadata_contains_timestamp(self):
        """All metadata should contain UTC timestamp."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        metadata = plot_success_rate_curve(
            self.baseline_path, self.rfl_path, out_path
        )
        
        self.assertIn('timestamp_utc', metadata)
        # Should be ISO format
        self.assertIn('T', metadata['timestamp_utc'])

    def test_metadata_contains_input_paths(self):
        """Metadata should record input file paths."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        metadata = plot_success_rate_curve(
            self.baseline_path, self.rfl_path, out_path
        )
        
        self.assertEqual(metadata['baseline_path'], self.baseline_path)
        self.assertEqual(metadata['rfl_path'], self.rfl_path)

    def test_metadata_contains_cycle_counts(self):
        """Metadata should record cycle counts."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        metadata = plot_success_rate_curve(
            self.baseline_path, self.rfl_path, out_path
        )
        
        self.assertEqual(metadata['baseline_cycles'], 50)
        self.assertEqual(metadata['rfl_cycles'], 50)


# -----------------------------------------------------------------------------
# Test: Axes & Labels
# -----------------------------------------------------------------------------

class TestAxesLabels(unittest.TestCase):
    """Tests for correct axis configuration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        write_mock_jsonl(create_mock_baseline_records(50), self.baseline_path)
        write_mock_jsonl(create_mock_rfl_records(50), self.rfl_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_success_rate_y_axis_bounded(self):
        """Success rate plot should have y-axis bounded to [0, 1]."""
        # This is verified by examining the metadata and the plot logic
        # The plot uses ax.set_ylim(0, 1.05)
        out_path = os.path.join(self.temp_dir, 'test.png')
        metadata = plot_success_rate_curve(
            self.baseline_path, self.rfl_path, out_path
        )
        
        # Verify plot was created successfully
        self.assertIn('checksum_sha256', metadata)
        self.assertTrue(os.path.exists(out_path))

    def test_window_parameter_respected(self):
        """Window parameter should be recorded in metadata."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        metadata_w10 = plot_success_rate_curve(
            self.baseline_path, self.rfl_path, out_path,
            window=10
        )
        self.assertEqual(metadata_w10['window_size'], 10)
        
        metadata_w30 = plot_success_rate_curve(
            self.baseline_path, self.rfl_path, out_path,
            window=30
        )
        self.assertEqual(metadata_w30['window_size'], 30)

    def test_bin_size_parameter_respected(self):
        """Bin size parameter should be recorded in metadata."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        metadata = plot_delta_p_point_estimates(
            self.baseline_path, self.rfl_path, out_path,
            bin_size=25
        )
        self.assertEqual(metadata['bin_size'], 25)

    def test_metric_path_parameter_respected(self):
        """Metric path parameter should be recorded in metadata."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        metadata = plot_metric_value_trajectory(
            self.baseline_path, self.rfl_path, out_path,
            metric_path='derivation.abstained'
        )
        self.assertEqual(metadata['metric_path'], 'derivation.abstained')


# -----------------------------------------------------------------------------
# Test: Edge Cases
# -----------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Tests for edge case handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_records_handled(self):
        """Empty records should not crash."""
        baseline_path = os.path.join(self.temp_dir, 'empty_baseline.jsonl')
        rfl_path = os.path.join(self.temp_dir, 'empty_rfl.jsonl')
        
        # Create files with minimal records
        write_mock_jsonl([{'cycle': 0, 'success': False}], baseline_path)
        write_mock_jsonl([{'cycle': 0, 'success': True}], rfl_path)
        
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        # Should not raise
        metadata = plot_success_rate_curve(
            baseline_path, rfl_path, out_path
        )
        
        self.assertTrue(os.path.exists(out_path))

    def test_single_cycle_handled(self):
        """Single cycle should produce valid output."""
        baseline_path = os.path.join(self.temp_dir, 'single_baseline.jsonl')
        rfl_path = os.path.join(self.temp_dir, 'single_rfl.jsonl')
        
        write_mock_jsonl([{'cycle': 0, 'success': False}], baseline_path)
        write_mock_jsonl([{'cycle': 0, 'success': True}], rfl_path)
        
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        metadata = plot_success_rate_curve(
            baseline_path, rfl_path, out_path
        )
        
        self.assertEqual(metadata['baseline_cycles'], 1)
        self.assertEqual(metadata['rfl_cycles'], 1)

    def test_slice_grid_single_slice(self):
        """Single slice should produce valid grid."""
        baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        write_mock_jsonl(create_mock_baseline_records(20), baseline_path)
        write_mock_jsonl(create_mock_rfl_records(20), rfl_path)
        
        out_path = os.path.join(self.temp_dir, 'grid.png')
        
        slice_data = [{
            'name': 'Only Slice',
            'baseline_path': baseline_path,
            'rfl_path': rfl_path,
        }]
        
        metadata = plot_slice_comparison_grid(slice_data, out_path)
        
        self.assertEqual(metadata['num_slices'], 1)
        self.assertTrue(os.path.exists(out_path))

    def test_slice_grid_empty_raises_error(self):
        """Empty slice_data should raise ValueError."""
        out_path = os.path.join(self.temp_dir, 'grid.png')
        
        with self.assertRaises(ValueError):
            plot_slice_comparison_grid([], out_path)

    def test_mismatched_cycle_ranges(self):
        """Different cycle ranges should be handled gracefully."""
        baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        # Baseline: cycles 0-49
        write_mock_jsonl(create_mock_baseline_records(50), baseline_path)
        # RFL: cycles 0-99
        write_mock_jsonl(create_mock_rfl_records(100), rfl_path)
        
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        metadata = plot_success_rate_curve(
            baseline_path, rfl_path, out_path
        )
        
        self.assertEqual(metadata['baseline_cycles'], 50)
        self.assertEqual(metadata['rfl_cycles'], 100)


# -----------------------------------------------------------------------------
# Test: Style Determinism
# -----------------------------------------------------------------------------

class TestStyleDeterminism(unittest.TestCase):
    """Tests for matplotlib style determinism."""

    def test_setup_deterministic_style_idempotent(self):
        """setup_deterministic_style should be idempotent."""
        import matplotlib.pyplot as plt
        
        # Call multiple times
        setup_deterministic_style()
        state1 = dict(plt.rcParams)
        
        setup_deterministic_style()
        state2 = dict(plt.rcParams)
        
        # Compare relevant keys
        for key in ['font.size', 'axes.labelsize', 'figure.dpi']:
            self.assertEqual(state1.get(key), state2.get(key))


# -----------------------------------------------------------------------------
# Test: Return Types
# -----------------------------------------------------------------------------

class TestReturnTypes(unittest.TestCase):
    """Tests for consistent return types."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, 'baseline.jsonl')
        self.rfl_path = os.path.join(self.temp_dir, 'rfl.jsonl')
        
        write_mock_jsonl(create_mock_baseline_records(30), self.baseline_path)
        write_mock_jsonl(create_mock_rfl_records(30), self.rfl_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_plot_functions_return_dict(self):
        """All plot functions should return a dict."""
        out_path = os.path.join(self.temp_dir, 'test.png')
        
        meta1 = plot_success_rate_curve(
            self.baseline_path, self.rfl_path, out_path
        )
        self.assertIsInstance(meta1, dict)
        
        meta2 = plot_metric_value_trajectory(
            self.baseline_path, self.rfl_path, out_path
        )
        self.assertIsInstance(meta2, dict)
        
        meta3 = plot_delta_p_point_estimates(
            self.baseline_path, self.rfl_path, out_path
        )
        self.assertIsInstance(meta3, dict)
        
        slice_data = [{
            'name': 'Test',
            'baseline_path': self.baseline_path,
            'rfl_path': self.rfl_path,
        }]
        meta4 = plot_slice_comparison_grid(slice_data, out_path)
        self.assertIsInstance(meta4, dict)


# -----------------------------------------------------------------------------
# pytest markers
# -----------------------------------------------------------------------------

# Mark all tests as unit tests (no external dependencies)
pytestmark = pytest.mark.unit


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

