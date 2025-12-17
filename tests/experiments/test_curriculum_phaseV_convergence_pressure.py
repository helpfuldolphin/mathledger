# PHASE II — NOT USED IN PHASE I
# File: tests/experiments/test_curriculum_phaseV_convergence_pressure.py
"""
Tests for Phase V Convergence Pressure Grid & Early-Warning Radar.

These tests verify:
1. Pressure tensor correctness (L2 norm, ranking, normalization)
2. Early-warning radar logic (likelihood bands, root drivers)
3. Director tile status-light decoding rules
4. Neutral language enforcement
5. Determinism guarantees
"""

import unittest
import json
from typing import Dict, Any

from experiments.curriculum_health import (
    build_convergence_pressure_tensor,
    build_phase_transition_early_warning_radar,
    build_convergence_director_tile,
    TransitionLikelihoodBand,
    PRESSURE_TENSOR_SCHEMA_VERSION,
    StatusLight,
)


class TestConvergencePressureTensor(unittest.TestCase):
    """Tests for convergence pressure tensor construction."""

    def setUp(self):
        """Set up test fixtures."""
        self.convergence_map_basic = {
            'convergence_status': 'STABLE',
            'slices_converging': ['slice_a'],
            'slices_diverging': ['slice_b'],
            'cross_signal_correlations': {
                'metrics↔topology': 0.5,
                'topology↔confusability': 0.5,
            },
        }

        self.convergence_map_empty = {
            'convergence_status': 'STABLE',
            'slices_converging': [],
            'slices_diverging': [],
            'cross_signal_correlations': {},
        }

    def test_pressure_tensor_structure(self):
        """Test that pressure tensor has correct structure."""
        tensor = build_convergence_pressure_tensor(self.convergence_map_basic)

        self.assertIn('schema_version', tensor)
        self.assertIn('slice_pressure_vectors', tensor)
        self.assertIn('global_pressure_norm', tensor)
        self.assertIn('pressure_ranked_slices', tensor)

    def test_pressure_tensor_schema_version(self):
        """Test that schema version is correctly set."""
        tensor = build_convergence_pressure_tensor(self.convergence_map_basic)
        self.assertEqual(tensor['schema_version'], PRESSURE_TENSOR_SCHEMA_VERSION)

    def test_pressure_vectors_normalized(self):
        """Test that pressure vectors are normalized to [0, 1]."""
        tensor = build_convergence_pressure_tensor(self.convergence_map_basic)
        vectors = tensor['slice_pressure_vectors']

        for slice_name, vector in vectors.items():
            self.assertGreaterEqual(vector['alignment'], 0.0)
            self.assertLessEqual(vector['alignment'], 1.0)
            self.assertGreaterEqual(vector['drift'], 0.0)
            self.assertLessEqual(vector['drift'], 1.0)
            self.assertGreaterEqual(vector['metric'], 0.0)
            self.assertLessEqual(vector['metric'], 1.0)

    def test_pressure_vector_converging_slice(self):
        """Test that converging slices have low alignment pressure."""
        tensor = build_convergence_pressure_tensor(self.convergence_map_basic)

        if 'slice_a' in tensor['slice_pressure_vectors']:
            vector = tensor['slice_pressure_vectors']['slice_a']
            # Converging slice should have alignment pressure = 0.0
            self.assertEqual(vector['alignment'], 0.0)

    def test_pressure_vector_diverging_slice(self):
        """Test that diverging slices have high alignment pressure."""
        tensor = build_convergence_pressure_tensor(self.convergence_map_basic)

        if 'slice_b' in tensor['slice_pressure_vectors']:
            vector = tensor['slice_pressure_vectors']['slice_b']
            # Diverging slice should have alignment pressure = 1.0
            self.assertEqual(vector['alignment'], 1.0)

    def test_global_pressure_norm_is_l2(self):
        """Test that global pressure norm is computed via L2 norm."""
        tensor = build_convergence_pressure_tensor(self.convergence_map_basic)
        global_norm = tensor['global_pressure_norm']

        # Manually compute expected L2 norm
        vectors = tensor['slice_pressure_vectors']
        all_pressures = []
        for vector in vectors.values():
            all_pressures.extend([vector['alignment'], vector['drift'], vector['metric']])

        if all_pressures:
            expected_norm = (sum(p ** 2 for p in all_pressures)) ** 0.5
            self.assertAlmostEqual(global_norm, expected_norm, places=3)

    def test_pressure_ranking_descending(self):
        """Test that slices are ranked by descending pressure norm."""
        tensor = build_convergence_pressure_tensor(self.convergence_map_basic)
        ranked = tensor['pressure_ranked_slices']
        vectors = tensor['slice_pressure_vectors']

        if len(ranked) > 1:
            # Compute norms for verification
            norms = {}
            for slice_name, vector in vectors.items():
                norm = (
                    vector['alignment'] ** 2 +
                    vector['drift'] ** 2 +
                    vector['metric'] ** 2
                ) ** 0.5
                norms[slice_name] = norm

            # Verify descending order
            for i in range(len(ranked) - 1):
                self.assertGreaterEqual(
                    norms[ranked[i]],
                    norms[ranked[i + 1]],
                    f"Ranking violation: {ranked[i]} should have higher norm than {ranked[i + 1]}"
                )

    def test_pressure_tensor_empty_slices(self):
        """Test that empty slices produce zero pressure tensor."""
        tensor = build_convergence_pressure_tensor(self.convergence_map_empty)

        self.assertEqual(tensor['global_pressure_norm'], 0.0)
        self.assertEqual(tensor['pressure_ranked_slices'], [])
        self.assertEqual(tensor['slice_pressure_vectors'], {})

    def test_pressure_tensor_is_json_serializable(self):
        """Test that pressure tensor can be serialized to JSON."""
        tensor = build_convergence_pressure_tensor(self.convergence_map_basic)
        json_str = json.dumps(tensor, sort_keys=True)
        self.assertIsInstance(json_str, str)

    def test_pressure_tensor_is_deterministic(self):
        """Test that pressure tensor is deterministic."""
        tensor1 = build_convergence_pressure_tensor(self.convergence_map_basic)
        tensor2 = build_convergence_pressure_tensor(self.convergence_map_basic)

        self.assertEqual(
            json.dumps(tensor1, sort_keys=True),
            json.dumps(tensor2, sort_keys=True)
        )

    def test_pressure_vector_keys_sorted(self):
        """Test that pressure vector keys are sorted for determinism."""
        tensor = build_convergence_pressure_tensor(self.convergence_map_basic)
        vectors = tensor['slice_pressure_vectors']

        # Keys should be sorted
        keys = list(vectors.keys())
        self.assertEqual(keys, sorted(keys))

        # Vector values should also have sorted keys
        for vector in vectors.values():
            vector_keys = list(vector.keys())
            self.assertEqual(vector_keys, sorted(vector_keys))


class TestPhaseTransitionEarlyWarningRadar(unittest.TestCase):
    """Tests for phase-transition early-warning radar."""

    def setUp(self):
        """Set up test fixtures."""
        self.pressure_tensor_low = {
            'global_pressure_norm': 0.5,
            'pressure_ranked_slices': ['slice_a'],
            'slice_pressure_vectors': {
                'slice_a': {'alignment': 0.2, 'drift': 0.3, 'metric': 0.2},
            },
        }

        self.pressure_tensor_high = {
            'global_pressure_norm': 2.5,
            'pressure_ranked_slices': ['slice_b', 'slice_c'],
            'slice_pressure_vectors': {
                'slice_b': {'alignment': 1.0, 'drift': 0.9, 'metric': 0.8},
                'slice_c': {'alignment': 0.8, 'drift': 0.7, 'metric': 0.6},
            },
        }

        self.phase_forecast_low = {
            'confidence': 0.3,
            'predicted_boundary': None,
            'estimated_versions_until': None,
            'reasons': ['Weak signals'],
        }

        self.phase_forecast_high = {
            'confidence': 0.9,
            'predicted_boundary': 'PARTIAL→MISALIGNED',
            'estimated_versions_until': 3,
            'reasons': ['Strong divergence signals'],
        }

        self.drift_timeline_low = {
            'events': [
                {'drift_status': 'NONE', 'blocking': False},
                {'drift_status': 'NONE', 'blocking': False},
            ],
        }

        self.drift_timeline_high = {
            'events': [
                {'drift_status': 'MAJOR', 'blocking': True},
                {'drift_status': 'MAJOR', 'blocking': True},
                {'drift_status': 'MAJOR', 'blocking': True},
            ],
        }

    def test_early_warning_structure(self):
        """Test that early warning has correct structure."""
        radar = build_phase_transition_early_warning_radar(
            self.pressure_tensor_low,
            self.phase_forecast_low,
            self.drift_timeline_low,
        )

        self.assertIn('transition_likelihood_band', radar)
        self.assertIn('root_drivers', radar)
        self.assertIn('first_slices_at_risk', radar)
        self.assertIn('time_to_inflection_estimate', radar)

    def test_early_warning_low_likelihood(self):
        """Test that low likelihood band is detected."""
        radar = build_phase_transition_early_warning_radar(
            self.pressure_tensor_low,
            self.phase_forecast_low,
            self.drift_timeline_low,
        )

        self.assertEqual(radar['transition_likelihood_band'], 'LOW')

    def test_early_warning_high_likelihood(self):
        """Test that high likelihood band is detected."""
        radar = build_phase_transition_early_warning_radar(
            self.pressure_tensor_high,
            self.phase_forecast_high,
            self.drift_timeline_high,
        )

        self.assertEqual(radar['transition_likelihood_band'], 'HIGH')

    def test_early_warning_medium_likelihood(self):
        """Test that medium likelihood band is detected."""
        # Create medium-pressure scenario
        pressure_medium = {
            'global_pressure_norm': 1.2,
            'pressure_ranked_slices': ['slice_a'],
            'slice_pressure_vectors': {
                'slice_a': {'alignment': 0.5, 'drift': 0.5, 'metric': 0.5},
            },
        }
        forecast_medium = {
            'confidence': 0.5,
            'predicted_boundary': 'STABLE→PARTIAL',
            'estimated_versions_until': 8,
            'reasons': ['Moderate signals'],
        }
        drift_medium = {
            'events': [
                {'drift_status': 'MINOR', 'blocking': False},
            ],
        }

        radar = build_phase_transition_early_warning_radar(
            pressure_medium,
            forecast_medium,
            drift_medium,
        )

        self.assertEqual(radar['transition_likelihood_band'], 'MEDIUM')

    def test_early_warning_root_drivers(self):
        """Test that root drivers are identified."""
        radar = build_phase_transition_early_warning_radar(
            self.pressure_tensor_high,
            self.phase_forecast_high,
            self.drift_timeline_high,
        )

        self.assertGreater(len(radar['root_drivers']), 0)
        # All drivers should be neutral strings
        for driver in radar['root_drivers']:
            self.assertIsInstance(driver, str)

    def test_early_warning_root_drivers_neutral(self):
        """Test that root drivers use neutral language."""
        radar = build_phase_transition_early_warning_radar(
            self.pressure_tensor_high,
            self.phase_forecast_high,
            self.drift_timeline_high,
        )

        forbidden_words = ['good', 'bad', 'better', 'worse', 'improve', 'degrade']
        for driver in radar['root_drivers']:
            driver_lower = driver.lower()
            for word in forbidden_words:
                self.assertNotIn(
                    word, driver_lower,
                    f"Driver contains forbidden word '{word}': {driver}"
                )

    def test_early_warning_first_slices_at_risk(self):
        """Test that first slices at risk are identified."""
        radar = build_phase_transition_early_warning_radar(
            self.pressure_tensor_high,
            self.phase_forecast_high,
            self.drift_timeline_high,
        )

        self.assertIsInstance(radar['first_slices_at_risk'], list)
        # Should be sorted
        self.assertEqual(
            radar['first_slices_at_risk'],
            sorted(radar['first_slices_at_risk'])
        )

    def test_early_warning_time_to_inflection(self):
        """Test that time to inflection is estimated."""
        radar = build_phase_transition_early_warning_radar(
            self.pressure_tensor_high,
            self.phase_forecast_high,
            self.drift_timeline_high,
        )

        # Should match forecast estimate
        self.assertEqual(
            radar['time_to_inflection_estimate'],
            self.phase_forecast_high['estimated_versions_until']
        )

    def test_early_warning_is_json_serializable(self):
        """Test that early warning can be serialized to JSON."""
        radar = build_phase_transition_early_warning_radar(
            self.pressure_tensor_low,
            self.phase_forecast_low,
            self.drift_timeline_low,
        )

        json_str = json.dumps(radar, sort_keys=True)
        self.assertIsInstance(json_str, str)

    def test_early_warning_is_deterministic(self):
        """Test that early warning is deterministic."""
        radar1 = build_phase_transition_early_warning_radar(
            self.pressure_tensor_low,
            self.phase_forecast_low,
            self.drift_timeline_low,
        )
        radar2 = build_phase_transition_early_warning_radar(
            self.pressure_tensor_low,
            self.phase_forecast_low,
            self.drift_timeline_low,
        )

        self.assertEqual(
            json.dumps(radar1, sort_keys=True),
            json.dumps(radar2, sort_keys=True)
        )


class TestConvergenceDirectorTile(unittest.TestCase):
    """Tests for convergence director tile."""

    def setUp(self):
        """Set up test fixtures."""
        self.pressure_tensor_low = {
            'global_pressure_norm': 0.8,
            'pressure_ranked_slices': [],
            'slice_pressure_vectors': {},
        }

        self.pressure_tensor_high = {
            'global_pressure_norm': 2.3,
            'pressure_ranked_slices': ['slice_a', 'slice_b'],
            'slice_pressure_vectors': {
                'slice_a': {'alignment': 1.0, 'drift': 0.9, 'metric': 0.8},
                'slice_b': {'alignment': 0.8, 'drift': 0.7, 'metric': 0.6},
            },
        }

        self.early_warning_low = {
            'transition_likelihood_band': 'LOW',
            'root_drivers': ['No strong signals'],
            'first_slices_at_risk': [],
            'time_to_inflection_estimate': None,
        }

        self.early_warning_high = {
            'transition_likelihood_band': 'HIGH',
            'root_drivers': [
                'Global pressure norm elevated (2.30)',
                'Phase boundary forecast confidence high (90.0%)',
                'Recent drift events include 3 MAJOR classification(s)',
            ],
            'first_slices_at_risk': ['slice_a', 'slice_b'],
            'time_to_inflection_estimate': 3,
        }

    def test_director_tile_structure(self):
        """Test that director tile has correct structure."""
        tile = build_convergence_director_tile(
            self.pressure_tensor_low,
            self.early_warning_low,
        )

        self.assertIn('status_light', tile)
        self.assertIn('transition_band', tile)
        self.assertIn('global_pressure_norm', tile)
        self.assertIn('headline', tile)
        self.assertIn('pressure_drivers', tile)

    def test_director_tile_green_light(self):
        """Test that GREEN light is set for low pressure and low likelihood."""
        tile = build_convergence_director_tile(
            self.pressure_tensor_low,
            self.early_warning_low,
        )

        self.assertEqual(tile['status_light'], 'GREEN')

    def test_director_tile_red_light(self):
        """Test that RED light is set for high pressure or high likelihood."""
        tile = build_convergence_director_tile(
            self.pressure_tensor_high,
            self.early_warning_high,
        )

        self.assertEqual(tile['status_light'], 'RED')

    def test_director_tile_yellow_light(self):
        """Test that YELLOW light is set for medium conditions."""
        pressure_medium = {
            'global_pressure_norm': 1.5,
            'pressure_ranked_slices': [],
            'slice_pressure_vectors': {},
        }
        warning_medium = {
            'transition_likelihood_band': 'MEDIUM',
            'root_drivers': ['Moderate signals'],
            'first_slices_at_risk': [],
            'time_to_inflection_estimate': None,
        }

        tile = build_convergence_director_tile(
            pressure_medium,
            warning_medium,
        )

        self.assertEqual(tile['status_light'], 'YELLOW')

    def test_director_tile_headline_is_neutral(self):
        """Test that headline uses neutral language."""
        tile = build_convergence_director_tile(
            self.pressure_tensor_high,
            self.early_warning_high,
        )

        headline = tile['headline'].lower()
        forbidden_words = ['good', 'bad', 'better', 'worse', 'improve', 'degrade']
        for word in forbidden_words:
            self.assertNotIn(
                word, headline,
                f"Headline contains forbidden word '{word}': {tile['headline']}"
            )

    def test_director_tile_pressure_drivers_limited(self):
        """Test that pressure drivers are limited to top 3."""
        early_warning_many = {
            'transition_likelihood_band': 'HIGH',
            'root_drivers': [
                'Driver 1',
                'Driver 2',
                'Driver 3',
                'Driver 4',
                'Driver 5',
            ],
            'first_slices_at_risk': [],
            'time_to_inflection_estimate': None,
        }

        tile = build_convergence_director_tile(
            self.pressure_tensor_high,
            early_warning_many,
        )

        self.assertLessEqual(len(tile['pressure_drivers']), 3)

    def test_director_tile_is_json_serializable(self):
        """Test that director tile can be serialized to JSON."""
        tile = build_convergence_director_tile(
            self.pressure_tensor_low,
            self.early_warning_low,
        )

        json_str = json.dumps(tile, sort_keys=True)
        self.assertIsInstance(json_str, str)

    def test_director_tile_is_deterministic(self):
        """Test that director tile is deterministic."""
        tile1 = build_convergence_director_tile(
            self.pressure_tensor_low,
            self.early_warning_low,
        )
        tile2 = build_convergence_director_tile(
            self.pressure_tensor_low,
            self.early_warning_low,
        )

        self.assertEqual(
            json.dumps(tile1, sort_keys=True),
            json.dumps(tile2, sort_keys=True)
        )


class TestTransitionLikelihoodBandEnum(unittest.TestCase):
    """Tests for TransitionLikelihoodBand enum."""

    def test_likelihood_band_values(self):
        """Test that TransitionLikelihoodBand has expected values."""
        self.assertEqual(TransitionLikelihoodBand.LOW.value, 'LOW')
        self.assertEqual(TransitionLikelihoodBand.MEDIUM.value, 'MEDIUM')
        self.assertEqual(TransitionLikelihoodBand.HIGH.value, 'HIGH')


class TestPressureTensorIntegration(unittest.TestCase):
    """Integration tests for pressure tensor with convergence map."""

    def test_pressure_tensor_from_convergence_map(self):
        """Test that pressure tensor correctly processes convergence map."""
        from experiments.curriculum_health import build_curriculum_convergence_map

        convergence_map = {
            'convergence_status': 'DIVERGING',
            'slices_converging': [],
            'slices_diverging': ['slice_a', 'slice_b'],
            'cross_signal_correlations': {
                'metrics↔topology': 0.2,
                'topology↔confusability': 0.3,
            },
        }

        tensor = build_convergence_pressure_tensor(convergence_map)

        # Diverging slices should have high alignment pressure
        self.assertIn('slice_a', tensor['slice_pressure_vectors'])
        self.assertEqual(
            tensor['slice_pressure_vectors']['slice_a']['alignment'],
            1.0
        )

        # Low correlations should result in high drift pressure
        self.assertGreater(
            tensor['slice_pressure_vectors']['slice_a']['drift'],
            0.7
        )


if __name__ == '__main__':
    unittest.main()

