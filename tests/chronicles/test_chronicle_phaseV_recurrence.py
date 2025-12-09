# PHASE V — NOT USED IN PHASE I
# File: tests/chronicles/test_chronicle_phaseV_recurrence.py
"""
Tests for Phase V Chronicle Recurrence Projection and Invariant Checking.

Tests validate:
- Recurrence projection engine
- Phase-transition drift invariant checker
- Director tile builder
- Integration with CurriculumHashLedger
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import yaml

from experiments.curriculum_hash_ledger import (
    CurriculumHashLedger,
    DriftType,
    RiskLevel,
    # Recurrence Projection
    build_recurrence_projection_engine,
    RecurrenceBand,
    RECURRENCE_HORIZON_BASE,
    RECURRENCE_DENSITY_THRESHOLD,
    # Invariant Checker
    build_phase_transition_drift_invariant_checker,
    InvariantStatus,
    # Director Tile
    build_director_tile,
    StatusLight,
    StabilityBand,
)


class TestRecurrenceProjectionEngine(unittest.TestCase):
    """Test recurrence projection engine."""

    def test_recurrence_projection_has_required_fields(self):
        """Recurrence projection should have all required fields."""
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        drift_events = {}
        stability_scores = {
            "evidence_fields": {
                "churn": {"high_churn_slice_count": 0}
            }
        }
        projection = build_recurrence_projection_engine(
            causality_map, drift_events, stability_scores
        )
        
        required_fields = [
            "recurrence_likelihood",
            "drivers",
            "projected_recurrence_horizon",
            "neutral_explanation"
        ]
        for field in required_fields:
            self.assertIn(field, projection, f"Missing field: {field}")

    def test_recurrence_likelihood_bounds(self):
        """Recurrence likelihood should be in [0.0, 1.0]."""
        causality_map = {
            "causal_links": [("event_1", "event_2")],
            "likely_root_causes": ["event_1"],
            "causality_strength_score": 0.8,
            "neutral_notes": [],
        }
        drift_events = {
            "event_1": {
                "timestamp": "2025-01-01T10:00:00Z",
                "drift_type": DriftType.STRUCTURAL.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.BLOCK.value,
            }
        }
        stability_scores = {
            "evidence_fields": {
                "churn": {"high_churn_slice_count": 2}
            }
        }
        projection = build_recurrence_projection_engine(
            causality_map, drift_events, stability_scores
        )
        
        likelihood = projection["recurrence_likelihood"]
        self.assertGreaterEqual(likelihood, 0.0)
        self.assertLessEqual(likelihood, 1.0)

    def test_recurrence_formula_applied(self):
        """Recurrence should follow formula: ∝ causality_strength × density × churn."""
        # High causality, high density, high churn → high recurrence
        causality_map = {
            "causal_links": [("event_1", "event_2"), ("event_2", "event_3")],
            "likely_root_causes": ["event_1"],
            "causality_strength_score": 0.9,
            "neutral_notes": [],
        }
        # Multiple events in short time span (high density)
        now = datetime.now(timezone.utc)
        drift_events = {
            f"event_{i}": {
                "timestamp": (now + timedelta(hours=i)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.PARAMETRIC_MINOR.value,
                "slice_name": "slice_a",
                "snapshot_index": i,
                "risk_level": RiskLevel.WARN.value,
            }
            for i in range(5)
        }
        stability_scores = {
            "evidence_fields": {
                "churn": {"high_churn_slice_count": 3}  # High churn
            }
        }
        projection = build_recurrence_projection_engine(
            causality_map, drift_events, stability_scores
        )
        
        # Should have high recurrence likelihood
        self.assertGreater(projection["recurrence_likelihood"], 0.5)

    def test_drivers_identified(self):
        """Drivers should be identified from input metrics."""
        causality_map = {
            "causal_links": [("event_1", "event_2")] * 6,  # >5 links
            "likely_root_causes": ["event_1", "event_2", "event_3"],  # >2 root causes
            "causality_strength_score": 0.7,  # >0.5
            "neutral_notes": [],
        }
        drift_events = {
            "event_1": {
                "timestamp": "2025-01-01T10:00:00Z",
                "drift_type": DriftType.STRUCTURAL.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.BLOCK.value,
            }
        }
        stability_scores = {
            "evidence_fields": {
                "churn": {"high_churn_slice_count": 4}  # High churn
            }
        }
        projection = build_recurrence_projection_engine(
            causality_map, drift_events, stability_scores
        )
        
        drivers = projection["drivers"]
        self.assertGreater(len(drivers), 0)
        # Should mention high causality strength
        driver_text = " ".join(drivers).lower()
        self.assertIn("causality", driver_text)

    def test_projected_horizon_scales_with_likelihood(self):
        """Higher recurrence likelihood should result in shorter horizon."""
        causality_map_low = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.1,
            "neutral_notes": [],
        }
        causality_map_high = {
            "causal_links": [("event_1", "event_2")] * 10,
            "likely_root_causes": ["event_1"],
            "causality_strength_score": 0.9,
            "neutral_notes": [],
        }
        drift_events = {
            "event_1": {
                "timestamp": "2025-01-01T10:00:00Z",
                "drift_type": DriftType.PARAMETRIC_MINOR.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.WARN.value,
            }
        }
        stability_scores = {
            "evidence_fields": {
                "churn": {"high_churn_slice_count": 0}
            }
        }
        
        projection_low = build_recurrence_projection_engine(
            causality_map_low, drift_events, stability_scores
        )
        projection_high = build_recurrence_projection_engine(
            causality_map_high, drift_events, stability_scores
        )
        
        # High likelihood should have shorter horizon
        self.assertLess(
            projection_high["projected_recurrence_horizon"],
            projection_low["projected_recurrence_horizon"]
        )

    def test_neutral_explanation_includes_metrics(self):
        """Neutral explanation should include key metrics."""
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.5,
            "neutral_notes": [],
        }
        drift_events = {
            "event_1": {
                "timestamp": "2025-01-01T10:00:00Z",
                "drift_type": DriftType.PARAMETRIC_MINOR.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.WARN.value,
            }
        }
        stability_scores = {
            "evidence_fields": {
                "churn": {"high_churn_slice_count": 0}
            }
        }
        projection = build_recurrence_projection_engine(
            causality_map, drift_events, stability_scores
        )
        
        explanation = projection["neutral_explanation"]
        self.assertIn("likelihood", explanation.lower())
        self.assertIn("horizon", explanation.lower())


class TestPhaseTransitionDriftInvariantChecker(unittest.TestCase):
    """Test phase-transition drift invariant checker."""

    def test_invariant_check_has_required_fields(self):
        """Invariant check should have all required fields."""
        drift_events = {}
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        check = build_phase_transition_drift_invariant_checker(drift_events, causality_map)
        
        required_fields = [
            "invariant_status",
            "broken_invariants",
            "explanations"
        ]
        for field in required_fields:
            self.assertIn(field, check, f"Missing field: {field}")

    def test_empty_events_returns_ok(self):
        """Empty drift events should return OK status."""
        drift_events = {}
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        check = build_phase_transition_drift_invariant_checker(drift_events, causality_map)
        
        self.assertEqual(check["invariant_status"], InvariantStatus.OK.value)
        self.assertEqual(len(check["broken_invariants"]), 0)

    def test_block_drift_twice_in_window_violation(self):
        """BLOCK drift twice in 3-event window should violate invariant."""
        now = datetime.now(timezone.utc)
        drift_events = {
            "event_1": {
                "timestamp": (now + timedelta(hours=1)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.STRUCTURAL.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.BLOCK.value,
            },
            "event_2": {
                "timestamp": (now + timedelta(hours=2)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.SEMANTIC.value,
                "slice_name": "slice_b",
                "snapshot_index": 2,
                "risk_level": RiskLevel.INFO.value,
            },
            "event_3": {
                "timestamp": (now + timedelta(hours=3)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.PARAMETRIC_MAJOR.value,
                "slice_name": "slice_a",  # Same slice as event_1
                "snapshot_index": 3,
                "risk_level": RiskLevel.BLOCK.value,  # Second BLOCK in same slice
            },
        }
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        check = build_phase_transition_drift_invariant_checker(drift_events, causality_map)
        
        # Should detect violation
        self.assertEqual(check["invariant_status"], InvariantStatus.VIOLATED.value)
        self.assertGreater(len(check["broken_invariants"]), 0)
        # Should mention slice_a
        violations_text = " ".join(check["broken_invariants"]).lower()
        self.assertIn("slice_a", violations_text)

    def test_structural_cascade_violation(self):
        """STRUCTURAL → PARAMETRIC_MAJOR > once per window should violate."""
        now = datetime.now(timezone.utc)
        drift_events = {
            "event_struct": {
                "timestamp": (now + timedelta(hours=1)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.STRUCTURAL.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.BLOCK.value,
            },
            "event_param1": {
                "timestamp": (now + timedelta(hours=2)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.PARAMETRIC_MAJOR.value,
                "slice_name": "slice_b",
                "snapshot_index": 2,
                "risk_level": RiskLevel.BLOCK.value,
            },
            "event_param2": {
                "timestamp": (now + timedelta(hours=3)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.PARAMETRIC_MAJOR.value,
                "slice_name": "slice_c",
                "snapshot_index": 3,
                "risk_level": RiskLevel.BLOCK.value,
            },
        }
        # Create causal links: STRUCTURAL → both PARAMETRIC_MAJOR
        causality_map = {
            "causal_links": [
                ("event_struct", "event_param1"),
                ("event_struct", "event_param2"),
            ],
            "likely_root_causes": ["event_struct"],
            "causality_strength_score": 0.5,
            "neutral_notes": [],
        }
        check = build_phase_transition_drift_invariant_checker(drift_events, causality_map)
        
        # Should detect violation (STRUCTURAL cascades to PARAMETRIC_MAJOR twice)
        violations_text = " ".join(check["broken_invariants"]).lower()
        self.assertIn("structural", violations_text)
        self.assertIn("parametric_major", violations_text)

    def test_chronological_inversion_violation(self):
        """Chronological inversion in causal links should violate."""
        now = datetime.now(timezone.utc)
        drift_events = {
            "event_later": {
                "timestamp": (now + timedelta(hours=2)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.STRUCTURAL.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.BLOCK.value,
            },
            "event_earlier": {
                "timestamp": (now + timedelta(hours=1)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.SEMANTIC.value,
                "slice_name": "slice_b",
                "snapshot_index": 2,
                "risk_level": RiskLevel.BLOCK.value,
            },
        }
        # Create causal link: later → earlier (inversion)
        causality_map = {
            "causal_links": [("event_later", "event_earlier")],
            "likely_root_causes": [],
            "causality_strength_score": 0.3,
            "neutral_notes": [],
        }
        check = build_phase_transition_drift_invariant_checker(drift_events, causality_map)
        
        # Should detect chronological inversion
        violations_text = " ".join(check["broken_invariants"]).lower()
        self.assertIn("chronological", violations_text.lower())

    def test_all_invariants_satisfied_returns_ok(self):
        """When all invariants satisfied, status should be OK."""
        now = datetime.now(timezone.utc)
        drift_events = {
            "event_1": {
                "timestamp": (now + timedelta(hours=1)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.PARAMETRIC_MINOR.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.WARN.value,
            },
            "event_2": {
                "timestamp": (now + timedelta(hours=2)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.COSMETIC.value,
                "slice_name": "slice_b",
                "snapshot_index": 2,
                "risk_level": RiskLevel.INFO.value,
            },
        }
        causality_map = {
            "causal_links": [("event_1", "event_2")],  # Valid chronological order
            "likely_root_causes": ["event_1"],
            "causality_strength_score": 0.2,
            "neutral_notes": [],
        }
        check = build_phase_transition_drift_invariant_checker(drift_events, causality_map)
        
        self.assertEqual(check["invariant_status"], InvariantStatus.OK.value)
        self.assertEqual(len(check["broken_invariants"]), 0)

    def test_explanations_provided_for_violations(self):
        """Explanations should be provided for each violation."""
        now = datetime.now(timezone.utc)
        drift_events = {
            "event_1": {
                "timestamp": (now + timedelta(hours=1)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.STRUCTURAL.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.BLOCK.value,
            },
            "event_2": {
                "timestamp": (now + timedelta(hours=2)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.SEMANTIC.value,
                "slice_name": "slice_a",  # Same slice
                "snapshot_index": 2,
                "risk_level": RiskLevel.BLOCK.value,  # Second BLOCK
            },
            "event_3": {
                "timestamp": (now + timedelta(hours=3)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.PARAMETRIC_MAJOR.value,
                "slice_name": "slice_a",  # Same slice
                "snapshot_index": 3,
                "risk_level": RiskLevel.BLOCK.value,  # Third BLOCK in window
            },
        }
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        check = build_phase_transition_drift_invariant_checker(drift_events, causality_map)
        
        # Should have explanations
        self.assertGreater(len(check["explanations"]), 0)
        # Explanations should mention the violation
        explanations_text = " ".join(check["explanations"]).lower()
        self.assertIn("block", explanations_text)


class TestDirectorTile(unittest.TestCase):
    """Test director tile builder."""

    def test_director_tile_has_required_fields(self):
        """Director tile should have all required fields."""
        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": ["Low recurrence indicators"],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test explanation",
        }
        invariant_check = {
            "invariant_status": InvariantStatus.OK.value,
            "broken_invariants": [],
            "explanations": ["All invariants satisfied."],
        }
        stability_estimate = {
            "stability_band": StabilityBand.HIGH.value,
            "axes_contributing": [],
            "headline": "Stable",
            "evidence_fields": {},
            "composite_score": 0.8,
        }
        tile = build_director_tile(recurrence_projection, invariant_check, stability_estimate)
        
        required_fields = [
            "status_light",
            "recurrence_band",
            "invariants_ok",
            "highlighted_cases",
            "headline"
        ]
        for field in required_fields:
            self.assertIn(field, tile, f"Missing field: {field}")

    def test_status_light_red_when_invariants_violated(self):
        """Status light should be RED when invariants violated."""
        recurrence_projection = {
            "recurrence_likelihood": 0.2,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": InvariantStatus.VIOLATED.value,
            "broken_invariants": ["Test violation"],
            "explanations": ["Violation explanation"],
        }
        stability_estimate = {
            "stability_band": StabilityBand.HIGH.value,
            "axes_contributing": [],
            "headline": "Stable",
            "evidence_fields": {},
            "composite_score": 0.8,
        }
        tile = build_director_tile(recurrence_projection, invariant_check, stability_estimate)
        
        self.assertEqual(tile["status_light"], StatusLight.RED.value)

    def test_status_light_yellow_when_high_recurrence(self):
        """Status light should be YELLOW when high recurrence."""
        recurrence_projection = {
            "recurrence_likelihood": 0.8,  # High recurrence
            "drivers": ["High causality"],
            "projected_recurrence_horizon": 15,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": InvariantStatus.OK.value,
            "broken_invariants": [],
            "explanations": ["All invariants satisfied."],
        }
        stability_estimate = {
            "stability_band": StabilityBand.MEDIUM.value,
            "axes_contributing": [],
            "headline": "Moderate",
            "evidence_fields": {},
            "composite_score": 0.5,
        }
        tile = build_director_tile(recurrence_projection, invariant_check, stability_estimate)
        
        self.assertEqual(tile["status_light"], StatusLight.YELLOW.value)

    def test_status_light_green_when_all_ok(self):
        """Status light should be GREEN when all systems nominal."""
        recurrence_projection = {
            "recurrence_likelihood": 0.2,  # Low recurrence
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": InvariantStatus.OK.value,
            "broken_invariants": [],
            "explanations": ["All invariants satisfied."],
        }
        stability_estimate = {
            "stability_band": StabilityBand.HIGH.value,
            "axes_contributing": [],
            "headline": "Stable",
            "evidence_fields": {},
            "composite_score": 0.8,
        }
        tile = build_director_tile(recurrence_projection, invariant_check, stability_estimate)
        
        self.assertEqual(tile["status_light"], StatusLight.GREEN.value)

    def test_recurrence_band_mapped_correctly(self):
        """Recurrence band should map correctly from likelihood."""
        for likelihood, expected_band in [
            (0.8, RecurrenceBand.HIGH),
            (0.5, RecurrenceBand.MEDIUM),
            (0.2, RecurrenceBand.LOW),
        ]:
            recurrence_projection = {
                "recurrence_likelihood": likelihood,
                "drivers": [],
                "projected_recurrence_horizon": 30,
                "neutral_explanation": "Test",
            }
            invariant_check = {
                "invariant_status": InvariantStatus.OK.value,
                "broken_invariants": [],
                "explanations": [],
            }
            stability_estimate = {
                "stability_band": StabilityBand.HIGH.value,
                "axes_contributing": [],
                "headline": "Test",
                "evidence_fields": {},
                "composite_score": 0.8,
            }
            tile = build_director_tile(recurrence_projection, invariant_check, stability_estimate)
            
            self.assertEqual(tile["recurrence_band"], expected_band.value)

    def test_highlighted_cases_includes_violations(self):
        """Highlighted cases should include invariant violations."""
        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": InvariantStatus.VIOLATED.value,
            "broken_invariants": ["Violation 1", "Violation 2"],
            "explanations": [],
        }
        stability_estimate = {
            "stability_band": StabilityBand.HIGH.value,
            "axes_contributing": [],
            "headline": "Test",
            "evidence_fields": {},
            "composite_score": 0.8,
        }
        tile = build_director_tile(recurrence_projection, invariant_check, stability_estimate)
        
        cases_text = " ".join(tile["highlighted_cases"]).lower()
        self.assertIn("violation", cases_text)

    def test_headline_includes_status(self):
        """Headline should include status information."""
        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": InvariantStatus.OK.value,
            "broken_invariants": [],
            "explanations": [],
        }
        stability_estimate = {
            "stability_band": StabilityBand.HIGH.value,
            "axes_contributing": [],
            "headline": "Test",
            "evidence_fields": {},
            "composite_score": 0.8,
        }
        tile = build_director_tile(recurrence_projection, invariant_check, stability_estimate)
        
        headline = tile["headline"].lower()
        self.assertIn("status", headline)
        self.assertIn("recurrence", headline)
        self.assertIn("invariant", headline)

    def test_headline_neutral_tone(self):
        """Headline should use neutral, descriptive language."""
        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": InvariantStatus.OK.value,
            "broken_invariants": [],
            "explanations": [],
        }
        stability_estimate = {
            "stability_band": StabilityBand.HIGH.value,
            "axes_contributing": [],
            "headline": "Test",
            "evidence_fields": {},
            "composite_score": 0.8,
        }
        tile = build_director_tile(recurrence_projection, invariant_check, stability_estimate)
        
        headline = tile["headline"].lower()
        # Should not contain judgmental words
        self.assertNotIn("bad", headline)
        self.assertNotIn("good", headline)
        self.assertNotIn("problem", headline)
        self.assertNotIn("error", headline)


class TestLedgerRecurrenceMethods(unittest.TestCase):
    """Test CurriculumHashLedger recurrence and invariant methods."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "slice_a": {"depth": 4},
                "slice_b": {"depth": 6}
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_get_recurrence_projection(self):
        """get_recurrence_projection should return valid projection."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        projection = self.ledger.get_recurrence_projection()
        
        self.assertIn("recurrence_likelihood", projection)
        self.assertIn("projected_recurrence_horizon", projection)

    def test_get_invariant_check(self):
        """get_invariant_check should return valid check."""
        self.ledger.record_snapshot(str(self.config_path))
        
        check = self.ledger.get_invariant_check()
        
        self.assertIn("invariant_status", check)
        self.assertIn("broken_invariants", check)

    def test_get_director_tile(self):
        """get_director_tile should return valid tile."""
        self.ledger.record_snapshot(str(self.config_path))
        
        tile = self.ledger.get_director_tile()
        
        self.assertIn("status_light", tile)
        self.assertIn("recurrence_band", tile)
        self.assertIn("invariants_ok", tile)
        self.assertIn("highlighted_cases", tile)
        self.assertIn("headline", tile)


class TestRecurrenceDeterminism(unittest.TestCase):
    """Test recurrence projection determinism."""

    def test_recurrence_deterministic_across_runs(self):
        """Recurrence projection should be deterministic."""
        causality_map = {
            "causal_links": [("event_1", "event_2")],
            "likely_root_causes": ["event_1"],
            "causality_strength_score": 0.5,
            "neutral_notes": [],
        }
        drift_events = {
            "event_1": {
                "timestamp": "2025-01-01T10:00:00Z",
                "drift_type": DriftType.STRUCTURAL.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.BLOCK.value,
            },
            "event_2": {
                "timestamp": "2025-01-01T11:00:00Z",
                "drift_type": DriftType.SEMANTIC.value,
                "slice_name": "slice_b",
                "snapshot_index": 2,
                "risk_level": RiskLevel.BLOCK.value,
            },
        }
        stability_scores = {
            "evidence_fields": {
                "churn": {"high_churn_slice_count": 1}
            }
        }
        
        proj1 = build_recurrence_projection_engine(causality_map, drift_events, stability_scores)
        proj2 = build_recurrence_projection_engine(causality_map, drift_events, stability_scores)
        
        self.assertEqual(proj1["recurrence_likelihood"], proj2["recurrence_likelihood"])
        self.assertEqual(proj1["projected_recurrence_horizon"], proj2["projected_recurrence_horizon"])


class TestInvariantCheckerDeterminism(unittest.TestCase):
    """Test invariant checker determinism."""

    def test_invariant_check_deterministic(self):
        """Invariant check should be deterministic."""
        now = datetime.now(timezone.utc)
        drift_events = {
            "event_1": {
                "timestamp": (now + timedelta(hours=1)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.STRUCTURAL.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.BLOCK.value,
            },
            "event_2": {
                "timestamp": (now + timedelta(hours=2)).isoformat().replace('+00:00', 'Z'),
                "drift_type": DriftType.SEMANTIC.value,
                "slice_name": "slice_a",
                "snapshot_index": 2,
                "risk_level": RiskLevel.BLOCK.value,
            },
        }
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        
        check1 = build_phase_transition_drift_invariant_checker(drift_events, causality_map)
        check2 = build_phase_transition_drift_invariant_checker(drift_events, causality_map)
        
        self.assertEqual(check1["invariant_status"], check2["invariant_status"])
        self.assertEqual(len(check1["broken_invariants"]), len(check2["broken_invariants"]))


if __name__ == '__main__':
    unittest.main()

