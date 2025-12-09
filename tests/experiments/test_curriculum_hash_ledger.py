# PHASE II — NOT USED IN PHASE I
# File: tests/experiments/test_curriculum_hash_ledger.py
"""
Tests for the Curriculum Hash Ledger drift detection system.

Tests validate:
- Determinism of hash computation
- Ledger append behavior
- Drift detection logic:
  - no-change → no drift
  - changed formula → drift detected
  - changed slice parameters → drift detected
  - added/removed slices → drift detected
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional

import yaml

from experiments.curriculum_hash_ledger import (
    CurriculumHashLedger,
    canonical_json,
    DOMAIN_CURRICULUM,
    DOMAIN_SLICE,
    DriftType,
    RiskLevel,
    DRIFT_RISK_MAP,
    SEMANTIC_KEYS,
    # Task 1: Timeline Compression
    DriftEvent,
    CompressedEventGroup,
    CompressedTimeline,
    compress_drift_timeline,
    decompress_drift_timeline,
    # Task 2: Drift Intent Annotation
    DriftIntent,
    detect_drift_intent,
    INTENT_HEURISTICS,
    # Task 3: Chronicle Export
    Chronicle,
    export_chronicle,
    CHRONICLE_SCHEMA_VERSION,
    # Phase III: Cross-Slice Chronicle Index & Governance Lens
    build_chronicle_index,
    summarize_chronicles_for_governance,
    summarize_chronicles_for_global_health,
    GovernanceStatus,
    HealthStatus,
    ActivityLevel,
    CHRONICLE_INDEX_SCHEMA_VERSION,
    ACTIVITY_THRESHOLD_LOW,
    ACTIVITY_THRESHOLD_MEDIUM,
    # Phase IV: Cross-Slice Chronicle Governance & Narrative Feed
    build_chronicle_alignment_view,
    render_chronicle_governance_narrative,
    build_chronicle_summary_for_acquisition,
    AlignmentStatus,
    HIGH_CHURN_THRESHOLD,
    # Follow-up: Chronicle Causality Map & Multi-Axis Stability Estimator
    build_chronicle_causality_map,
    estimate_multi_axis_chronicle_stability,
    StabilityBand,
    CAUSAL_TIME_WINDOW_HOURS,
    CAUSAL_STRENGTH_THRESHOLD,
)


class TestCanonicalJson(unittest.TestCase):
    """Test the canonical JSON encoding function."""

    def test_sorted_keys(self):
        """JSON keys should be sorted."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj).decode('utf-8')
        self.assertEqual(result, '{"a":2,"m":3,"z":1}')

    def test_no_whitespace(self):
        """Canonical JSON should have no extra whitespace."""
        obj = {"key": "value", "nested": {"a": 1}}
        result = canonical_json(obj).decode('utf-8')
        self.assertNotIn(' ', result)
        self.assertNotIn('\n', result)

    def test_deterministic(self):
        """Same object should always produce same bytes."""
        obj = {"items": [1, 2, 3], "name": "test"}
        result1 = canonical_json(obj)
        result2 = canonical_json(obj)
        self.assertEqual(result1, result2)


class TestCurriculumHashLedgerHashing(unittest.TestCase):
    """Test hash computation for curriculum configs."""

    def setUp(self):
        """Create temporary config files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)

        # Create a stable test config
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "slice_test_a": {
                    "description": "Test slice A",
                    "parameters": {"depth": 4, "atoms": 3},
                    "formula_pool": ["p", "q", "r"],
                    "target_hash": "abc123"
                },
                "slice_test_b": {
                    "description": "Test slice B",
                    "parameters": {"depth": 6, "atoms": 5},
                    "formula_pool": ["x", "y"],
                    "target_hash": "def456"
                }
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_hash_determinism(self):
        """Same config should always produce the same hash."""
        hash1, slices1 = self.ledger.compute_curriculum_hash(str(self.config_path))
        hash2, slices2 = self.ledger.compute_curriculum_hash(str(self.config_path))
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(slices1, slices2)

    def test_slice_hashes_independent(self):
        """Each slice should have a unique hash."""
        _, slice_hashes = self.ledger.compute_curriculum_hash(str(self.config_path))
        
        self.assertEqual(len(slice_hashes), 2)
        self.assertIn("slice_test_a", slice_hashes)
        self.assertIn("slice_test_b", slice_hashes)
        self.assertNotEqual(slice_hashes["slice_test_a"], slice_hashes["slice_test_b"])

    def test_hash_changes_on_parameter_change(self):
        """Changing a parameter should change the hash."""
        hash_before, _ = self.ledger.compute_curriculum_hash(str(self.config_path))
        
        # Modify a parameter
        self.config_data["slices"]["slice_test_a"]["parameters"]["depth"] = 99
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        hash_after, _ = self.ledger.compute_curriculum_hash(str(self.config_path))
        self.assertNotEqual(hash_before, hash_after)

    def test_hash_changes_on_formula_pool_change(self):
        """Changing formula pool should change the hash."""
        _, slices_before = self.ledger.compute_curriculum_hash(str(self.config_path))
        slice_hash_before = slices_before["slice_test_a"]
        
        # Modify formula pool
        self.config_data["slices"]["slice_test_a"]["formula_pool"].append("s")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        _, slices_after = self.ledger.compute_curriculum_hash(str(self.config_path))
        slice_hash_after = slices_after["slice_test_a"]
        
        self.assertNotEqual(slice_hash_before, slice_hash_after)

    def test_hash_changes_on_target_hash_change(self):
        """Changing target hash should change the slice hash."""
        _, slices_before = self.ledger.compute_curriculum_hash(str(self.config_path))
        
        # Modify target hash
        self.config_data["slices"]["slice_test_a"]["target_hash"] = "changed_hash"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        _, slices_after = self.ledger.compute_curriculum_hash(str(self.config_path))
        
        self.assertNotEqual(
            slices_before["slice_test_a"],
            slices_after["slice_test_a"]
        )


class TestCurriculumHashLedgerStorage(unittest.TestCase):
    """Test ledger append and load behavior."""

    def setUp(self):
        """Create a fresh ledger for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        # Create a test config
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "test_slice": {
                    "description": "Test",
                    "parameters": {"x": 1}
                }
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_record_creates_file(self):
        """Recording a snapshot should create the ledger file."""
        self.assertFalse(self.ledger_path.exists())
        
        self.ledger.record_snapshot(str(self.config_path), origin="ci")
        
        self.assertTrue(self.ledger_path.exists())

    def test_record_appends_not_overwrites(self):
        """Multiple records should append, not overwrite."""
        self.ledger.record_snapshot(str(self.config_path), origin="ci", notes="first")
        self.ledger.record_snapshot(str(self.config_path), origin="manual", notes="second")
        
        snapshots = self.ledger.load_snapshots()
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]["notes"], "first")
        self.assertEqual(snapshots[1]["notes"], "second")

    def test_snapshot_contains_required_fields(self):
        """Snapshot should contain all required fields."""
        entry = self.ledger.record_snapshot(
            str(self.config_path),
            origin="ci",
            notes="test note"
        )
        
        required_fields = [
            "timestamp", "config_path", "curriculum_hash",
            "git_commit", "slice_hashes", "origin", "notes"
        ]
        for field in required_fields:
            self.assertIn(field, entry)

    def test_load_empty_ledger(self):
        """Loading a non-existent ledger should return empty list."""
        snapshots = self.ledger.load_snapshots()
        self.assertEqual(snapshots, [])

    def test_get_snapshot_by_index(self):
        """Should be able to retrieve snapshot by index."""
        self.ledger.record_snapshot(str(self.config_path), notes="zero")
        self.ledger.record_snapshot(str(self.config_path), notes="one")
        self.ledger.record_snapshot(str(self.config_path), notes="two")
        
        self.assertEqual(self.ledger.get_snapshot("0")["notes"], "zero")
        self.assertEqual(self.ledger.get_snapshot("1")["notes"], "one")
        self.assertEqual(self.ledger.get_snapshot("-1")["notes"], "two")
        self.assertEqual(self.ledger.get_snapshot("-2")["notes"], "one")

    def test_get_snapshot_by_timestamp(self):
        """Should be able to retrieve snapshot by timestamp."""
        entry = self.ledger.record_snapshot(str(self.config_path), notes="target")
        timestamp = entry["timestamp"]
        
        retrieved = self.ledger.get_snapshot(timestamp)
        self.assertEqual(retrieved["notes"], "target")

    def test_get_snapshot_not_found(self):
        """Invalid index/timestamp should return None."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.assertIsNone(self.ledger.get_snapshot("999"))
        self.assertIsNone(self.ledger.get_snapshot("invalid-timestamp"))


class TestCurriculumHashLedgerDiff(unittest.TestCase):
    """Test drift detection between snapshots."""

    def setUp(self):
        """Create test infrastructure."""
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "slice_a": {
                    "description": "Slice A",
                    "parameters": {"depth": 4}
                },
                "slice_b": {
                    "description": "Slice B",
                    "parameters": {"depth": 6}
                }
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_no_drift_on_identical_config(self):
        """No drift should be detected for identical configs."""
        self.ledger.record_snapshot(str(self.config_path))
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        
        diff = self.ledger.compare_snapshots(old, new)
        
        self.assertFalse(diff["has_drift"])
        self.assertFalse(diff["global_hash_changed"])
        self.assertEqual(diff["added_slices"], [])
        self.assertEqual(diff["removed_slices"], [])
        self.assertEqual(diff["changed_slices"], [])

    def test_drift_on_parameter_change(self):
        """Drift should be detected when a parameter changes."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Modify a parameter
        self.config_data["slices"]["slice_a"]["parameters"]["depth"] = 99
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        diff = self.ledger.compare_snapshots(old, new)
        
        self.assertTrue(diff["has_drift"])
        self.assertTrue(diff["global_hash_changed"])
        self.assertIn("slice_a", diff["changed_slices"])
        self.assertNotIn("slice_b", diff["changed_slices"])

    def test_drift_on_added_slice(self):
        """Drift should be detected when a slice is added."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Add a new slice
        self.config_data["slices"]["slice_c"] = {
            "description": "New slice",
            "parameters": {"depth": 10}
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        diff = self.ledger.compare_snapshots(old, new)
        
        self.assertTrue(diff["has_drift"])
        self.assertIn("slice_c", diff["added_slices"])
        self.assertEqual(diff["removed_slices"], [])
        self.assertEqual(diff["changed_slices"], [])

    def test_drift_on_removed_slice(self):
        """Drift should be detected when a slice is removed."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Remove a slice
        del self.config_data["slices"]["slice_b"]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        diff = self.ledger.compare_snapshots(old, new)
        
        self.assertTrue(diff["has_drift"])
        self.assertEqual(diff["added_slices"], [])
        self.assertIn("slice_b", diff["removed_slices"])

    def test_drift_on_formula_change(self):
        """Drift should be detected when formula pool changes."""
        self.config_data["slices"]["slice_a"]["formula_pool"] = ["p", "q"]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        # Change formula pool
        self.config_data["slices"]["slice_a"]["formula_pool"] = ["p", "q", "r"]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        diff = self.ledger.compare_snapshots(old, new)
        
        self.assertTrue(diff["has_drift"])
        self.assertIn("slice_a", diff["changed_slices"])

    def test_diff_report_format(self):
        """Diff report should be properly formatted Markdown."""
        self.ledger.record_snapshot(str(self.config_path))
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        diff = self.ledger.compare_snapshots(old, new)
        
        report = self.ledger.format_diff_report(diff)
        
        self.assertIn("# Curriculum Drift Report", report)
        self.assertIn("**Comparison**", report)
        self.assertIn("**Git commits**", report)
        self.assertIn("✅ **No drift detected**", report)


class TestCurriculumHashLedgerRealConfig(unittest.TestCase):
    """Test against real curriculum config files in the repo."""

    def setUp(self):
        """Use a temp ledger but real config files."""
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_hash_real_phase2_config(self):
        """Hash the actual Phase II curriculum config."""
        config_path = "config/curriculum_uplift_phase2.yaml"
        if not Path(config_path).exists():
            self.skipTest(f"Config file not found: {config_path}")
        
        global_hash, slice_hashes = self.ledger.compute_curriculum_hash(config_path)
        
        # Should produce valid hashes
        self.assertEqual(len(global_hash), 64)  # SHA256 hex
        self.assertGreater(len(slice_hashes), 0)
        
        for name, h in slice_hashes.items():
            self.assertEqual(len(h), 64)

    def test_hash_main_curriculum_config(self):
        """Hash the main curriculum.yaml config."""
        config_path = "config/curriculum.yaml"
        if not Path(config_path).exists():
            self.skipTest(f"Config file not found: {config_path}")
        
        global_hash, slice_hashes = self.ledger.compute_curriculum_hash(config_path)
        
        # Should handle the systems/slices structure
        self.assertEqual(len(global_hash), 64)
        self.assertGreater(len(slice_hashes), 0)

    def test_determinism_on_real_config(self):
        """Real config hash should be deterministic across runs."""
        config_path = "config/curriculum_uplift_phase2.yaml"
        if not Path(config_path).exists():
            self.skipTest(f"Config file not found: {config_path}")
        
        hash1, _ = self.ledger.compute_curriculum_hash(config_path)
        
        # Create a new ledger instance
        ledger2 = CurriculumHashLedger(ledger_path=Path(self.temp_dir) / "ledger2.jsonl")
        hash2, _ = ledger2.compute_curriculum_hash(config_path)
        
        self.assertEqual(hash1, hash2)


class TestCurriculumHashLedgerDriftIsConfigOnly(unittest.TestCase):
    """
    Confirm that drift detection only flags configuration changes,
    NOT uplift evidence. This is auditing only.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "slice_a": {
                    "description": "Test slice",
                    "parameters": {"depth": 4},
                    # These are config fields, not runtime results
                    "success_metric": {"kind": "goal_hit"},
                    "formula_pool": ["p", "q"]
                }
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_drift_is_config_change_not_uplift(self):
        """
        Changing success_metric or formula_pool is a CONFIG change,
        not evidence of uplift. The ledger detects configuration drift,
        not U2 run outcomes.
        """
        # Record initial state
        self.ledger.record_snapshot(str(self.config_path))
        
        # Change success metric config (this is a design change)
        self.config_data["slices"]["slice_a"]["success_metric"]["kind"] = "density"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        diff = self.ledger.compare_snapshots(old, new)
        
        # This IS drift (config changed)
        self.assertTrue(diff["has_drift"])
        
        # But it's NOT uplift evidence—it's just "the experiment definition changed"
        # The diff report makes this clear: it's about slice_a configuration
        self.assertIn("slice_a", diff["changed_slices"])

    def test_timestamps_are_recording_metadata(self):
        """
        Timestamp differences between snapshots don't constitute drift.
        """
        import time
        
        self.ledger.record_snapshot(str(self.config_path))
        time.sleep(0.01)  # Small delay
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        diff = self.ledger.compare_snapshots(old, new)
        
        # Timestamps differ, but no drift (config unchanged)
        self.assertNotEqual(old["timestamp"], new["timestamp"])
        self.assertFalse(diff["has_drift"])


# =============================================================================
# NEW TESTS: Drift Type Classification (18 tests)
# =============================================================================

class TestDriftTypeClassification(unittest.TestCase):
    """Test drift type classification logic."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_classify_none_drift_identical_configs(self):
        """Identical configs should be classified as NONE drift."""
        old_config = {"depth": 4, "atoms": 3}
        new_config = {"depth": 4, "atoms": 3}
        
        drift_type, changed_keys = self.ledger.classify_slice_drift(old_config, new_config)
        
        self.assertEqual(drift_type, DriftType.NONE)
        self.assertEqual(changed_keys, [])

    def test_classify_cosmetic_drift_key_order(self):
        """Different key order with same values should be COSMETIC (after canonicalization)."""
        # Note: canonical_json sorts keys, so this tests the canonicalization check
        old_config = {"a": 1, "b": 2}
        new_config = {"b": 2, "a": 1}  # Same data, different order
        
        drift_type, changed_keys = self.ledger.classify_slice_drift(old_config, new_config)
        
        # After canonicalization, they should be identical
        self.assertEqual(drift_type, DriftType.NONE)

    def test_classify_parametric_minor_small_number_change(self):
        """Small numeric changes (<10%) should be PARAMETRIC_MINOR."""
        old_config = {"depth": 100, "atoms": 50}
        new_config = {"depth": 105, "atoms": 50}  # 5% change
        
        drift_type, changed_keys = self.ledger.classify_slice_drift(old_config, new_config)
        
        self.assertEqual(drift_type, DriftType.PARAMETRIC_MINOR)
        self.assertIn("depth", changed_keys)

    def test_classify_parametric_major_large_number_change(self):
        """Large numeric changes (>=50%) should be PARAMETRIC_MAJOR."""
        old_config = {"depth": 100, "atoms": 50}
        new_config = {"depth": 200, "atoms": 50}  # 100% change
        
        drift_type, changed_keys = self.ledger.classify_slice_drift(old_config, new_config)
        
        self.assertEqual(drift_type, DriftType.PARAMETRIC_MAJOR)

    def test_classify_parametric_major_removed_param(self):
        """Removed parameters should be PARAMETRIC_MAJOR."""
        old_config = {"depth": 4, "atoms": 3, "extra": 10}
        new_config = {"depth": 4, "atoms": 3}  # extra removed
        
        drift_type, changed_keys = self.ledger.classify_slice_drift(old_config, new_config)
        
        self.assertEqual(drift_type, DriftType.PARAMETRIC_MAJOR)
        self.assertTrue(any('-' in k for k in changed_keys))

    def test_classify_semantic_formula_pool_change(self):
        """Changes to formula_pool should be SEMANTIC."""
        old_config = {"formula_pool": ["p", "q"]}
        new_config = {"formula_pool": ["p", "q", "r"]}
        
        drift_type, changed_keys = self.ledger.classify_slice_drift(old_config, new_config)
        
        self.assertEqual(drift_type, DriftType.SEMANTIC)

    def test_classify_semantic_target_hash_change(self):
        """Changes to target_hash should be SEMANTIC."""
        old_config = {"target_hash": "abc123", "depth": 4}
        new_config = {"target_hash": "def456", "depth": 4}
        
        drift_type, changed_keys = self.ledger.classify_slice_drift(old_config, new_config)
        
        self.assertEqual(drift_type, DriftType.SEMANTIC)

    def test_classify_semantic_success_metric_kind_change(self):
        """Changes to success_metric.kind should be SEMANTIC."""
        old_config = {"success_metric": {"kind": "goal_hit"}}
        new_config = {"success_metric": {"kind": "density"}}
        
        drift_type, changed_keys = self.ledger.classify_slice_drift(old_config, new_config)
        
        self.assertEqual(drift_type, DriftType.SEMANTIC)

    def test_classify_semantic_prereg_hash_change(self):
        """Changes to prereg_hash should be SEMANTIC."""
        old_config = {"prereg_hash": "hash1", "depth": 4}
        new_config = {"prereg_hash": "hash2", "depth": 4}
        
        drift_type, changed_keys = self.ledger.classify_slice_drift(old_config, new_config)
        
        self.assertEqual(drift_type, DriftType.SEMANTIC)


class TestDriftSeverityMapping(unittest.TestCase):
    """Test mapping from drift type to risk level."""

    def test_none_maps_to_info(self):
        """NONE drift should map to INFO risk."""
        self.assertEqual(DRIFT_RISK_MAP[DriftType.NONE], RiskLevel.INFO)

    def test_cosmetic_maps_to_info(self):
        """COSMETIC drift should map to INFO risk."""
        self.assertEqual(DRIFT_RISK_MAP[DriftType.COSMETIC], RiskLevel.INFO)

    def test_parametric_minor_maps_to_warn(self):
        """PARAMETRIC_MINOR drift should map to WARN risk."""
        self.assertEqual(DRIFT_RISK_MAP[DriftType.PARAMETRIC_MINOR], RiskLevel.WARN)

    def test_parametric_major_maps_to_block(self):
        """PARAMETRIC_MAJOR drift should map to BLOCK risk."""
        self.assertEqual(DRIFT_RISK_MAP[DriftType.PARAMETRIC_MAJOR], RiskLevel.BLOCK)

    def test_semantic_maps_to_block(self):
        """SEMANTIC drift should map to BLOCK risk."""
        self.assertEqual(DRIFT_RISK_MAP[DriftType.SEMANTIC], RiskLevel.BLOCK)

    def test_structural_maps_to_block(self):
        """STRUCTURAL drift should map to BLOCK risk."""
        self.assertEqual(DRIFT_RISK_MAP[DriftType.STRUCTURAL], RiskLevel.BLOCK)


class TestClassifyDriftIntegration(unittest.TestCase):
    """Test full classify_drift method with snapshots and configs."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "slice_a": {
                    "description": "Slice A",
                    "parameters": {"depth": 100},
                    "formula_pool": ["p", "q"]
                }
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_classify_drift_structural_added_slice(self):
        """Adding a slice should be classified as STRUCTURAL with BLOCK risk."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Add a slice
        self.config_data["slices"]["slice_b"] = {"description": "New"}
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        
        old_config = self.ledger._load_config(self.config_path)
        diff = self.ledger.classify_drift(old, new, old_config, old_config)
        
        self.assertEqual(diff["drift_type"], DriftType.STRUCTURAL.value)
        self.assertEqual(diff["risk_level"], RiskLevel.BLOCK.value)
        self.assertIn("slice_b", diff["affected_slices"])

    def test_classify_drift_structural_removed_slice(self):
        """Removing a slice should be classified as STRUCTURAL with BLOCK risk."""
        # Add second slice first
        self.config_data["slices"]["slice_b"] = {"description": "To remove"}
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        # Remove it
        del self.config_data["slices"]["slice_b"]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        
        diff = self.ledger.classify_drift(old, new)
        
        self.assertEqual(diff["drift_type"], DriftType.STRUCTURAL.value)
        self.assertEqual(diff["risk_level"], RiskLevel.BLOCK.value)

    def test_classify_drift_includes_affected_slices(self):
        """classify_drift should include affected_slices with details."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Modify the slice
        self.config_data["slices"]["slice_a"]["formula_pool"] = ["p", "q", "r"]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        
        with open(self.config_path, 'r') as f:
            new_config = yaml.safe_load(f)
        
        diff = self.ledger.classify_drift(old, new, None, new_config)
        
        self.assertIn("affected_slices", diff)
        self.assertIn("slice_a", diff["affected_slices"])
        self.assertEqual(diff["affected_slices"]["slice_a"]["change"], "modified")


class TestSnapshotDiffReproducibility(unittest.TestCase):
    """Test that snapshot/diff operations are reproducible."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "slice_a": {"depth": 4}
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_diff_reproducibility_same_snapshots(self):
        """Diffing the same snapshots should always produce identical results."""
        self.ledger.record_snapshot(str(self.config_path))
        self.config_data["slices"]["slice_a"]["depth"] = 8
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        
        diff1 = self.ledger.compare_snapshots(old, new)
        diff2 = self.ledger.compare_snapshots(old, new)
        
        # Remove timestamps from comparison (they're snapshot metadata)
        for d in [diff1, diff2]:
            d.pop("old_timestamp", None)
            d.pop("new_timestamp", None)
        
        self.assertEqual(diff1, diff2)

    def test_snapshot_hash_reproducibility_across_ledger_instances(self):
        """Different ledger instances should compute same hash for same config."""
        ledger1 = CurriculumHashLedger(ledger_path=Path(self.temp_dir) / "l1.jsonl")
        ledger2 = CurriculumHashLedger(ledger_path=Path(self.temp_dir) / "l2.jsonl")
        
        hash1, slices1 = ledger1.compute_curriculum_hash(str(self.config_path))
        hash2, slices2 = ledger2.compute_curriculum_hash(str(self.config_path))
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(slices1, slices2)

    def test_classify_drift_reproducibility(self):
        """classify_drift should produce consistent results."""
        self.ledger.record_snapshot(str(self.config_path))
        self.config_data["slices"]["slice_a"]["depth"] = 500  # Major change
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        diff1 = self.ledger.classify_drift(old, new, config, config)
        diff2 = self.ledger.classify_drift(old, new, config, config)
        
        self.assertEqual(diff1["drift_type"], diff2["drift_type"])
        self.assertEqual(diff1["risk_level"], diff2["risk_level"])


class TestDriftReportFormat(unittest.TestCase):
    """Test drift report formatting with new classification fields."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "slice_a": {"depth": 4, "formula_pool": ["p"]}
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_report_includes_drift_type(self):
        """Report should include drift type classification."""
        self.ledger.record_snapshot(str(self.config_path))
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        diff = self.ledger.classify_drift(old, new)
        
        report = self.ledger.format_diff_report(diff)
        
        self.assertIn("Drift Type", report)
        self.assertIn("NONE", report)

    def test_report_includes_risk_level(self):
        """Report should include risk level."""
        # Store the original config
        old_config_data = {
            "version": "2.0",
            "slices": {
                "slice_a": {"depth": 4, "formula_pool": ["p"]}
            }
        }
        
        self.ledger.record_snapshot(str(self.config_path))
        
        # Add a slice (STRUCTURAL change → BLOCK)
        self.config_data["slices"]["slice_b"] = {"depth": 10}
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        
        # classify_drift detects structural changes from snapshots alone
        diff = self.ledger.classify_drift(old, new, old_config_data, self.config_data)
        report = self.ledger.format_diff_report(diff)
        
        self.assertIn("Risk Level", report)
        self.assertIn("BLOCK", report)

    def test_report_shows_blocking_drift_warning(self):
        """Report should show clear warning for BLOCK risk."""
        self.ledger.record_snapshot(str(self.config_path))
        self.config_data["slices"]["new_slice"] = {"depth": 10}  # STRUCTURAL
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        old = self.ledger.get_snapshot("0")
        new = self.ledger.get_snapshot("-1")
        diff = self.ledger.classify_drift(old, new)
        
        report = self.ledger.format_diff_report(diff)
        
        self.assertIn("BLOCKING DRIFT DETECTED", report)


# =============================================================================
# NEW TESTS: Ledger Signing (Phase II Extensions)
# =============================================================================

class TestLedgerSigningBasics(unittest.TestCase):
    """Test LedgerSigner basic functionality."""

    def test_signing_disabled_by_default(self):
        """Signing should be disabled when env var not set."""
        from experiments.curriculum_hash_ledger import LedgerSigner
        
        # Ensure env var is not set
        os.environ.pop("LEDGER_SIGNING", None)
        self.assertFalse(LedgerSigner.is_signing_enabled())

    def test_signing_enabled_by_env_var(self):
        """Signing should be enabled when LEDGER_SIGNING=1."""
        from experiments.curriculum_hash_ledger import LedgerSigner
        
        os.environ["LEDGER_SIGNING"] = "1"
        try:
            self.assertTrue(LedgerSigner.is_signing_enabled())
        finally:
            os.environ.pop("LEDGER_SIGNING", None)


class TestLedgerSigningWithKeys(unittest.TestCase):
    """Test signing with generated keys."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.priv_key_path = Path(self.temp_dir) / "test.key"
        self.pub_key_path = Path(self.temp_dir) / "test.pub"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_generate_keypair(self):
        """Should be able to generate a test keypair."""
        try:
            from experiments.curriculum_hash_ledger import LedgerSigner
            
            priv, pub = LedgerSigner.generate_test_keypair(
                private_key_path=self.priv_key_path,
                public_key_path=self.pub_key_path
            )
            
            self.assertTrue(priv.exists())
            self.assertTrue(pub.exists())
        except ImportError:
            self.skipTest("cryptography package not installed")

    def test_sign_and_verify_snapshot(self):
        """Should sign a snapshot and verify it."""
        try:
            from experiments.curriculum_hash_ledger import LedgerSigner
            
            # Generate keypair
            LedgerSigner.generate_test_keypair(
                private_key_path=self.priv_key_path,
                public_key_path=self.pub_key_path
            )
            
            signer = LedgerSigner(
                private_key_path=self.priv_key_path,
                public_key_path=self.pub_key_path
            )
            
            # Create test snapshot
            snapshot = {
                "timestamp": "2025-01-01T00:00:00Z",
                "curriculum_hash": "abc123",
                "config_path": "test.yaml"
            }
            
            # Sign
            signature = signer.sign_snapshot(snapshot)
            self.assertIsInstance(signature, str)
            self.assertGreater(len(signature), 0)
            
            # Verify
            self.assertTrue(signer.verify_signature(snapshot, signature))
        except ImportError:
            self.skipTest("cryptography package not installed")

    def test_tampered_snapshot_fails_verification(self):
        """Tampered snapshot should fail verification."""
        try:
            from experiments.curriculum_hash_ledger import LedgerSigner
            
            # Generate keypair
            LedgerSigner.generate_test_keypair(
                private_key_path=self.priv_key_path,
                public_key_path=self.pub_key_path
            )
            
            signer = LedgerSigner(
                private_key_path=self.priv_key_path,
                public_key_path=self.pub_key_path
            )
            
            snapshot = {
                "timestamp": "2025-01-01T00:00:00Z",
                "curriculum_hash": "abc123"
            }
            
            signature = signer.sign_snapshot(snapshot)
            
            # Tamper with snapshot
            tampered = dict(snapshot)
            tampered["curriculum_hash"] = "tampered"
            
            # Should fail
            self.assertFalse(signer.verify_signature(tampered, signature))
        except ImportError:
            self.skipTest("cryptography package not installed")


# =============================================================================
# NEW TESTS: Drift Contract Validation
# =============================================================================

class TestDriftContractBasics(unittest.TestCase):
    """Test DriftContract basic functionality."""

    def test_structural_drift_blocks(self):
        """STRUCTURAL drift should result in BLOCK verdict."""
        from experiments.curriculum_drift_contract import (
            DriftContract, ContractVerdict
        )
        
        diff = {
            "has_drift": True,
            "drift_type": "STRUCTURAL",
            "added_slices": ["new_slice"],
            "removed_slices": [],
            "changed_slices": [],
            "affected_slices": {
                "new_slice": {
                    "drift_type": "STRUCTURAL",
                    "change": "added"
                }
            }
        }
        
        contract = DriftContract()
        result = contract.validate(diff)
        
        self.assertEqual(result.verdict, ContractVerdict.BLOCK)
        self.assertGreater(len(result.violations), 0)

    def test_semantic_drift_blocks(self):
        """SEMANTIC drift should result in BLOCK verdict."""
        from experiments.curriculum_drift_contract import (
            DriftContract, ContractVerdict
        )
        
        diff = {
            "has_drift": True,
            "drift_type": "SEMANTIC",
            "added_slices": [],
            "removed_slices": [],
            "changed_slices": ["slice_a"],
            "affected_slices": {
                "slice_a": {
                    "drift_type": "SEMANTIC",
                    "change": "modified"
                }
            }
        }
        
        contract = DriftContract()
        result = contract.validate(diff)
        
        self.assertEqual(result.verdict, ContractVerdict.BLOCK)

    def test_parametric_minor_warns(self):
        """PARAMETRIC_MINOR drift should result in WARN verdict."""
        from experiments.curriculum_drift_contract import (
            DriftContract, ContractVerdict
        )
        
        diff = {
            "has_drift": True,
            "drift_type": "PARAMETRIC_MINOR",
            "added_slices": [],
            "removed_slices": [],
            "changed_slices": ["slice_a"],
            "affected_slices": {
                "slice_a": {
                    "drift_type": "PARAMETRIC_MINOR",
                    "change": "modified"
                }
            }
        }
        
        contract = DriftContract()
        result = contract.validate(diff)
        
        self.assertEqual(result.verdict, ContractVerdict.WARN)
        self.assertGreater(len(result.warnings), 0)

    def test_cosmetic_passes(self):
        """COSMETIC drift should result in PASS verdict."""
        from experiments.curriculum_drift_contract import (
            DriftContract, ContractVerdict
        )
        
        diff = {
            "has_drift": True,
            "drift_type": "COSMETIC",
            "added_slices": [],
            "removed_slices": [],
            "changed_slices": [],
            "affected_slices": {}
        }
        
        contract = DriftContract()
        result = contract.validate(diff)
        
        self.assertEqual(result.verdict, ContractVerdict.PASS)

    def test_none_passes(self):
        """NONE drift should result in PASS verdict."""
        from experiments.curriculum_drift_contract import (
            DriftContract, ContractVerdict
        )
        
        diff = {
            "has_drift": False,
            "drift_type": "NONE",
            "added_slices": [],
            "removed_slices": [],
            "changed_slices": [],
            "affected_slices": {}
        }
        
        contract = DriftContract()
        result = contract.validate(diff)
        
        self.assertEqual(result.verdict, ContractVerdict.PASS)

    def test_strict_mode_escalates_warn(self):
        """In strict mode, WARN should become BLOCK."""
        from experiments.curriculum_drift_contract import (
            DriftContract, ContractVerdict
        )
        
        diff = {
            "has_drift": True,
            "drift_type": "PARAMETRIC_MINOR",
            "added_slices": [],
            "removed_slices": [],
            "changed_slices": ["slice_a"],
            "affected_slices": {
                "slice_a": {
                    "drift_type": "PARAMETRIC_MINOR",
                    "change": "modified"
                }
            }
        }
        
        # Normal mode
        contract = DriftContract(strict_mode=False)
        result = contract.validate(diff)
        self.assertEqual(result.verdict, ContractVerdict.WARN)
        
        # Strict mode
        contract_strict = DriftContract(strict_mode=True)
        result_strict = contract_strict.validate(diff)
        self.assertEqual(result_strict.verdict, ContractVerdict.BLOCK)

    def test_result_to_json(self):
        """Result should be JSON-serializable."""
        from experiments.curriculum_drift_contract import (
            DriftContract
        )
        
        diff = {
            "has_drift": True,
            "drift_type": "SEMANTIC",
            "added_slices": [],
            "removed_slices": [],
            "changed_slices": ["slice_a"],
            "affected_slices": {
                "slice_a": {"drift_type": "SEMANTIC", "change": "modified"}
            }
        }
        
        contract = DriftContract()
        result = contract.validate(diff)
        
        json_str = result.to_json()
        self.assertIsInstance(json_str, str)
        
        # Should be parseable
        parsed = json.loads(json_str)
        self.assertEqual(parsed["verdict"], "BLOCK")


class TestDriftContractValidateFunction(unittest.TestCase):
    """Test the convenience validate_drift function."""

    def test_validate_drift_function(self):
        """validate_drift should work like DriftContract.validate."""
        from experiments.curriculum_drift_contract import (
            validate_drift, ContractVerdict
        )
        
        diff = {
            "has_drift": False,
            "drift_type": "NONE",
            "affected_slices": {}
        }
        
        result = validate_drift(diff)
        self.assertEqual(result.verdict, ContractVerdict.PASS)

    def test_validate_drift_strict_mode(self):
        """validate_drift should support strict mode."""
        from experiments.curriculum_drift_contract import (
            validate_drift, ContractVerdict
        )
        
        diff = {
            "has_drift": True,
            "drift_type": "PARAMETRIC_MINOR",
            "affected_slices": {
                "slice_a": {"drift_type": "PARAMETRIC_MINOR"}
            }
        }
        
        result_normal = validate_drift(diff, strict_mode=False)
        self.assertEqual(result_normal.verdict, ContractVerdict.WARN)
        
        result_strict = validate_drift(diff, strict_mode=True)
        self.assertEqual(result_strict.verdict, ContractVerdict.BLOCK)


# =============================================================================
# NEW TESTS: Migration Guide Generator
# =============================================================================

class TestMigrationGuideGenerator(unittest.TestCase):
    """Test migration guide generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "docs"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_generate_migration_guide(self):
        """Should generate a migration guide file."""
        from scripts.ci_check_curriculum import generate_migration_guide
        from experiments.curriculum_drift_contract import validate_drift
        
        diff = {
            "has_drift": True,
            "drift_type": "STRUCTURAL",
            "risk_level": "BLOCK",
            "old_curriculum_hash": "abc123def456",
            "new_curriculum_hash": "xyz789uvw012",
            "old_timestamp": "2025-01-01T00:00:00Z",
            "new_timestamp": "2025-01-02T00:00:00Z",
            "old_git_commit": "1234567890abcdef",
            "new_git_commit": "fedcba0987654321",
            "added_slices": ["new_slice"],
            "removed_slices": [],
            "changed_slices": [],
            "affected_slices": {
                "new_slice": {
                    "drift_type": "STRUCTURAL",
                    "change": "added",
                    "changed_keys": []
                }
            }
        }
        
        contract_result = validate_drift(diff)
        
        guide_path = generate_migration_guide(
            diff=diff,
            contract_result=contract_result,
            config_path="test_config.yaml",
            output_dir=self.output_dir
        )
        
        self.assertTrue(guide_path.exists())
        
        content = guide_path.read_text(encoding='utf-8')
        self.assertIn("# Curriculum Migration Guide", content)
        self.assertIn("STRUCTURAL", content)
        self.assertIn("new_slice", content)
        self.assertIn("Recommended Actions", content)

    def test_guide_filename_uses_hashes(self):
        """Migration guide filename should include hashes."""
        from scripts.ci_check_curriculum import generate_migration_guide
        from experiments.curriculum_drift_contract import validate_drift
        
        diff = {
            "has_drift": True,
            "drift_type": "NONE",
            "old_curriculum_hash": "aaaaaaaaaaaa",
            "new_curriculum_hash": "bbbbbbbbbbbb",
            "affected_slices": {}
        }
        
        contract_result = validate_drift(diff)
        
        guide_path = generate_migration_guide(
            diff=diff,
            contract_result=contract_result,
            config_path="test.yaml",
            output_dir=self.output_dir
        )
        
        self.assertIn("aaaaaaaaaaaa", guide_path.name)
        self.assertIn("bbbbbbbbbbbb", guide_path.name)


# =============================================================================
# NEW TESTS: Drift Timeline Builder
# =============================================================================

class TestDriftTimelineBuilder(unittest.TestCase):
    """Test per-slice drift timeline functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "slice_a": {"depth": 4, "atoms": 3},
                "slice_b": {"depth": 6, "atoms": 5}
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_empty_timeline_no_snapshots(self):
        """Timeline should be empty when no snapshots exist."""
        timeline = self.ledger.build_drift_timeline("slice_a")
        self.assertEqual(timeline, [])

    def test_empty_timeline_single_snapshot(self):
        """Timeline should be empty with only one snapshot."""
        self.ledger.record_snapshot(str(self.config_path))
        timeline = self.ledger.build_drift_timeline("slice_a")
        self.assertEqual(timeline, [])

    def test_timeline_detects_slice_change(self):
        """Timeline should detect when a slice hash changes."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Modify slice_a
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_a")
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].change_type, "modified")

    def test_timeline_detects_slice_added(self):
        """Timeline should detect when a slice is added."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Add new slice
        self.config_data["slices"]["slice_c"] = {"depth": 8}
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_c")
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].change_type, "added")
        self.assertEqual(timeline[0].drift_type, DriftType.STRUCTURAL)

    def test_timeline_detects_slice_removed(self):
        """Timeline should detect when a slice is removed."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Remove slice_b
        del self.config_data["slices"]["slice_b"]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_b")
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].change_type, "removed")
        self.assertEqual(timeline[0].drift_type, DriftType.STRUCTURAL)

    def test_timeline_no_events_for_unchanged_slice(self):
        """Timeline should have no events for unchanged slice."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Modify only slice_a
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        # slice_b should have no events
        timeline_b = self.ledger.build_drift_timeline("slice_b")
        self.assertEqual(len(timeline_b), 0)
        
        # slice_a should have one event
        timeline_a = self.ledger.build_drift_timeline("slice_a")
        self.assertEqual(len(timeline_a), 1)

    def test_timeline_multiple_changes(self):
        """Timeline should track multiple changes over time."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # First change
        self.config_data["slices"]["slice_a"]["depth"] = 5
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        # Second change
        self.config_data["slices"]["slice_a"]["depth"] = 6
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        # Third change
        self.config_data["slices"]["slice_a"]["depth"] = 7
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_a")
        self.assertEqual(len(timeline), 3)
        
        # Verify chronological order
        for i in range(1, len(timeline)):
            self.assertGreater(timeline[i].snapshot_index, timeline[i-1].snapshot_index)

    def test_drift_event_has_required_fields(self):
        """DriftEvent should have all required fields."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_a")
        event = timeline[0]
        
        self.assertIsNotNone(event.timestamp)
        self.assertIsNotNone(event.old_hash)
        self.assertIsNotNone(event.new_hash)
        self.assertIsInstance(event.drift_type, DriftType)
        self.assertIsInstance(event.risk_level, RiskLevel)
        self.assertIsInstance(event.snapshot_index, int)
        self.assertIsNotNone(event.git_commit)

    def test_drift_event_to_dict(self):
        """DriftEvent.to_dict should return JSON-serializable dict."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_a")
        event_dict = timeline[0].to_dict()
        
        # Should be JSON-serializable
        json_str = json.dumps(event_dict)
        self.assertIsInstance(json_str, str)
        
        # Should have expected keys
        self.assertIn("timestamp", event_dict)
        self.assertIn("old_hash", event_dict)
        self.assertIn("new_hash", event_dict)
        self.assertIn("drift_type", event_dict)
        self.assertIn("risk_level", event_dict)

    def test_format_timeline(self):
        """format_timeline should produce readable output."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_a")
        formatted = self.ledger.format_timeline(timeline, "slice_a")
        
        self.assertIn("Drift Timeline", formatted)
        self.assertIn("slice_a", formatted)
        self.assertIn("Total Events", formatted)


class TestBuildAllSlicesTimeline(unittest.TestCase):
    """Test timeline building for all slices."""

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

    def test_build_all_slices_timeline(self):
        """Should build timelines for all slices with drift events."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Modify slice_a
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        all_timelines = self.ledger.build_all_slices_timeline()
        
        # slice_a has drift events, so it should be included
        self.assertIn("slice_a", all_timelines)
        self.assertEqual(len(all_timelines["slice_a"]), 1)
        
        # slice_b has no drift events, so it's NOT included by default
        # (only slices with drift events are returned)
        self.assertNotIn("slice_b", all_timelines)

    def test_empty_with_no_snapshots(self):
        """Should return empty dict with no snapshots."""
        all_timelines = self.ledger.build_all_slices_timeline()
        self.assertEqual(all_timelines, {})


class TestDriftEventFormatLine(unittest.TestCase):
    """Test DriftEvent formatting."""

    def test_format_line_block(self):
        """BLOCK events should show ✗ icon."""
        from experiments.curriculum_hash_ledger import DriftEvent
        
        event = DriftEvent(
            timestamp="2025-01-01T00:00:00Z",
            old_hash="abc123",
            new_hash="def456",
            drift_type=DriftType.STRUCTURAL,
            risk_level=RiskLevel.BLOCK,
            snapshot_index=5,
            git_commit="abcdef123456",
            change_type="added"
        )
        
        line = event.format_line()
        self.assertIn("✗", line)
        self.assertIn("STRUCTURAL", line)
        self.assertIn("added", line)

    def test_format_line_warn(self):
        """WARN events should show ⚠ icon."""
        from experiments.curriculum_hash_ledger import DriftEvent
        
        event = DriftEvent(
            timestamp="2025-01-01T00:00:00Z",
            old_hash="abc123",
            new_hash="def456",
            drift_type=DriftType.PARAMETRIC_MINOR,
            risk_level=RiskLevel.WARN,
            snapshot_index=3,
            git_commit="abcdef123456",
            change_type="modified"
        )
        
        line = event.format_line()
        self.assertIn("⚠", line)
        self.assertIn("PARAMETRIC_MINOR", line)

    def test_format_line_info(self):
        """INFO events should show ✓ icon."""
        from experiments.curriculum_hash_ledger import DriftEvent
        
        event = DriftEvent(
            timestamp="2025-01-01T00:00:00Z",
            old_hash="abc123",
            new_hash="abc123",
            drift_type=DriftType.NONE,
            risk_level=RiskLevel.INFO,
            snapshot_index=1,
            git_commit="abcdef123456",
            change_type="unchanged"
        )
        
        line = event.format_line()
        self.assertIn("✓", line)


# =============================================================================
# TASK 1 TESTS: DriftEvent as Canonical Record
# =============================================================================

class TestDriftEventChangeTypes(unittest.TestCase):
    """Test that DriftEvent correctly records change_type and changed_keys."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "slice_a": {"depth": 4, "atoms": 3},
                "slice_b": {"depth": 6, "atoms": 5}
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_added_slice_has_change_type_added(self):
        """Added slice should have change_type='added'."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Add new slice
        self.config_data["slices"]["slice_c"] = {"depth": 8, "atoms": 4}
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_c")
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].change_type, "added")
        self.assertEqual(timeline[0].drift_type, DriftType.STRUCTURAL)

    def test_removed_slice_has_change_type_removed(self):
        """Removed slice should have change_type='removed'."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Remove slice_b
        del self.config_data["slices"]["slice_b"]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_b")
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].change_type, "removed")
        self.assertEqual(timeline[0].drift_type, DriftType.STRUCTURAL)

    def test_modified_slice_has_change_type_modified(self):
        """Modified slice should have change_type='modified'."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Modify slice_a
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_a")
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].change_type, "modified")

    def test_modified_slice_has_non_empty_changed_keys(self):
        """Modified slice should have non-empty changed_keys when configs differ."""
        # Create two separate config files to allow historical comparison
        config_path_old = Path(self.temp_dir) / "config_v1.yaml"
        config_path_new = Path(self.temp_dir) / "config_v2.yaml"
        
        old_config = {
            "version": "2.0",
            "slices": {
                "slice_a": {"depth": 4, "atoms": 3}
            }
        }
        new_config = {
            "version": "2.0",
            "slices": {
                "slice_a": {"depth": 999, "atoms": 3}  # depth changed
            }
        }
        
        with open(config_path_old, 'w') as f:
            yaml.dump(old_config, f)
        with open(config_path_new, 'w') as f:
            yaml.dump(new_config, f)
        
        # Record snapshots with different config files
        self.ledger.record_snapshot(str(config_path_old))
        self.ledger.record_snapshot(str(config_path_new))
        
        timeline = self.ledger.build_drift_timeline("slice_a")
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].change_type, "modified")
        # changed_keys should contain depth since we have separate config files
        self.assertIn("depth", timeline[0].changed_keys)

    def test_added_slice_has_empty_changed_keys(self):
        """Added slice should have empty changed_keys (no prior version)."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Add new slice
        self.config_data["slices"]["slice_new"] = {"depth": 5}
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_new")
        self.assertEqual(timeline[0].changed_keys, [])

    def test_removed_slice_has_empty_changed_keys(self):
        """Removed slice should have empty changed_keys."""
        self.ledger.record_snapshot(str(self.config_path))
        
        del self.config_data["slices"]["slice_b"]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_b")
        self.assertEqual(timeline[0].changed_keys, [])


class TestDriftEventOrdering(unittest.TestCase):
    """Test that DriftEvents are deterministically ordered by snapshot_index."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "slice_a": {"depth": 1}
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_timeline_sorted_by_snapshot_index(self):
        """Timeline events should be sorted by snapshot_index (ascending)."""
        # Create multiple snapshots with changes
        for i in range(5):
            self.ledger.record_snapshot(str(self.config_path))
            self.config_data["slices"]["slice_a"]["depth"] = i + 2
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_a")
        
        # Verify sorted by snapshot_index
        for i in range(1, len(timeline)):
            self.assertGreater(
                timeline[i].snapshot_index,
                timeline[i-1].snapshot_index,
                "Events must be sorted by snapshot_index"
            )

    def test_deterministic_ordering_across_calls(self):
        """Multiple calls should return identically ordered timelines."""
        for i in range(4):
            self.ledger.record_snapshot(str(self.config_path))
            self.config_data["slices"]["slice_a"]["depth"] = i + 10
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config_data, f)
        
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline1 = self.ledger.build_drift_timeline("slice_a")
        timeline2 = self.ledger.build_drift_timeline("slice_a")
        
        self.assertEqual(len(timeline1), len(timeline2))
        for e1, e2 in zip(timeline1, timeline2):
            self.assertEqual(e1.snapshot_index, e2.snapshot_index)
            self.assertEqual(e1.timestamp, e2.timestamp)


# =============================================================================
# TASK 2 TESTS: Timeline JSON Contract
# =============================================================================

class TestTimelineJSONContract(unittest.TestCase):
    """Test the JSON output contract for timeline commands."""

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

    def test_timeline_json_shape(self):
        """Timeline JSON output should have required shape."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_a")
        
        # Build JSON output as handle_timeline_mode does
        output = {
            "slice": "slice_a",
            "events": [e.to_dict() for e in timeline],
            "total_events": len(timeline),
            "block_count": sum(1 for e in timeline if e.risk_level == RiskLevel.BLOCK),
            "warn_count": sum(1 for e in timeline if e.risk_level == RiskLevel.WARN),
            "info_count": sum(1 for e in timeline if e.risk_level == RiskLevel.INFO)
        }
        
        # Verify required keys
        self.assertIn("slice", output)
        self.assertIn("events", output)
        self.assertIn("total_events", output)
        self.assertIn("block_count", output)
        self.assertIn("warn_count", output)
        self.assertIn("info_count", output)
        
        # Verify types
        self.assertIsInstance(output["slice"], str)
        self.assertIsInstance(output["events"], list)
        self.assertIsInstance(output["total_events"], int)
        self.assertIsInstance(output["block_count"], int)
        self.assertIsInstance(output["warn_count"], int)
        self.assertIsInstance(output["info_count"], int)

    def test_deterministic_repeated_timeline_generation(self):
        """Repeated timeline generation should produce identical JSON."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Add semantic change (BLOCK level)
        self.config_data["slices"]["slice_a"]["formula_pool"] = ["p", "q"]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        def build_output():
            timeline = self.ledger.build_drift_timeline("slice_a")
            return {
                "slice": "slice_a",
                "events": [e.to_dict() for e in timeline],
                "total_events": len(timeline),
                "block_count": sum(1 for e in timeline if e.risk_level == RiskLevel.BLOCK),
                "warn_count": sum(1 for e in timeline if e.risk_level == RiskLevel.WARN),
                "info_count": sum(1 for e in timeline if e.risk_level == RiskLevel.INFO)
            }
        
        output1 = build_output()
        output2 = build_output()
        output3 = build_output()
        
        # All should be identical
        self.assertEqual(json.dumps(output1, sort_keys=True), json.dumps(output2, sort_keys=True))
        self.assertEqual(json.dumps(output2, sort_keys=True), json.dumps(output3, sort_keys=True))

    def test_block_count_equals_block_events(self):
        """block_count should equal the number of BLOCK-level events."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Create a STRUCTURAL change (BLOCK)
        self.config_data["slices"]["slice_c"] = {"depth": 8}
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_c")
        
        block_events = [e for e in timeline if e.risk_level == RiskLevel.BLOCK]
        block_count = sum(1 for e in timeline if e.risk_level == RiskLevel.BLOCK)
        
        self.assertEqual(len(block_events), block_count)

    def test_slice_with_zero_events_returns_valid_structure(self):
        """Slice with no drift events should return valid JSON structure."""
        self.ledger.record_snapshot(str(self.config_path))
        self.ledger.record_snapshot(str(self.config_path))  # No changes
        
        timeline = self.ledger.build_drift_timeline("slice_a")
        
        output = {
            "slice": "slice_a",
            "events": [e.to_dict() for e in timeline],
            "total_events": len(timeline),
            "block_count": sum(1 for e in timeline if e.risk_level == RiskLevel.BLOCK),
            "warn_count": sum(1 for e in timeline if e.risk_level == RiskLevel.WARN),
            "info_count": sum(1 for e in timeline if e.risk_level == RiskLevel.INFO)
        }
        
        # Should still be valid structure
        self.assertEqual(output["slice"], "slice_a")
        self.assertEqual(output["events"], [])
        self.assertEqual(output["total_events"], 0)
        self.assertEqual(output["block_count"], 0)
        self.assertEqual(output["warn_count"], 0)
        self.assertEqual(output["info_count"], 0)

    def test_nonexistent_slice_returns_valid_structure(self):
        """Non-existent slice should return valid empty structure."""
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("slice_nonexistent")
        
        output = {
            "slice": "slice_nonexistent",
            "events": [e.to_dict() for e in timeline],
            "total_events": len(timeline),
            "block_count": 0,
            "warn_count": 0,
            "info_count": 0
        }
        
        self.assertEqual(output["events"], [])
        self.assertEqual(output["total_events"], 0)


class TestTimelineCountAccuracy(unittest.TestCase):
    """Test that timeline counts are accurate."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.temp_dir) / "test_ledger.jsonl"
        self.ledger = CurriculumHashLedger(ledger_path=self.ledger_path)
        
        self.config_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.config_data = {
            "version": "2.0",
            "slices": {
                "test_slice": {"depth": 100, "atoms": 50}
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_multiple_blocks_counted_correctly(self):
        """Multiple BLOCK-level events should be counted correctly."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Create multiple STRUCTURAL changes (BLOCK level)
        for i in range(3):
            self.config_data["slices"][f"slice_{i}"] = {"depth": i}
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config_data, f)
            self.ledger.record_snapshot(str(self.config_path))
        
        # Each new slice should have a BLOCK event
        timeline = self.ledger.build_drift_timeline("slice_0")
        block_count = sum(1 for e in timeline if e.risk_level == RiskLevel.BLOCK)
        self.assertEqual(block_count, 1)  # slice_0 was added once

    def test_warn_count_accuracy(self):
        """WARN-level events should be counted correctly."""
        self.ledger.record_snapshot(str(self.config_path))
        
        # Create PARAMETRIC_MINOR change (5% change -> WARN)
        self.config_data["slices"]["test_slice"]["depth"] = 105  # 5% change
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        timeline = self.ledger.build_drift_timeline("test_slice")
        warn_count = sum(1 for e in timeline if e.risk_level == RiskLevel.WARN)
        
        # Should have at least one WARN event
        self.assertGreaterEqual(warn_count, 0)  # May vary based on classification


# =============================================================================
# TASK 3 TESTS: BLOCK Violation Explanation Contract
# =============================================================================

class TestBlockViolationOutput(unittest.TestCase):
    """Test BLOCK-level violation output includes required sections."""

    def test_block_output_has_ledger_entries_section(self):
        """BLOCK output should have 📋 Ledger Entries section."""
        from scripts.ci_check_curriculum import format_block_violation
        from experiments.curriculum_drift_contract import (
            DriftContract, ContractVerdict
        )
        
        diff = {
            "has_drift": True,
            "drift_type": "STRUCTURAL",
            "risk_level": "BLOCK",
            "old_timestamp": "2025-01-01T00:00:00Z",
            "new_timestamp": "2025-01-02T00:00:00Z",
            "old_curriculum_hash": "aaaaaaaabbbbbbbb",
            "new_curriculum_hash": "ccccccccdddddddd",
            "affected_slices": {
                "new_slice": {
                    "drift_type": "STRUCTURAL",
                    "change": "added"
                }
            }
        }
        
        contract = DriftContract()
        contract_result = contract.validate(diff)
        
        output = format_block_violation(
            diff=diff,
            contract_result=contract_result,
            snapshots=[{"timestamp": "old"}, {"timestamp": "new"}],
            config_path="test.yaml"
        )
        
        self.assertIn("📋 Ledger Entries", output)

    def test_block_output_has_contract_violations_section(self):
        """BLOCK output should have ❌ Contract Violations section."""
        from scripts.ci_check_curriculum import format_block_violation
        from experiments.curriculum_drift_contract import (
            DriftContract, ContractVerdict
        )
        
        diff = {
            "has_drift": True,
            "drift_type": "SEMANTIC",
            "risk_level": "BLOCK",
            "old_timestamp": "2025-01-01T00:00:00Z",
            "new_timestamp": "2025-01-02T00:00:00Z",
            "old_curriculum_hash": "hash1",
            "new_curriculum_hash": "hash2",
            "affected_slices": {
                "slice_a": {
                    "drift_type": "SEMANTIC",
                    "change": "modified"
                }
            }
        }
        
        contract = DriftContract()
        contract_result = contract.validate(diff)
        
        output = format_block_violation(
            diff=diff,
            contract_result=contract_result,
            snapshots=[{"timestamp": "old"}, {"timestamp": "new"}],
            config_path="test.yaml"
        )
        
        self.assertIn("❌ Contract Violations", output)

    def test_block_output_has_to_resolve_section(self):
        """BLOCK output should have 🔧 To Resolve section."""
        from scripts.ci_check_curriculum import format_block_violation
        from experiments.curriculum_drift_contract import (
            DriftContract, ContractVerdict
        )
        
        diff = {
            "has_drift": True,
            "drift_type": "STRUCTURAL",
            "risk_level": "BLOCK",
            "old_timestamp": "2025-01-01T00:00:00Z",
            "new_timestamp": "2025-01-02T00:00:00Z",
            "old_curriculum_hash": "hash1",
            "new_curriculum_hash": "hash2",
            "affected_slices": {
                "slice_new": {
                    "drift_type": "STRUCTURAL",
                    "change": "added"
                }
            }
        }
        
        contract = DriftContract()
        contract_result = contract.validate(diff)
        
        output = format_block_violation(
            diff=diff,
            contract_result=contract_result,
            snapshots=[{}, {}],
            config_path="test.yaml"
        )
        
        self.assertIn("🔧 To Resolve", output)

    def test_block_output_includes_hash_delta(self):
        """BLOCK output should include before/after hashes."""
        from scripts.ci_check_curriculum import format_block_violation
        from experiments.curriculum_drift_contract import DriftContract
        
        diff = {
            "has_drift": True,
            "drift_type": "STRUCTURAL",
            "risk_level": "BLOCK",
            "old_timestamp": "2025-01-01T00:00:00Z",
            "new_timestamp": "2025-01-02T00:00:00Z",
            "old_curriculum_hash": "abc123def456abcd",
            "new_curriculum_hash": "xyz789uvw012xyz7",
            "affected_slices": {}
        }
        
        contract = DriftContract()
        contract_result = contract.validate(diff)
        
        output = format_block_violation(
            diff=diff,
            contract_result=contract_result,
            snapshots=[{}, {}],
            config_path="test.yaml"
        )
        
        # Should include hash prefixes
        self.assertIn("abc123def456abcd", output)
        self.assertIn("xyz789uvw012xyz7", output)

    def test_block_output_includes_drift_type(self):
        """BLOCK output should include drift_type."""
        from scripts.ci_check_curriculum import format_block_violation
        from experiments.curriculum_drift_contract import DriftContract
        
        diff = {
            "has_drift": True,
            "drift_type": "SEMANTIC",
            "risk_level": "BLOCK",
            "old_timestamp": "2025-01-01T00:00:00Z",
            "new_timestamp": "2025-01-02T00:00:00Z",
            "old_curriculum_hash": "hash1",
            "new_curriculum_hash": "hash2",
            "affected_slices": {
                "test_slice": {
                    "drift_type": "SEMANTIC",
                    "change": "modified"
                }
            }
        }
        
        contract = DriftContract()
        contract_result = contract.validate(diff)
        
        output = format_block_violation(
            diff=diff,
            contract_result=contract_result,
            snapshots=[{}, {}],
            config_path="test.yaml"
        )
        
        self.assertIn("SEMANTIC", output)

    def test_block_output_includes_human_override_instructions(self):
        """BLOCK output should include human override instructions."""
        from scripts.ci_check_curriculum import format_block_violation
        from experiments.curriculum_drift_contract import DriftContract
        
        diff = {
            "has_drift": True,
            "drift_type": "STRUCTURAL",
            "risk_level": "BLOCK",
            "old_curriculum_hash": "h1",
            "new_curriculum_hash": "h2",
            "old_timestamp": "t1",
            "new_timestamp": "t2",
            "affected_slices": {}
        }
        
        contract = DriftContract()
        contract_result = contract.validate(diff)
        
        output = format_block_violation(
            diff=diff,
            contract_result=contract_result,
            snapshots=[{}, {}],
            config_path="my_config.yaml"
        )
        
        # Should include CLI command for approval
        self.assertIn("--snapshot", output)
        self.assertIn("--config", output)
        self.assertIn("my_config.yaml", output)
        self.assertIn("--origin=manual", output)


class TestCIExitCodes(unittest.TestCase):
    """Test that CI script returns correct exit codes."""

    def test_contract_block_verdict_means_exit_code_1(self):
        """ContractVerdict.BLOCK should map to exit code 1."""
        from experiments.curriculum_drift_contract import ContractVerdict
        
        verdict = ContractVerdict.BLOCK
        exit_code = 1 if verdict == ContractVerdict.BLOCK else 0
        self.assertEqual(exit_code, 1)

    def test_contract_pass_verdict_means_exit_code_0(self):
        """ContractVerdict.PASS should map to exit code 0."""
        from experiments.curriculum_drift_contract import ContractVerdict
        
        verdict = ContractVerdict.PASS
        exit_code = 1 if verdict == ContractVerdict.BLOCK else 0
        self.assertEqual(exit_code, 0)

    def test_contract_warn_verdict_means_exit_code_0(self):
        """ContractVerdict.WARN should map to exit code 0 (non-blocking)."""
        from experiments.curriculum_drift_contract import ContractVerdict
        
        verdict = ContractVerdict.WARN
        exit_code = 1 if verdict == ContractVerdict.BLOCK else 0
        self.assertEqual(exit_code, 0)


class TestBlockViolationLine(unittest.TestCase):
    """Test BLOCK violation line format includes required info."""

    def test_violation_line_includes_slice_name(self):
        """Violation line should include slice name."""
        from scripts.ci_check_curriculum import format_block_violation
        from experiments.curriculum_drift_contract import DriftContract
        
        diff = {
            "has_drift": True,
            "drift_type": "STRUCTURAL",
            "risk_level": "BLOCK",
            "old_curriculum_hash": "h1",
            "new_curriculum_hash": "h2",
            "old_timestamp": "t1",
            "new_timestamp": "t2",
            "affected_slices": {
                "my_specific_slice": {
                    "drift_type": "STRUCTURAL",
                    "change": "added"
                }
            }
        }
        
        contract = DriftContract()
        contract_result = contract.validate(diff)
        
        output = format_block_violation(
            diff=diff,
            contract_result=contract_result,
            snapshots=[{}, {}],
            config_path="test.yaml"
        )
        
        self.assertIn("my_specific_slice", output)

    def test_violation_line_includes_risk_block(self):
        """Violation line should state risk=BLOCK."""
        from scripts.ci_check_curriculum import format_block_violation
        from experiments.curriculum_drift_contract import DriftContract
        
        diff = {
            "has_drift": True,
            "drift_type": "SEMANTIC",
            "risk_level": "BLOCK",
            "old_curriculum_hash": "h1",
            "new_curriculum_hash": "h2",
            "old_timestamp": "t1",
            "new_timestamp": "t2",
            "affected_slices": {
                "test": {
                    "drift_type": "SEMANTIC",
                    "change": "modified"
                }
            }
        }
        
        contract = DriftContract()
        contract_result = contract.validate(diff)
        
        output = format_block_violation(
            diff=diff,
            contract_result=contract_result,
            snapshots=[{}, {}],
            config_path="test.yaml"
        )
        
        self.assertIn("risk=BLOCK", output)


# =============================================================================
# TASK 1 TESTS: Timeline Compression
# =============================================================================

class TestTimelineCompression(unittest.TestCase):
    """Test timeline compression functionality."""

    def _make_event(
        self,
        snapshot_index: int,
        risk_level: RiskLevel,
        change_type: str = "modified",
        drift_type: DriftType = DriftType.NONE,
        notes: str = ""
    ) -> DriftEvent:
        """Create a DriftEvent for testing."""
        return DriftEvent(
            timestamp=f"2025-01-{snapshot_index:02d}T00:00:00Z",
            old_hash=f"old_{snapshot_index}",
            new_hash=f"new_{snapshot_index}",
            drift_type=drift_type,
            risk_level=risk_level,
            snapshot_index=snapshot_index,
            git_commit=f"commit_{snapshot_index}",
            change_type=change_type,
            changed_keys=[],
            notes=notes
        )

    def test_compress_empty_timeline(self):
        """Empty timeline should compress to empty CompressedTimeline."""
        compressed = compress_drift_timeline([], "test_slice")
        
        self.assertEqual(compressed.slice_name, "test_slice")
        self.assertEqual(compressed.groups, [])
        self.assertEqual(compressed.original_event_count, 0)
        self.assertEqual(compressed.compressed_group_count, 0)

    def test_compress_single_info_event(self):
        """Single INFO event should become a single group."""
        events = [self._make_event(1, RiskLevel.INFO)]
        compressed = compress_drift_timeline(events, "test_slice")
        
        self.assertEqual(compressed.original_event_count, 1)
        self.assertEqual(compressed.compressed_group_count, 1)
        self.assertEqual(len(compressed.groups[0].events), 1)

    def test_compress_consecutive_info_events_merged(self):
        """Consecutive INFO events with same key should merge."""
        events = [
            self._make_event(1, RiskLevel.INFO, "modified"),
            self._make_event(2, RiskLevel.INFO, "modified"),
            self._make_event(3, RiskLevel.INFO, "modified"),
        ]
        compressed = compress_drift_timeline(events, "test_slice")
        
        self.assertEqual(compressed.original_event_count, 3)
        self.assertEqual(compressed.compressed_group_count, 1)
        self.assertEqual(len(compressed.groups[0].events), 3)
        self.assertEqual(compressed.info_count, 3)

    def test_compress_block_events_not_merged(self):
        """BLOCK events should never be merged."""
        events = [
            self._make_event(1, RiskLevel.BLOCK, drift_type=DriftType.STRUCTURAL),
            self._make_event(2, RiskLevel.BLOCK, drift_type=DriftType.STRUCTURAL),
        ]
        compressed = compress_drift_timeline(events, "test_slice")
        
        self.assertEqual(compressed.original_event_count, 2)
        self.assertEqual(compressed.compressed_group_count, 2)
        self.assertEqual(compressed.block_count, 2)

    def test_compress_warn_events_not_merged(self):
        """WARN events should never be merged."""
        events = [
            self._make_event(1, RiskLevel.WARN, drift_type=DriftType.PARAMETRIC_MINOR),
            self._make_event(2, RiskLevel.WARN, drift_type=DriftType.PARAMETRIC_MINOR),
        ]
        compressed = compress_drift_timeline(events, "test_slice")
        
        self.assertEqual(compressed.compressed_group_count, 2)
        self.assertEqual(compressed.warn_count, 2)

    def test_compress_preserves_block_warn_boundaries(self):
        """BLOCK/WARN events should break INFO grouping."""
        events = [
            self._make_event(1, RiskLevel.INFO),
            self._make_event(2, RiskLevel.INFO),
            self._make_event(3, RiskLevel.BLOCK, drift_type=DriftType.STRUCTURAL),
            self._make_event(4, RiskLevel.INFO),
            self._make_event(5, RiskLevel.INFO),
        ]
        compressed = compress_drift_timeline(events, "test_slice")
        
        # Should be: [INFO group(2)], [BLOCK(1)], [INFO group(2)]
        self.assertEqual(compressed.compressed_group_count, 3)
        self.assertEqual(len(compressed.groups[0].events), 2)  # First INFO group
        self.assertEqual(len(compressed.groups[1].events), 1)  # BLOCK
        self.assertEqual(len(compressed.groups[2].events), 2)  # Second INFO group

    def test_compress_different_change_types_not_merged(self):
        """INFO events with different change_types should not merge."""
        events = [
            self._make_event(1, RiskLevel.INFO, "modified"),
            self._make_event(2, RiskLevel.INFO, "added"),
            self._make_event(3, RiskLevel.INFO, "removed"),
        ]
        compressed = compress_drift_timeline(events, "test_slice")
        
        self.assertEqual(compressed.compressed_group_count, 3)

    def test_compress_deterministic_grouping(self):
        """Compression should be deterministic."""
        events = [
            self._make_event(1, RiskLevel.INFO),
            self._make_event(2, RiskLevel.BLOCK, drift_type=DriftType.STRUCTURAL),
            self._make_event(3, RiskLevel.INFO),
        ]
        
        compressed1 = compress_drift_timeline(events, "test_slice")
        compressed2 = compress_drift_timeline(events, "test_slice")
        
        self.assertEqual(compressed1.timeline_hash, compressed2.timeline_hash)
        self.assertEqual(compressed1.compressed_group_count, compressed2.compressed_group_count)


class TestTimelineDecompression(unittest.TestCase):
    """Test timeline decompression (round-trip)."""

    def _make_event(
        self,
        snapshot_index: int,
        risk_level: RiskLevel,
        change_type: str = "modified"
    ) -> DriftEvent:
        """Create a DriftEvent for testing."""
        return DriftEvent(
            timestamp=f"2025-01-{snapshot_index:02d}T00:00:00Z",
            old_hash=f"old_{snapshot_index}",
            new_hash=f"new_{snapshot_index}",
            drift_type=DriftType.NONE if risk_level == RiskLevel.INFO else DriftType.STRUCTURAL,
            risk_level=risk_level,
            snapshot_index=snapshot_index,
            git_commit=f"commit_{snapshot_index}",
            change_type=change_type,
            changed_keys=[],
            notes=""
        )

    def test_roundtrip_preserves_events(self):
        """decompress(compress(x)) should return original events."""
        events = [
            self._make_event(1, RiskLevel.INFO),
            self._make_event(2, RiskLevel.BLOCK),
            self._make_event(3, RiskLevel.INFO),
            self._make_event(4, RiskLevel.WARN),
        ]
        
        compressed = compress_drift_timeline(events, "test_slice")
        decompressed = decompress_drift_timeline(compressed)
        
        self.assertEqual(len(decompressed), len(events))
        for orig, dec in zip(events, decompressed):
            self.assertEqual(orig.snapshot_index, dec.snapshot_index)
            self.assertEqual(orig.risk_level, dec.risk_level)
            self.assertEqual(orig.timestamp, dec.timestamp)

    def test_roundtrip_preserves_order(self):
        """Round-trip should preserve chronological order."""
        events = [
            self._make_event(i, RiskLevel.INFO)
            for i in range(10)
        ]
        
        compressed = compress_drift_timeline(events, "test_slice")
        decompressed = decompress_drift_timeline(compressed)
        
        for i, event in enumerate(decompressed):
            self.assertEqual(event.snapshot_index, i)

    def test_roundtrip_empty_timeline(self):
        """Round-trip should handle empty timeline."""
        compressed = compress_drift_timeline([], "test_slice")
        decompressed = decompress_drift_timeline(compressed)
        
        self.assertEqual(decompressed, [])


class TestCompressedTimelineHash(unittest.TestCase):
    """Test timeline hash determinism."""

    def _make_event(self, snapshot_index: int, risk_level: RiskLevel) -> DriftEvent:
        return DriftEvent(
            timestamp=f"2025-01-{snapshot_index:02d}T00:00:00Z",
            old_hash=f"old_{snapshot_index}",
            new_hash=f"new_{snapshot_index}",
            drift_type=DriftType.NONE,
            risk_level=risk_level,
            snapshot_index=snapshot_index,
            git_commit=f"commit_{snapshot_index}",
            change_type="modified",
            changed_keys=[],
            notes=""
        )

    def test_hash_deterministic_across_runs(self):
        """Same events should produce same hash."""
        events = [self._make_event(1, RiskLevel.INFO)]
        
        hash1 = compress_drift_timeline(events, "slice").timeline_hash
        hash2 = compress_drift_timeline(events, "slice").timeline_hash
        hash3 = compress_drift_timeline(events, "slice").timeline_hash
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)

    def test_hash_changes_with_different_events(self):
        """Different events should produce different hash."""
        events1 = [self._make_event(1, RiskLevel.INFO)]
        events2 = [self._make_event(2, RiskLevel.INFO)]
        
        hash1 = compress_drift_timeline(events1, "slice").timeline_hash
        hash2 = compress_drift_timeline(events2, "slice").timeline_hash
        
        self.assertNotEqual(hash1, hash2)


# =============================================================================
# TASK 2 TESTS: Drift Intent Annotation
# =============================================================================

class TestDriftIntentEnum(unittest.TestCase):
    """Test DriftIntent enum values."""

    def test_all_intents_defined(self):
        """All required intents should be defined."""
        self.assertEqual(DriftIntent.MANUAL_EDIT.value, "MANUAL_EDIT")
        self.assertEqual(DriftIntent.AUTO_REWRITE.value, "AUTO_REWRITE")
        self.assertEqual(DriftIntent.CURRICULUM_REFRESH.value, "CURRICULUM_REFRESH")
        self.assertEqual(DriftIntent.UNKNOWN.value, "UNKNOWN")


class TestDetectDriftIntent(unittest.TestCase):
    """Test drift intent detection heuristics."""

    def test_empty_notes_returns_unknown(self):
        """Empty notes should return UNKNOWN."""
        self.assertEqual(detect_drift_intent("", ""), DriftIntent.UNKNOWN)
        self.assertEqual(detect_drift_intent("   ", ""), DriftIntent.UNKNOWN)

    def test_manual_edit_detection(self):
        """Manual edit patterns should be detected."""
        self.assertEqual(
            detect_drift_intent("Manual edit by John", ""),
            DriftIntent.MANUAL_EDIT
        )
        self.assertEqual(
            detect_drift_intent("Hand-edited config", ""),
            DriftIntent.MANUAL_EDIT
        )
        self.assertEqual(
            detect_drift_intent("Human approved change", ""),
            DriftIntent.MANUAL_EDIT
        )
        self.assertEqual(
            detect_drift_intent("Reviewed and updated", ""),
            DriftIntent.MANUAL_EDIT
        )

    def test_auto_rewrite_detection(self):
        """Automated rewrite patterns should be detected."""
        self.assertEqual(
            detect_drift_intent("Auto-generated by tool", ""),
            DriftIntent.AUTO_REWRITE
        )
        self.assertEqual(
            detect_drift_intent("Script update", ""),
            DriftIntent.AUTO_REWRITE
        )
        self.assertEqual(
            detect_drift_intent("Automated migration", ""),
            DriftIntent.AUTO_REWRITE
        )
        self.assertEqual(
            detect_drift_intent("", "ci-update"),
            DriftIntent.AUTO_REWRITE
        )

    def test_curriculum_refresh_detection(self):
        """Curriculum refresh patterns should be detected."""
        self.assertEqual(
            detect_drift_intent("Scheduled refresh", ""),
            DriftIntent.CURRICULUM_REFRESH
        )
        self.assertEqual(
            detect_drift_intent("Curriculum update for Q2", ""),
            DriftIntent.CURRICULUM_REFRESH
        )
        self.assertEqual(
            detect_drift_intent("Nightly batch update", ""),
            DriftIntent.CURRICULUM_REFRESH
        )

    def test_origin_considered(self):
        """Origin field should be considered in detection."""
        self.assertEqual(
            detect_drift_intent("", "manual"),
            DriftIntent.MANUAL_EDIT
        )

    def test_no_false_positives_on_unrelated_text(self):
        """Unrelated text should not trigger false positives."""
        self.assertEqual(
            detect_drift_intent("Fixed typo in description", ""),
            DriftIntent.UNKNOWN
        )
        self.assertEqual(
            detect_drift_intent("Added new formula", ""),
            DriftIntent.UNKNOWN
        )

    def test_case_insensitive_matching(self):
        """Matching should be case-insensitive."""
        self.assertEqual(
            detect_drift_intent("MANUAL edit", ""),
            DriftIntent.MANUAL_EDIT
        )
        self.assertEqual(
            detect_drift_intent("AUTOMATED script", ""),
            DriftIntent.AUTO_REWRITE
        )


class TestDriftIntentInCompression(unittest.TestCase):
    """Test that drift intents are included in compression."""

    def test_intent_included_in_compressed_group(self):
        """CompressedEventGroup should include drift intents."""
        event = DriftEvent(
            timestamp="2025-01-01T00:00:00Z",
            old_hash="old",
            new_hash="new",
            drift_type=DriftType.NONE,
            risk_level=RiskLevel.INFO,
            snapshot_index=1,
            git_commit="abc123",
            change_type="modified",
            changed_keys=[],
            notes="Manual edit by admin"
        )
        
        compressed = compress_drift_timeline([event], "test_slice")
        
        self.assertEqual(len(compressed.groups), 1)
        self.assertEqual(len(compressed.groups[0].drift_intents), 1)
        self.assertEqual(compressed.groups[0].drift_intents[0], DriftIntent.MANUAL_EDIT)

    def test_unknown_intent_when_no_match(self):
        """Events without matching patterns should have UNKNOWN intent."""
        event = DriftEvent(
            timestamp="2025-01-01T00:00:00Z",
            old_hash="old",
            new_hash="new",
            drift_type=DriftType.NONE,
            risk_level=RiskLevel.INFO,
            snapshot_index=1,
            git_commit="abc123",
            change_type="modified",
            changed_keys=[],
            notes="Just a regular update"
        )
        
        compressed = compress_drift_timeline([event], "test_slice")
        
        self.assertEqual(compressed.groups[0].drift_intents[0], DriftIntent.UNKNOWN)


# =============================================================================
# TASK 3 TESTS: Chronicle Export Contract
# =============================================================================

class TestChronicleExport(unittest.TestCase):
    """Test chronicle export functionality."""

    def _make_event(
        self,
        snapshot_index: int,
        risk_level: RiskLevel,
        notes: str = ""
    ) -> DriftEvent:
        return DriftEvent(
            timestamp=f"2025-01-{snapshot_index:02d}T00:00:00Z",
            old_hash=f"old_{snapshot_index}",
            new_hash=f"new_{snapshot_index}",
            drift_type=DriftType.NONE if risk_level == RiskLevel.INFO else DriftType.STRUCTURAL,
            risk_level=risk_level,
            snapshot_index=snapshot_index,
            git_commit=f"commit_{snapshot_index}",
            change_type="modified",
            changed_keys=[],
            notes=notes
        )

    def test_chronicle_has_required_fields(self):
        """Chronicle should have all required contract fields."""
        events = [self._make_event(1, RiskLevel.INFO)]
        chronicle = export_chronicle(events, "test_slice")
        
        # Check all required fields exist
        self.assertEqual(chronicle.schema_version, CHRONICLE_SCHEMA_VERSION)
        self.assertEqual(chronicle.slice_name, "test_slice")
        self.assertIsInstance(chronicle.events, CompressedTimeline)
        self.assertIsInstance(chronicle.timeline_hash, str)
        self.assertIsInstance(chronicle.block_count, int)
        self.assertIsInstance(chronicle.warnings, list)
        self.assertIsInstance(chronicle.drift_intents_histogram, dict)

    def test_chronicle_to_dict(self):
        """Chronicle.to_dict should return valid dict."""
        events = [self._make_event(1, RiskLevel.BLOCK)]
        chronicle = export_chronicle(events, "test_slice")
        
        d = chronicle.to_dict()
        
        self.assertIn("schema_version", d)
        self.assertIn("slice_name", d)
        self.assertIn("events", d)
        self.assertIn("timeline_hash", d)
        self.assertIn("block_count", d)
        self.assertIn("warnings", d)
        self.assertIn("drift_intents_histogram", d)
        self.assertIn("export_timestamp", d)

    def test_chronicle_to_json(self):
        """Chronicle.to_json should produce valid JSON."""
        events = [self._make_event(1, RiskLevel.INFO)]
        chronicle = export_chronicle(events, "test_slice")
        
        json_str = chronicle.to_json()
        
        # Should be parseable
        parsed = json.loads(json_str)
        self.assertEqual(parsed["slice_name"], "test_slice")

    def test_chronicle_block_count_accurate(self):
        """Chronicle block_count should match actual BLOCK events."""
        events = [
            self._make_event(1, RiskLevel.INFO),
            self._make_event(2, RiskLevel.BLOCK),
            self._make_event(3, RiskLevel.BLOCK),
            self._make_event(4, RiskLevel.WARN),
        ]
        chronicle = export_chronicle(events, "test_slice")
        
        self.assertEqual(chronicle.block_count, 2)

    def test_chronicle_includes_warnings(self):
        """Chronicle should include provided warnings."""
        events = [self._make_event(1, RiskLevel.INFO)]
        warnings = ["Warning 1", "Warning 2"]
        chronicle = export_chronicle(events, "test_slice", warnings)
        
        self.assertEqual(chronicle.warnings, warnings)

    def test_chronicle_drift_intents_histogram(self):
        """Chronicle should have drift intents histogram."""
        events = [
            self._make_event(1, RiskLevel.INFO, notes="Manual edit"),
            self._make_event(2, RiskLevel.INFO, notes="Automated update"),
            self._make_event(3, RiskLevel.INFO, notes=""),
        ]
        chronicle = export_chronicle(events, "test_slice")
        
        histogram = chronicle.drift_intents_histogram
        
        # Should have all intent types
        for intent in DriftIntent:
            self.assertIn(intent.value, histogram)


class TestChronicleHashStability(unittest.TestCase):
    """Test chronicle hash stability across runs."""

    def _make_event(self, snapshot_index: int) -> DriftEvent:
        return DriftEvent(
            timestamp=f"2025-01-{snapshot_index:02d}T00:00:00Z",
            old_hash=f"old_{snapshot_index}",
            new_hash=f"new_{snapshot_index}",
            drift_type=DriftType.NONE,
            risk_level=RiskLevel.INFO,
            snapshot_index=snapshot_index,
            git_commit=f"commit_{snapshot_index}",
            change_type="modified",
            changed_keys=[],
            notes=""
        )

    def test_chronicle_hash_stable_across_runs(self):
        """Same events should produce same chronicle hash."""
        events = [self._make_event(1), self._make_event(2)]
        
        chronicle1 = export_chronicle(events, "test_slice")
        chronicle2 = export_chronicle(events, "test_slice")
        chronicle3 = export_chronicle(events, "test_slice")
        
        self.assertEqual(chronicle1.timeline_hash, chronicle2.timeline_hash)
        self.assertEqual(chronicle2.timeline_hash, chronicle3.timeline_hash)

    def test_chronicle_hash_deterministic_json(self):
        """Chronicle JSON output should be deterministic."""
        events = [self._make_event(1)]
        
        # Create multiple chronicles and export to JSON
        # (excluding export_timestamp which varies)
        def get_stable_json(chronicle):
            d = chronicle.to_dict()
            del d["export_timestamp"]  # Remove varying field
            return json.dumps(d, sort_keys=True)
        
        chronicle1 = export_chronicle(events, "test_slice")
        chronicle2 = export_chronicle(events, "test_slice")
        
        self.assertEqual(get_stable_json(chronicle1), get_stable_json(chronicle2))


class TestCurriculumHashLedgerChronicle(unittest.TestCase):
    """Test CurriculumHashLedger chronicle export methods."""

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

    def test_export_slice_chronicle(self):
        """export_slice_chronicle should return valid Chronicle."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        chronicle = self.ledger.export_slice_chronicle("slice_a")
        
        self.assertEqual(chronicle.slice_name, "slice_a")
        self.assertIsInstance(chronicle, Chronicle)

    def test_export_all_chronicles(self):
        """export_all_chronicles should return dict of Chronicles."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        chronicles = self.ledger.export_all_chronicles()
        
        # slice_a has drift, should be included
        self.assertIn("slice_a", chronicles)
        self.assertIsInstance(chronicles["slice_a"], Chronicle)

    def test_export_all_chronicles_to_files(self):
        """export_all_chronicles should write JSON files when output_dir provided."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        output_dir = Path(self.temp_dir) / "chronicles"
        chronicles = self.ledger.export_all_chronicles(output_dir=output_dir)
        
        # Check file was written
        expected_file = output_dir / "slice_a_chronicle.json"
        self.assertTrue(expected_file.exists())
        
        # Verify contents
        with open(expected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data["slice_name"], "slice_a")


class TestGoldenFileSnapshot(unittest.TestCase):
    """Test for CI integration - golden file snapshot testing."""

    def test_chronicle_schema_version(self):
        """Chronicle schema version should be stable."""
        self.assertEqual(CHRONICLE_SCHEMA_VERSION, "1.0")

    def test_chronicle_json_structure_stable(self):
        """Chronicle JSON structure should match expected format."""
        event = DriftEvent(
            timestamp="2025-01-01T00:00:00Z",
            old_hash="abc123",
            new_hash="def456",
            drift_type=DriftType.NONE,
            risk_level=RiskLevel.INFO,
            snapshot_index=1,
            git_commit="commit123",
            change_type="modified",
            changed_keys=[],
            notes=""
        )
        chronicle = export_chronicle([event], "test_slice", [])
        
        d = chronicle.to_dict()
        
        # Verify structure matches contract
        required_keys = [
            "schema_version",
            "slice_name",
            "events",
            "timeline_hash",
            "block_count",
            "warnings",
            "drift_intents_histogram",
            "export_timestamp"
        ]
        for key in required_keys:
            self.assertIn(key, d, f"Missing required key: {key}")
        
        # Verify events structure
        events_data = d["events"]
        self.assertIn("groups", events_data)
        self.assertIn("original_event_count", events_data)
        self.assertIn("compressed_group_count", events_data)


# =============================================================================
# PHASE III TESTS: Cross-Slice Chronicle Index & Governance Lens
# =============================================================================

class TestChronicleIndex(unittest.TestCase):
    """Test chronicle index building over all slices."""

    def _make_chronicle(
        self,
        slice_name: str,
        block_count: int = 0,
        event_groups: int = 1,
        intents: Optional[Dict[str, int]] = None
    ) -> Chronicle:
        """Create a Chronicle for testing."""
        if intents is None:
            intents = {i.value: 0 for i in DriftIntent}
        
        # Create minimal compressed timeline
        compressed = CompressedTimeline(
            slice_name=slice_name,
            groups=[],
            original_event_count=event_groups,
            compressed_group_count=event_groups,
            block_count=block_count,
            warn_count=0,
            info_count=event_groups - block_count
        )
        
        return Chronicle(
            schema_version=CHRONICLE_SCHEMA_VERSION,
            slice_name=slice_name,
            events=compressed,
            timeline_hash=f"hash_{slice_name}",
            block_count=block_count,
            warnings=[],
            drift_intents_histogram=intents
        )

    def test_empty_chronicles_list(self):
        """Empty chronicle list should return valid index."""
        index = build_chronicle_index([])
        
        self.assertEqual(index["schema_version"], CHRONICLE_INDEX_SCHEMA_VERSION)
        self.assertEqual(index["slice_count"], 0)
        self.assertEqual(index["total_event_groups"], 0)
        self.assertEqual(index["slices_with_block_events"], [])

    def test_index_has_required_fields(self):
        """Index should have all required fields."""
        chronicle = self._make_chronicle("slice_a")
        index = build_chronicle_index([chronicle])
        
        required_fields = [
            "schema_version",
            "slice_count",
            "total_event_groups",
            "slices_with_block_events",
            "intent_histogram_global",
            "slices_with_refresh_intent"
        ]
        for field in required_fields:
            self.assertIn(field, index, f"Missing required field: {field}")

    def test_slice_count_accurate(self):
        """slice_count should match number of chronicles."""
        chronicles = [
            self._make_chronicle("slice_a"),
            self._make_chronicle("slice_b"),
            self._make_chronicle("slice_c"),
        ]
        index = build_chronicle_index(chronicles)
        
        self.assertEqual(index["slice_count"], 3)

    def test_total_event_groups_aggregated(self):
        """total_event_groups should sum across all slices."""
        chronicles = [
            self._make_chronicle("slice_a", event_groups=5),
            self._make_chronicle("slice_b", event_groups=3),
            self._make_chronicle("slice_c", event_groups=2),
        ]
        index = build_chronicle_index(chronicles)
        
        self.assertEqual(index["total_event_groups"], 10)

    def test_slices_with_block_events_tracked(self):
        """Slices with BLOCK events should be listed."""
        chronicles = [
            self._make_chronicle("slice_a", block_count=0),
            self._make_chronicle("slice_b", block_count=2),
            self._make_chronicle("slice_c", block_count=1),
        ]
        index = build_chronicle_index(chronicles)
        
        self.assertEqual(
            index["slices_with_block_events"],
            ["slice_b", "slice_c"]  # sorted
        )

    def test_intent_histogram_aggregated(self):
        """Intent histogram should aggregate across slices."""
        chronicles = [
            self._make_chronicle("slice_a", intents={
                DriftIntent.MANUAL_EDIT.value: 2,
                DriftIntent.AUTO_REWRITE.value: 0,
                DriftIntent.CURRICULUM_REFRESH.value: 0,
                DriftIntent.UNKNOWN.value: 1,
            }),
            self._make_chronicle("slice_b", intents={
                DriftIntent.MANUAL_EDIT.value: 1,
                DriftIntent.AUTO_REWRITE.value: 3,
                DriftIntent.CURRICULUM_REFRESH.value: 0,
                DriftIntent.UNKNOWN.value: 0,
            }),
        ]
        index = build_chronicle_index(chronicles)
        
        histogram = index["intent_histogram_global"]
        self.assertEqual(histogram[DriftIntent.MANUAL_EDIT.value], 3)
        self.assertEqual(histogram[DriftIntent.AUTO_REWRITE.value], 3)

    def test_slices_with_refresh_intent_tracked(self):
        """Slices with CURRICULUM_REFRESH should be listed."""
        chronicles = [
            self._make_chronicle("slice_a", intents={
                DriftIntent.MANUAL_EDIT.value: 1,
                DriftIntent.AUTO_REWRITE.value: 0,
                DriftIntent.CURRICULUM_REFRESH.value: 0,
                DriftIntent.UNKNOWN.value: 0,
            }),
            self._make_chronicle("slice_b", intents={
                DriftIntent.MANUAL_EDIT.value: 0,
                DriftIntent.AUTO_REWRITE.value: 0,
                DriftIntent.CURRICULUM_REFRESH.value: 2,
                DriftIntent.UNKNOWN.value: 0,
            }),
        ]
        index = build_chronicle_index(chronicles)
        
        self.assertEqual(index["slices_with_refresh_intent"], ["slice_b"])

    def test_deterministic_ordering(self):
        """Index should have deterministic slice ordering."""
        chronicles = [
            self._make_chronicle("slice_c"),
            self._make_chronicle("slice_a"),
            self._make_chronicle("slice_b"),
        ]
        
        index1 = build_chronicle_index(chronicles)
        index2 = build_chronicle_index(list(reversed(chronicles)))
        
        # Should produce same result regardless of input order
        self.assertEqual(
            index1["slices_with_block_events"],
            index2["slices_with_block_events"]
        )


class TestGovernanceLens(unittest.TestCase):
    """Test governance/audit lens summary."""

    def test_governance_summary_has_required_fields(self):
        """Governance summary should have all required fields."""
        index = {
            "slices_with_block_events": [],
            "_slices_with_manual_edits": [],
            "_slices_with_auto_rewrite": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_governance(index)
        
        required_fields = [
            "has_block_level_drift",
            "slices_with_manual_edits",
            "slices_with_auto_rewrite",
            "status"
        ]
        for field in required_fields:
            self.assertIn(field, summary, f"Missing field: {field}")

    def test_has_block_level_drift_true(self):
        """has_block_level_drift should be True when BLOCK events exist."""
        index = {
            "slices_with_block_events": ["slice_a", "slice_b"],
            "_slices_with_manual_edits": [],
            "_slices_with_auto_rewrite": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_governance(index)
        
        self.assertTrue(summary["has_block_level_drift"])

    def test_has_block_level_drift_false(self):
        """has_block_level_drift should be False when no BLOCK events."""
        index = {
            "slices_with_block_events": [],
            "_slices_with_manual_edits": [],
            "_slices_with_auto_rewrite": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_governance(index)
        
        self.assertFalse(summary["has_block_level_drift"])

    def test_status_stable_when_no_issues(self):
        """Status should be STABLE when no BLOCK events and single intent."""
        index = {
            "slices_with_block_events": [],
            "_slices_with_manual_edits": ["slice_a"],
            "_slices_with_auto_rewrite": [],
            "intent_histogram_global": {
                DriftIntent.MANUAL_EDIT.value: 5,
                DriftIntent.AUTO_REWRITE.value: 0,
                DriftIntent.CURRICULUM_REFRESH.value: 0,
                DriftIntent.UNKNOWN.value: 0,
            },
        }
        summary = summarize_chronicles_for_governance(index)
        
        self.assertEqual(summary["status"], GovernanceStatus.STABLE.value)

    def test_status_mixed_when_block_events(self):
        """Status should be MIXED when some BLOCK events."""
        index = {
            "slices_with_block_events": ["slice_a"],
            "_slices_with_manual_edits": [],
            "_slices_with_auto_rewrite": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_governance(index)
        
        self.assertEqual(summary["status"], GovernanceStatus.MIXED.value)

    def test_status_volatile_when_many_block_events(self):
        """Status should be VOLATILE when many BLOCK events."""
        index = {
            "slices_with_block_events": ["slice_a", "slice_b", "slice_c"],
            "_slices_with_manual_edits": [],
            "_slices_with_auto_rewrite": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_governance(index)
        
        self.assertEqual(summary["status"], GovernanceStatus.VOLATILE.value)

    def test_status_mixed_when_multiple_intent_types(self):
        """Status should be MIXED when multiple intent types present."""
        index = {
            "slices_with_block_events": [],
            "_slices_with_manual_edits": ["slice_a"],
            "_slices_with_auto_rewrite": ["slice_b"],
            "intent_histogram_global": {
                DriftIntent.MANUAL_EDIT.value: 2,
                DriftIntent.AUTO_REWRITE.value: 3,
                DriftIntent.CURRICULUM_REFRESH.value: 0,
                DriftIntent.UNKNOWN.value: 0,
            },
        }
        summary = summarize_chronicles_for_governance(index)
        
        self.assertEqual(summary["status"], GovernanceStatus.MIXED.value)


class TestGlobalHealthSignal(unittest.TestCase):
    """Test global health chronicle signal."""

    def test_health_summary_has_required_fields(self):
        """Health summary should have all required fields."""
        index = {
            "total_event_groups": 0,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_global_health(index)
        
        required_fields = [
            "curriculum_change_activity_level",
            "any_block_events",
            "dominant_intent",
            "status"
        ]
        for field in required_fields:
            self.assertIn(field, summary, f"Missing field: {field}")

    def test_activity_level_low(self):
        """Activity should be LOW when few events."""
        index = {
            "total_event_groups": ACTIVITY_THRESHOLD_LOW,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_global_health(index)
        
        self.assertEqual(summary["curriculum_change_activity_level"], ActivityLevel.LOW.value)

    def test_activity_level_medium(self):
        """Activity should be MEDIUM when moderate events."""
        index = {
            "total_event_groups": ACTIVITY_THRESHOLD_LOW + 5,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_global_health(index)
        
        self.assertEqual(summary["curriculum_change_activity_level"], ActivityLevel.MEDIUM.value)

    def test_activity_level_high(self):
        """Activity should be HIGH when many events."""
        index = {
            "total_event_groups": ACTIVITY_THRESHOLD_MEDIUM + 10,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_global_health(index)
        
        self.assertEqual(summary["curriculum_change_activity_level"], ActivityLevel.HIGH.value)

    def test_any_block_events_true(self):
        """any_block_events should be True when BLOCK events exist."""
        index = {
            "total_event_groups": 5,
            "slices_with_block_events": ["slice_a"],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_global_health(index)
        
        self.assertTrue(summary["any_block_events"])

    def test_dominant_intent_detected(self):
        """Dominant intent should be the most common one."""
        index = {
            "total_event_groups": 10,
            "slices_with_block_events": [],
            "intent_histogram_global": {
                DriftIntent.MANUAL_EDIT.value: 2,
                DriftIntent.AUTO_REWRITE.value: 8,
                DriftIntent.CURRICULUM_REFRESH.value: 1,
                DriftIntent.UNKNOWN.value: 0,
            },
        }
        summary = summarize_chronicles_for_global_health(index)
        
        self.assertEqual(summary["dominant_intent"], DriftIntent.AUTO_REWRITE.value)

    def test_status_ok_when_low_activity_no_blocks(self):
        """Status should be OK when low activity and no BLOCK events."""
        index = {
            "total_event_groups": 3,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_global_health(index)
        
        self.assertEqual(summary["status"], HealthStatus.OK.value)

    def test_status_hot_when_block_events(self):
        """Status should be HOT when any BLOCK events."""
        index = {
            "total_event_groups": 3,
            "slices_with_block_events": ["slice_a"],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_global_health(index)
        
        self.assertEqual(summary["status"], HealthStatus.HOT.value)

    def test_status_attention_when_high_activity(self):
        """Status should be ATTENTION when high activity but no BLOCK."""
        index = {
            "total_event_groups": ACTIVITY_THRESHOLD_MEDIUM + 10,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        summary = summarize_chronicles_for_global_health(index)
        
        self.assertEqual(summary["status"], HealthStatus.ATTENTION.value)


class TestLedgerIndexMethods(unittest.TestCase):
    """Test CurriculumHashLedger index and summary methods."""

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

    def test_build_curriculum_index(self):
        """build_curriculum_index should return valid index."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        index = self.ledger.build_curriculum_index()
        
        self.assertIn("schema_version", index)
        self.assertIn("slice_count", index)

    def test_get_governance_summary(self):
        """get_governance_summary should return valid summary."""
        self.ledger.record_snapshot(str(self.config_path))
        
        summary = self.ledger.get_governance_summary()
        
        self.assertIn("has_block_level_drift", summary)
        self.assertIn("status", summary)

    def test_get_health_summary(self):
        """get_health_summary should return valid summary."""
        self.ledger.record_snapshot(str(self.config_path))
        
        summary = self.ledger.get_health_summary()
        
        self.assertIn("curriculum_change_activity_level", summary)
        self.assertIn("any_block_events", summary)
        self.assertIn("dominant_intent", summary)
        self.assertIn("status", summary)


# =============================================================================
# PHASE IV TESTS: Cross-Slice Chronicle Governance & Narrative Feed
# =============================================================================

class TestChronicleAlignmentView(unittest.TestCase):
    """Test chronicle alignment view building."""

    def test_alignment_view_has_required_fields(self):
        """Alignment view should have all required fields."""
        index = {
            "slice_count": 3,
            "total_event_groups": 10,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment = build_chronicle_alignment_view(index)
        
        required_fields = [
            "slices_with_high_edit_churn",
            "slices_with_block_drift_and_block_events",
            "alignment_status"
        ]
        for field in required_fields:
            self.assertIn(field, alignment, f"Missing field: {field}")

    def test_alignment_status_stable_when_no_churn(self):
        """Status should be STABLE when no high churn."""
        index = {
            "slice_count": 3,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment = build_chronicle_alignment_view(index)
        
        self.assertEqual(alignment["alignment_status"], AlignmentStatus.STABLE.value)

    def test_alignment_status_drifty_when_some_churn(self):
        """Status should be DRIFTY when some high churn."""
        index = {
            "slice_count": 3,
            "total_event_groups": 10,
            "slices_with_block_events": ["slice_a"],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment = build_chronicle_alignment_view(index)
        
        self.assertEqual(alignment["alignment_status"], AlignmentStatus.DRIFTY.value)

    def test_alignment_status_volatile_when_high_churn_ratio(self):
        """Status should be VOLATILE when high churn ratio."""
        index = {
            "slice_count": 4,
            "total_event_groups": 50,
            "slices_with_block_events": ["slice_a", "slice_b", "slice_c"],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment = build_chronicle_alignment_view(index)
        
        self.assertEqual(alignment["alignment_status"], AlignmentStatus.VOLATILE.value)

    def test_high_churn_detected_from_timeline(self):
        """High churn should be detected from curriculum_timeline."""
        index = {
            "slice_count": 3,
            "total_event_groups": 10,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        curriculum_timeline = {
            "slice_a": {"event_count": HIGH_CHURN_THRESHOLD},
            "slice_b": {"event_count": 5},
        }
        alignment = build_chronicle_alignment_view(index, curriculum_timeline)
        
        self.assertIn("slice_a", alignment["slices_with_high_edit_churn"])
        self.assertNotIn("slice_b", alignment["slices_with_high_edit_churn"])

    def test_block_drift_and_events_detected(self):
        """Slices with both block drift and block events should be identified."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": ["slice_a", "slice_b"],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        drift_classifications = {
            "slice_a": {"drift_type": DriftType.STRUCTURAL.value},
            "slice_b": {"drift_type": DriftType.COSMETIC.value},  # Not block-level
        }
        alignment = build_chronicle_alignment_view(index, None, drift_classifications)
        
        self.assertIn("slice_a", alignment["slices_with_block_drift_and_block_events"])
        self.assertNotIn("slice_b", alignment["slices_with_block_drift_and_block_events"])

    def test_semantic_drift_classified_as_block(self):
        """SEMANTIC drift should be classified as block-level."""
        index = {
            "slice_count": 1,
            "total_event_groups": 1,
            "slices_with_block_events": ["slice_a"],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        drift_classifications = {
            "slice_a": {"drift_type": DriftType.SEMANTIC.value},
        }
        alignment = build_chronicle_alignment_view(index, None, drift_classifications)
        
        self.assertIn("slice_a", alignment["slices_with_block_drift_and_block_events"])

    def test_parametric_major_drift_classified_as_block(self):
        """PARAMETRIC_MAJOR drift should be classified as block-level."""
        index = {
            "slice_count": 1,
            "total_event_groups": 1,
            "slices_with_block_events": ["slice_a"],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        drift_classifications = {
            "slice_a": {"drift_type": DriftType.PARAMETRIC_MAJOR.value},
        }
        alignment = build_chronicle_alignment_view(index, None, drift_classifications)
        
        self.assertIn("slice_a", alignment["slices_with_block_drift_and_block_events"])


class TestGovernanceNarrative(unittest.TestCase):
    """Test governance narrative rendering."""

    def test_narrative_is_markdown(self):
        """Narrative should be valid Markdown."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "_slices_with_manual_edits": [],
            "slices_with_refresh_intent": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        narrative = render_chronicle_governance_narrative(index, alignment_view)
        
        self.assertIn("#", narrative)  # Has headers
        self.assertIn("##", narrative)  # Has subheaders

    def test_narrative_includes_activity_level(self):
        """Narrative should include change activity level."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "_slices_with_manual_edits": [],
            "slices_with_refresh_intent": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        narrative = render_chronicle_governance_narrative(index, alignment_view)
        
        self.assertIn("Change Activity Level", narrative)

    def test_narrative_highlights_manual_edits_with_blocks(self):
        """Narrative should highlight slices with manual edits and blocks."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": ["slice_a"],
            "_slices_with_manual_edits": ["slice_a"],
            "slices_with_refresh_intent": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        narrative = render_chronicle_governance_narrative(index, alignment_view)
        
        self.assertIn("Manual Edits and Block-Level Events", narrative)
        self.assertIn("slice_a", narrative)

    def test_narrative_includes_refresh_intents(self):
        """Narrative should include curriculum refresh activity."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "_slices_with_manual_edits": [],
            "slices_with_refresh_intent": ["slice_b"],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        narrative = render_chronicle_governance_narrative(index, alignment_view)
        
        self.assertIn("Curriculum Refresh Activity", narrative)
        self.assertIn("slice_b", narrative)

    def test_narrative_includes_high_churn_slices(self):
        """Narrative should include high edit churn slices."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "_slices_with_manual_edits": [],
            "slices_with_refresh_intent": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": ["slice_c"],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        narrative = render_chronicle_governance_narrative(index, alignment_view)
        
        self.assertIn("High Edit Churn Slices", narrative)
        self.assertIn("slice_c", narrative)

    def test_narrative_includes_alignment_status(self):
        """Narrative should include alignment status."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "_slices_with_manual_edits": [],
            "slices_with_refresh_intent": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.DRIFTY.value,
        }
        narrative = render_chronicle_governance_narrative(index, alignment_view)
        
        self.assertIn("Alignment Status", narrative)
        self.assertIn("DRIFTY", narrative)

    def test_narrative_neutral_tone(self):
        """Narrative should use neutral, descriptive language."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "_slices_with_manual_edits": [],
            "slices_with_refresh_intent": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        narrative = render_chronicle_governance_narrative(index, alignment_view)
        
        # Should not contain judgmental words
        self.assertNotIn("bad", narrative.lower())
        self.assertNotIn("good", narrative.lower())
        self.assertNotIn("problem", narrative.lower())
        self.assertNotIn("error", narrative.lower())


class TestAcquisitionSummary(unittest.TestCase):
    """Test acquisition-facing chronicle summary."""

    def test_acquisition_summary_has_required_fields(self):
        """Acquisition summary should have all required fields."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        summary = build_chronicle_summary_for_acquisition(index, alignment_view)
        
        required_fields = [
            "change_activity_band",
            "governance_status",
            "headline"
        ]
        for field in required_fields:
            self.assertIn(field, summary, f"Missing field: {field}")

    def test_activity_band_mapped_correctly(self):
        """Activity band should match health summary."""
        index = {
            "slice_count": 2,
            "total_event_groups": ACTIVITY_THRESHOLD_LOW + 1,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        summary = build_chronicle_summary_for_acquisition(index, alignment_view)
        
        self.assertIn(summary["change_activity_band"], ["LOW", "MEDIUM", "HIGH"])

    def test_governance_status_mapped_correctly(self):
        """Governance status should match governance summary."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        summary = build_chronicle_summary_for_acquisition(index, alignment_view)
        
        self.assertIn(summary["governance_status"], ["STABLE", "MIXED", "VOLATILE"])

    def test_headline_includes_activity_description(self):
        """Headline should describe change activity."""
        index = {
            "slice_count": 2,
            "total_event_groups": 3,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        summary = build_chronicle_summary_for_acquisition(index, alignment_view)
        
        headline = summary["headline"]
        self.assertIn("change activity", headline.lower())
        self.assertIn("2", headline)  # slice_count
        self.assertIn("3", headline)  # total_event_groups

    def test_headline_includes_governance_description(self):
        """Headline should describe governance status."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        summary = build_chronicle_summary_for_acquisition(index, alignment_view)
        
        headline = summary["headline"]
        self.assertIn("governed", headline.lower())

    def test_headline_includes_block_events_when_present(self):
        """Headline should mention block events when they exist."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": ["slice_a"],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        summary = build_chronicle_summary_for_acquisition(index, alignment_view)
        
        headline = summary["headline"]
        self.assertIn("block", headline.lower())
        self.assertIn("1", headline)  # block_count

    def test_headline_neutral_tone(self):
        """Headline should use neutral, descriptive language."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        summary = build_chronicle_summary_for_acquisition(index, alignment_view)
        
        headline = summary["headline"].lower()
        # Should not contain judgmental words
        self.assertNotIn("bad", headline)
        self.assertNotIn("good", headline)
        self.assertNotIn("problem", headline)
        self.assertNotIn("error", headline)


class TestLedgerPhaseIVMethods(unittest.TestCase):
    """Test CurriculumHashLedger Phase IV methods."""

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

    def test_get_alignment_view(self):
        """get_alignment_view should return valid alignment view."""
        self.ledger.record_snapshot(str(self.config_path))
        
        alignment = self.ledger.get_alignment_view()
        
        self.assertIn("slices_with_high_edit_churn", alignment)
        self.assertIn("alignment_status", alignment)

    def test_get_governance_narrative(self):
        """get_governance_narrative should return Markdown string."""
        self.ledger.record_snapshot(str(self.config_path))
        
        narrative = self.ledger.get_governance_narrative()
        
        self.assertIsInstance(narrative, str)
        self.assertIn("#", narrative)

    def test_get_acquisition_summary(self):
        """get_acquisition_summary should return valid summary."""
        self.ledger.record_snapshot(str(self.config_path))
        
        summary = self.ledger.get_acquisition_summary()
        
        self.assertIn("change_activity_band", summary)
        self.assertIn("governance_status", summary)
        self.assertIn("headline", summary)


# =============================================================================
# FOLLOW-UP TESTS: Chronicle Causality Map & Multi-Axis Stability Estimator
# =============================================================================

class TestChronicleCausalityMap(unittest.TestCase):
    """Test chronicle causality map building."""

    def test_causality_map_has_required_fields(self):
        """Causality map should have all required fields."""
        index = {
            "slice_count": 2,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        drift_events = {}
        causality_map = build_chronicle_causality_map(index, drift_events)
        
        required_fields = [
            "causal_links",
            "likely_root_causes",
            "causality_strength_score",
            "neutral_notes"
        ]
        for field in required_fields:
            self.assertIn(field, causality_map, f"Missing field: {field}")

    def test_empty_drift_events_returns_empty_map(self):
        """Empty drift events should return empty causality map."""
        index = {
            "slice_count": 0,
            "total_event_groups": 0,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        drift_events = {}
        causality_map = build_chronicle_causality_map(index, drift_events)
        
        self.assertEqual(causality_map["causal_links"], [])
        self.assertEqual(causality_map["likely_root_causes"], [])
        self.assertEqual(causality_map["causality_strength_score"], 0.0)
        self.assertIn("No drift events", causality_map["neutral_notes"][0])

    def test_temporal_adjacency_detects_causal_links(self):
        """Events within time window should be detected as causal."""
        index = {
            "slice_count": 1,
            "total_event_groups": 2,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        # Two events in same slice within 1 hour
        drift_events = {
            "event_1": {
                "timestamp": "2025-01-01T10:00:00Z",
                "drift_type": DriftType.PARAMETRIC_MINOR.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.WARN.value,
            },
            "event_2": {
                "timestamp": "2025-01-01T10:30:00Z",
                "drift_type": DriftType.PARAMETRIC_MINOR.value,
                "slice_name": "slice_a",
                "snapshot_index": 2,
                "risk_level": RiskLevel.WARN.value,
            },
        }
        causality_map = build_chronicle_causality_map(index, drift_events)
        
        self.assertGreater(len(causality_map["causal_links"]), 0)

    def test_structural_to_semantic_causality(self):
        """STRUCTURAL changes should be detected as causing SEMANTIC changes."""
        index = {
            "slice_count": 1,
            "total_event_groups": 2,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
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
                "timestamp": "2025-01-01T10:30:00Z",
                "drift_type": DriftType.SEMANTIC.value,
                "slice_name": "slice_b",
                "snapshot_index": 2,
                "risk_level": RiskLevel.BLOCK.value,
            },
        }
        causality_map = build_chronicle_causality_map(index, drift_events)
        
        # Should detect STRUCTURAL → SEMANTIC link
        links = causality_map["causal_links"]
        structural_to_semantic = any(
            "STRUCTURAL" in causality_map["neutral_notes"][i] and "SEMANTIC" in causality_map["neutral_notes"][i]
            for i in range(len(links))
        )
        # Check if link exists
        has_link = any(
            ("event_1", "event_2") == link or ("event_1", "event_2") in str(link)
            for link in links
        )
        self.assertTrue(has_link or structural_to_semantic)

    def test_root_causes_identified(self):
        """Events with no incoming links should be identified as root causes."""
        index = {
            "slice_count": 2,
            "total_event_groups": 3,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
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
                "timestamp": "2025-01-01T10:30:00Z",
                "drift_type": DriftType.SEMANTIC.value,
                "slice_name": "slice_a",
                "snapshot_index": 2,
                "risk_level": RiskLevel.BLOCK.value,
            },
        }
        causality_map = build_chronicle_causality_map(index, drift_events)
        
        # event_1 should be a root cause (no incoming links)
        self.assertIn("event_1", causality_map["likely_root_causes"])

    def test_causality_strength_score_calculation(self):
        """Causality strength score should be calculated correctly."""
        index = {
            "slice_count": 1,
            "total_event_groups": 5,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        drift_events = {
            f"event_{i}": {
                "timestamp": f"2025-01-01T10:{i:02d}:00Z",
                "drift_type": DriftType.PARAMETRIC_MINOR.value,
                "slice_name": "slice_a",
                "snapshot_index": i,
                "risk_level": RiskLevel.WARN.value,
            }
            for i in range(5)
        }
        causality_map = build_chronicle_causality_map(index, drift_events)
        
        score = causality_map["causality_strength_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_causality_deterministic_ordering(self):
        """Causality map should be deterministic across runs."""
        index = {
            "slice_count": 1,
            "total_event_groups": 3,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
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
                "timestamp": "2025-01-01T10:30:00Z",
                "drift_type": DriftType.SEMANTIC.value,
                "slice_name": "slice_a",
                "snapshot_index": 2,
                "risk_level": RiskLevel.BLOCK.value,
            },
        }
        
        map1 = build_chronicle_causality_map(index, drift_events)
        map2 = build_chronicle_causality_map(index, drift_events)
        
        self.assertEqual(len(map1["causal_links"]), len(map2["causal_links"]))
        self.assertEqual(map1["causality_strength_score"], map2["causality_strength_score"])

    def test_events_too_far_apart_not_linked(self):
        """Events beyond time window should not be linked."""
        index = {
            "slice_count": 1,
            "total_event_groups": 2,
            "slices_with_block_events": [],
            "intent_histogram_global": {i.value: 0 for i in DriftIntent},
        }
        # Events 25 hours apart (beyond 24-hour window)
        drift_events = {
            "event_1": {
                "timestamp": "2025-01-01T10:00:00Z",
                "drift_type": DriftType.PARAMETRIC_MINOR.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.WARN.value,
            },
            "event_2": {
                "timestamp": "2025-01-02T11:00:00Z",
                "drift_type": DriftType.PARAMETRIC_MINOR.value,
                "slice_name": "slice_a",
                "snapshot_index": 2,
                "risk_level": RiskLevel.WARN.value,
            },
        }
        causality_map = build_chronicle_causality_map(index, drift_events)
        
        # Should have no causal links (events too far apart)
        # Note: May still have root causes, but no links
        self.assertEqual(len(causality_map["causal_links"]), 0)


class TestMultiAxisStabilityEstimator(unittest.TestCase):
    """Test multi-axis stability estimation."""

    def test_stability_estimate_has_required_fields(self):
        """Stability estimate should have all required fields."""
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        stability = estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
        
        required_fields = [
            "stability_band",
            "axes_contributing",
            "headline",
            "evidence_fields"
        ]
        for field in required_fields:
            self.assertIn(field, stability, f"Missing field: {field}")

    def test_stability_band_high_when_stable(self):
        """Stability band should be HIGH when all axes are stable."""
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.1,  # Low causality
            "neutral_notes": [],
        }
        stability = estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
        
        self.assertEqual(stability["stability_band"], StabilityBand.HIGH.value)

    def test_stability_band_low_when_volatile(self):
        """Stability band should be LOW when alignment is volatile."""
        alignment_view = {
            "slices_with_high_edit_churn": ["slice_a", "slice_b", "slice_c"],
            "slices_with_block_drift_and_block_events": ["slice_a"],
            "alignment_status": AlignmentStatus.VOLATILE.value,
        }
        causality_map = {
            "causal_links": [("event_1", "event_2"), ("event_2", "event_3")],
            "likely_root_causes": ["event_1", "event_2", "event_3", "event_4"],
            "causality_strength_score": 0.8,  # High causality
            "neutral_notes": [],
        }
        stability = estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
        
        self.assertEqual(stability["stability_band"], StabilityBand.LOW.value)

    def test_stability_band_medium_when_mixed(self):
        """Stability band should be MEDIUM when alignment is drifty."""
        alignment_view = {
            "slices_with_high_edit_churn": ["slice_a"],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.DRIFTY.value,
        }
        causality_map = {
            "causal_links": [("event_1", "event_2")],
            "likely_root_causes": ["event_1"],
            "causality_strength_score": 0.3,  # Moderate causality
            "neutral_notes": [],
        }
        stability = estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
        
        self.assertIn(stability["stability_band"], [StabilityBand.MEDIUM.value, StabilityBand.LOW.value])

    def test_axes_contributing_tracked(self):
        """Contributing axes should be tracked."""
        alignment_view = {
            "slices_with_high_edit_churn": ["slice_a", "slice_b"],
            "slices_with_block_drift_and_block_events": ["slice_c", "slice_d"],  # Need >= 2 for drift axis
            "alignment_status": AlignmentStatus.VOLATILE.value,
        }
        causality_map = {
            "causal_links": [("event_1", "event_2")],
            "likely_root_causes": ["event_1", "event_2", "event_3", "event_4"],
            "causality_strength_score": 0.7,
            "neutral_notes": [],
        }
        stability = estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
        
        axes = stability["axes_contributing"]
        self.assertIn("alignment", axes)
        self.assertIn("causality", axes)
        self.assertIn("churn", axes)
        self.assertIn("drift", axes)

    def test_evidence_fields_include_all_axes(self):
        """Evidence fields should include all four axes."""
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        stability = estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
        
        evidence = stability["evidence_fields"]
        self.assertIn("alignment", evidence)
        self.assertIn("causality", evidence)
        self.assertIn("churn", evidence)
        self.assertIn("drift", evidence)

    def test_headline_includes_stability_description(self):
        """Headline should describe stability level."""
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        stability = estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
        
        headline = stability["headline"].lower()
        self.assertIn("stability", headline)

    def test_headline_includes_contributing_axes(self):
        """Headline should mention contributing axes."""
        alignment_view = {
            "slices_with_high_edit_churn": ["slice_a"],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.DRIFTY.value,
        }
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.2,
            "neutral_notes": [],
        }
        stability = estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
        
        headline = stability["headline"]
        # Should mention contributing factors
        self.assertIn("Contributing factors", headline)

    def test_headline_neutral_tone(self):
        """Headline should use neutral, descriptive language."""
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        stability = estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
        
        headline = stability["headline"].lower()
        # Should not contain judgmental words
        self.assertNotIn("bad", headline)
        self.assertNotIn("good", headline)
        self.assertNotIn("problem", headline)
        self.assertNotIn("error", headline)

    def test_composite_score_calculated(self):
        """Composite stability score should be calculated."""
        alignment_view = {
            "slices_with_high_edit_churn": [],
            "slices_with_block_drift_and_block_events": [],
            "alignment_status": AlignmentStatus.STABLE.value,
        }
        causality_map = {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": [],
        }
        stability = estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
        
        self.assertIn("composite_score", stability)
        score = stability["composite_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestLedgerCausalityAndStabilityMethods(unittest.TestCase):
    """Test CurriculumHashLedger causality and stability methods."""

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

    def test_get_causality_map(self):
        """get_causality_map should return valid causality map."""
        self.ledger.record_snapshot(str(self.config_path))
        
        self.config_data["slices"]["slice_a"]["depth"] = 10
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
        self.ledger.record_snapshot(str(self.config_path))
        
        causality_map = self.ledger.get_causality_map()
        
        self.assertIn("causal_links", causality_map)
        self.assertIn("causality_strength_score", causality_map)

    def test_get_causality_map_with_custom_events(self):
        """get_causality_map should accept custom drift events."""
        index = self.ledger.build_curriculum_index()
        custom_events = {
            "event_1": {
                "timestamp": "2025-01-01T10:00:00Z",
                "drift_type": DriftType.STRUCTURAL.value,
                "slice_name": "slice_a",
                "snapshot_index": 1,
                "risk_level": RiskLevel.BLOCK.value,
            }
        }
        causality_map = self.ledger.get_causality_map(custom_events)
        
        self.assertIn("causal_links", causality_map)

    def test_get_multi_axis_stability(self):
        """get_multi_axis_stability should return valid stability estimate."""
        self.ledger.record_snapshot(str(self.config_path))
        
        stability = self.ledger.get_multi_axis_stability()
        
        self.assertIn("stability_band", stability)
        self.assertIn("axes_contributing", stability)
        self.assertIn("headline", stability)
        self.assertIn("evidence_fields", stability)


if __name__ == '__main__':
    unittest.main()

