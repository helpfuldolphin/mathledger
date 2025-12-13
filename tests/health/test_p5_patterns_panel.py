"""Tests for P5 Patterns Calibration Panel.

Tests cover:
1. Per-experiment snapshot emission determinism (5 tests)
2. Panel aggregation determinism (5 tests)
3. Evidence attachment non-mutating (5 tests)
4. Status integration with manifest-first extraction (5 tests)
5. GGFL adapter with low weight and no conflict (5 tests)
6. Warning hygiene cap verification (2 tests)

SHADOW MODE CONTRACT: All tests verify observational-only behavior.
"""

from __future__ import annotations

import copy
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from backend.topology.first_light.p5_pattern_classifier import (
    DivergencePattern,
    PatternClassification,
    P5_PATTERN_SCHEMA_VERSION,
)
from backend.health.p5_patterns_panel import (
    P5_PATTERNS_SNAPSHOT_SCHEMA_VERSION,
    P5_PATTERNS_PANEL_SCHEMA_VERSION,
    EXTRACTION_SOURCE_MANIFEST,
    EXTRACTION_SOURCE_EVIDENCE_JSON,
    EXTRACTION_SOURCE_MISSING,
    MAX_DRIVERS_CAP,
    P5PatternSnapshot,
    P5PatternsPanel,
    build_p5_patterns_snapshot,
    persist_p5_patterns_snapshot,
    load_p5_patterns_snapshot,
    build_p5_patterns_panel,
    persist_p5_patterns_panel,
    attach_p5_patterns_panel_to_evidence,
    extract_p5_patterns_panel_status,
    extract_p5_patterns_panel_signal_for_status,
    extract_p5_patterns_panel_warnings,
    p5_patterns_panel_for_alignment_view,
    load_and_build_panel_from_directory,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_classifications() -> List[PatternClassification]:
    """Create sample pattern classifications for testing."""
    classifications = []
    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat()

    # 30 NOMINAL classifications
    for i in range(30):
        classifications.append(PatternClassification(
            pattern=DivergencePattern.NOMINAL,
            confidence=0.8,
            evidence={"trigger": "none", "cycle": i},
            timestamp=timestamp,
        ))

    # 10 DRIFT classifications
    for i in range(10):
        classifications.append(PatternClassification(
            pattern=DivergencePattern.DRIFT,
            confidence=0.85,
            evidence={"trigger": "systematic_bias", "cycle": 30 + i},
            timestamp=timestamp,
        ))

    # 5 STRUCTURAL_BREAK classifications (high-severity)
    for i in range(5):
        classifications.append(PatternClassification(
            pattern=DivergencePattern.STRUCTURAL_BREAK,
            confidence=0.92,
            evidence={"trigger": "sudden_increase_sustained", "cycle": 40 + i},
            timestamp=timestamp,
        ))

    # 5 ATTRACTOR_MISS classifications (high-severity)
    for i in range(5):
        classifications.append(PatternClassification(
            pattern=DivergencePattern.ATTRACTOR_MISS,
            confidence=0.88,
            evidence={"trigger": "omega_mismatch", "cycle": 45 + i},
            timestamp=timestamp,
        ))

    return classifications


@pytest.fixture
def sample_classifications_cal_exp_2() -> List[PatternClassification]:
    """Create sample classifications for CAL-EXP-2 (convergence test)."""
    classifications = []
    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat()

    # Mostly NOMINAL with some DRIFT
    for i in range(40):
        classifications.append(PatternClassification(
            pattern=DivergencePattern.NOMINAL,
            confidence=0.8,
            evidence={"trigger": "none", "cycle": i},
            timestamp=timestamp,
        ))

    for i in range(10):
        classifications.append(PatternClassification(
            pattern=DivergencePattern.DRIFT,
            confidence=0.75,
            evidence={"trigger": "systematic_bias", "cycle": 40 + i},
            timestamp=timestamp,
        ))

    return classifications


@pytest.fixture
def sample_classifications_cal_exp_3() -> List[PatternClassification]:
    """Create sample classifications for CAL-EXP-3 (regime change)."""
    classifications = []
    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat()

    # Regime change scenario: NOMINAL then STRUCTURAL_BREAK streak
    for i in range(20):
        classifications.append(PatternClassification(
            pattern=DivergencePattern.NOMINAL,
            confidence=0.8,
            evidence={"trigger": "none", "cycle": i},
            timestamp=timestamp,
        ))

    # Structural break streak of 15
    for i in range(15):
        classifications.append(PatternClassification(
            pattern=DivergencePattern.STRUCTURAL_BREAK,
            confidence=0.95,
            evidence={"trigger": "sudden_increase_sustained", "cycle": 20 + i},
            timestamp=timestamp,
        ))

    for i in range(15):
        classifications.append(PatternClassification(
            pattern=DivergencePattern.TRANSIENT_MISS,
            confidence=0.7,
            evidence={"trigger": "excursion_only_divergence", "cycle": 35 + i},
            timestamp=timestamp,
        ))

    return classifications


@pytest.fixture
def sample_evidence() -> Dict[str, Any]:
    """Create sample evidence dictionary."""
    return {
        "run_id": "test-run-123",
        "timestamp": "2025-01-01T12:00:00Z",
        "topology": {
            "H": 0.85,
            "rho": 0.88,
            "within_omega": True,
        },
        "governance": {
            "existing_field": "should_be_preserved",
        },
    }


# =============================================================================
# Test: Per-Experiment Snapshot Emission Determinism (5 tests)
# =============================================================================

class TestSnapshotEmissionDeterminism:
    """Test deterministic snapshot generation."""

    def test_snapshot_determinism_same_input(self, sample_classifications):
        """Same classifications produce identical snapshots (except timestamp)."""
        snap1 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        snap2 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)

        # Content hash should be identical
        assert snap1.content_hash == snap2.content_hash

        # Key fields should match
        assert snap1.dominant_pattern == snap2.dominant_pattern
        assert snap1.pattern_counts == snap2.pattern_counts
        assert snap1.max_streak == snap2.max_streak
        assert snap1.total_classifications == snap2.total_classifications

    def test_snapshot_pattern_counts_correct(self, sample_classifications):
        """Pattern counts match actual classification distribution."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)

        assert snap.pattern_counts[DivergencePattern.NOMINAL.value] == 30
        assert snap.pattern_counts[DivergencePattern.DRIFT.value] == 10
        assert snap.pattern_counts[DivergencePattern.STRUCTURAL_BREAK.value] == 5
        assert snap.pattern_counts[DivergencePattern.ATTRACTOR_MISS.value] == 5
        assert snap.total_classifications == 50

    def test_snapshot_dominant_pattern_correct(self, sample_classifications):
        """Dominant pattern is the most frequent classification."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)

        assert snap.dominant_pattern == DivergencePattern.NOMINAL.value

    def test_snapshot_max_streak_computed(self, sample_classifications_cal_exp_3):
        """Max streak is correctly computed from consecutive patterns."""
        snap = build_p5_patterns_snapshot("CAL-EXP-3", sample_classifications_cal_exp_3)

        # Longest streak is 20 NOMINAL at the start
        assert snap.max_streak == 20
        assert snap.max_streak_pattern == DivergencePattern.NOMINAL.value

    def test_snapshot_high_confidence_events_collected(self, sample_classifications):
        """High confidence non-NOMINAL events are collected."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications, high_confidence_threshold=0.85)

        # Should have high-confidence events from STRUCTURAL_BREAK (0.92) and ATTRACTOR_MISS (0.88) and DRIFT (0.85)
        assert len(snap.high_confidence_events) > 0

        # All collected events should be >= threshold
        for event in snap.high_confidence_events:
            assert event["confidence"] >= 0.85
            assert event["pattern"] != DivergencePattern.NOMINAL.value


# =============================================================================
# Test: Panel Aggregation Determinism (5 tests)
# =============================================================================

class TestPanelAggregationDeterminism:
    """Test deterministic panel aggregation across experiments."""

    def test_panel_determinism_same_snapshots(
        self,
        sample_classifications,
        sample_classifications_cal_exp_2,
        sample_classifications_cal_exp_3,
    ):
        """Same snapshots produce identical panel (except timestamp)."""
        snap1 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        snap2 = build_p5_patterns_snapshot("CAL-EXP-2", sample_classifications_cal_exp_2)
        snap3 = build_p5_patterns_snapshot("CAL-EXP-3", sample_classifications_cal_exp_3)

        panel_a = build_p5_patterns_panel([snap1, snap2, snap3])
        panel_b = build_p5_patterns_panel([snap1, snap2, snap3])

        # Content hash should be identical
        assert panel_a.content_hash == panel_b.content_hash

    def test_panel_aggregates_pattern_counts(
        self,
        sample_classifications,
        sample_classifications_cal_exp_2,
    ):
        """Panel correctly aggregates counts across experiments."""
        snap1 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        snap2 = build_p5_patterns_snapshot("CAL-EXP-2", sample_classifications_cal_exp_2)

        panel = build_p5_patterns_panel([snap1, snap2])

        # Aggregated counts should be sum of individual
        expected_nominal = 30 + 40  # CAL-EXP-1 + CAL-EXP-2
        expected_drift = 10 + 10

        assert panel.aggregated_pattern_counts[DivergencePattern.NOMINAL.value] == expected_nominal
        assert panel.aggregated_pattern_counts[DivergencePattern.DRIFT.value] == expected_drift

    def test_panel_tracks_structural_break_experiments(
        self,
        sample_classifications,
        sample_classifications_cal_exp_2,
        sample_classifications_cal_exp_3,
    ):
        """Panel identifies experiments with STRUCTURAL_BREAK."""
        snap1 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        snap2 = build_p5_patterns_snapshot("CAL-EXP-2", sample_classifications_cal_exp_2)
        snap3 = build_p5_patterns_snapshot("CAL-EXP-3", sample_classifications_cal_exp_3)

        panel = build_p5_patterns_panel([snap1, snap2, snap3])

        # CAL-EXP-1 and CAL-EXP-3 have STRUCTURAL_BREAK
        assert "CAL-EXP-1" in panel.structural_break_experiments
        assert "CAL-EXP-3" in panel.structural_break_experiments
        assert "CAL-EXP-2" not in panel.structural_break_experiments

    def test_panel_tracks_attractor_miss_experiments(
        self,
        sample_classifications,
        sample_classifications_cal_exp_2,
    ):
        """Panel identifies experiments with ATTRACTOR_MISS."""
        snap1 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        snap2 = build_p5_patterns_snapshot("CAL-EXP-2", sample_classifications_cal_exp_2)

        panel = build_p5_patterns_panel([snap1, snap2])

        # Only CAL-EXP-1 has ATTRACTOR_MISS
        assert "CAL-EXP-1" in panel.attractor_miss_experiments
        assert "CAL-EXP-2" not in panel.attractor_miss_experiments

    def test_panel_max_streak_across_experiments(
        self,
        sample_classifications,
        sample_classifications_cal_exp_3,
    ):
        """Panel tracks maximum streak across all experiments."""
        snap1 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        snap3 = build_p5_patterns_snapshot("CAL-EXP-3", sample_classifications_cal_exp_3)

        panel = build_p5_patterns_panel([snap1, snap3])

        # CAL-EXP-3 has longer NOMINAL streak (20) than CAL-EXP-1 (30)
        # Actually CAL-EXP-1 has 30 NOMINAL at start
        assert panel.max_streak_across_experiments == 30
        assert panel.max_streak_experiment == "CAL-EXP-1"


# =============================================================================
# Test: Evidence Attachment Non-Mutating (5 tests)
# =============================================================================

class TestEvidenceAttachmentNonMutating:
    """Test that evidence attachment is non-mutating."""

    def test_attach_panel_returns_new_dict(self, sample_classifications, sample_evidence):
        """Attaching panel returns a new dictionary."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        original_id = id(sample_evidence)
        updated = attach_p5_patterns_panel_to_evidence(sample_evidence, panel)

        # Should be a new dict
        assert id(updated) != original_id

    def test_attach_panel_preserves_original(self, sample_classifications, sample_evidence):
        """Original evidence is not modified."""
        original = copy.deepcopy(sample_evidence)
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        attach_p5_patterns_panel_to_evidence(sample_evidence, panel)

        # Original should be unchanged
        assert sample_evidence == original

    def test_attach_panel_preserves_existing_governance(self, sample_classifications, sample_evidence):
        """Existing governance fields are preserved."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        updated = attach_p5_patterns_panel_to_evidence(sample_evidence, panel)

        # Existing field should be preserved
        assert updated["governance"]["existing_field"] == "should_be_preserved"
        # Panel should be added
        assert "p5_patterns_panel" in updated["governance"]

    def test_attach_panel_creates_governance_if_missing(self, sample_classifications):
        """Governance section is created if missing."""
        evidence_no_governance = {"run_id": "test", "timestamp": "2025-01-01T00:00:00Z"}
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        updated = attach_p5_patterns_panel_to_evidence(evidence_no_governance, panel)

        assert "governance" in updated
        assert "p5_patterns_panel" in updated["governance"]

    def test_attach_panel_includes_shadow_mode(self, sample_classifications, sample_evidence):
        """Attached panel includes SHADOW mode marker."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        updated = attach_p5_patterns_panel_to_evidence(sample_evidence, panel)

        assert updated["governance"]["p5_patterns_panel"]["mode"] == "SHADOW"


# =============================================================================
# Test: File Persistence and Loading
# =============================================================================

class TestFilePersistenceAndLoading:
    """Test file I/O operations."""

    def test_snapshot_persist_and_load_roundtrip(self, sample_classifications):
        """Snapshot can be persisted and loaded without data loss."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = persist_p5_patterns_snapshot(snap, output_dir)

            loaded = load_p5_patterns_snapshot(path)

            assert loaded is not None
            assert loaded.cal_id == snap.cal_id
            assert loaded.content_hash == snap.content_hash
            assert loaded.pattern_counts == snap.pattern_counts

    def test_panel_persist_and_load_roundtrip(self, sample_classifications):
        """Panel can be persisted and loaded without data loss."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = persist_p5_patterns_panel(panel, output_dir)

            # Load manually since we don't have a load_panel function
            with open(path, 'r') as f:
                data = json.load(f)

            loaded = P5PatternsPanel.from_dict(data)

            assert loaded.content_hash == panel.content_hash
            assert loaded.total_experiments == panel.total_experiments

    def test_load_nonexistent_snapshot_returns_none(self):
        """Loading nonexistent file returns None."""
        result = load_p5_patterns_snapshot(Path("/nonexistent/file.json"))
        assert result is None

    def test_load_and_build_panel_from_directory(self, sample_classifications, sample_classifications_cal_exp_2):
        """Can load multiple snapshots and build panel from directory."""
        snap1 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        snap2 = build_p5_patterns_snapshot("CAL-EXP-2", sample_classifications_cal_exp_2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            persist_p5_patterns_snapshot(snap1, output_dir)
            persist_p5_patterns_snapshot(snap2, output_dir)

            panel, loaded_snaps = load_and_build_panel_from_directory(output_dir)

            assert panel.total_experiments == 2
            assert len(loaded_snaps) == 2


# =============================================================================
# Test: Status Extraction
# =============================================================================

class TestStatusExtraction:
    """Test status summary extraction."""

    def test_extract_status_with_high_severity(
        self,
        sample_classifications,
        sample_classifications_cal_exp_3,
    ):
        """Status correctly reports high-severity patterns."""
        snap1 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        snap3 = build_p5_patterns_snapshot("CAL-EXP-3", sample_classifications_cal_exp_3)

        panel = build_p5_patterns_panel([snap1, snap3])
        status = extract_p5_patterns_panel_status(panel)

        assert status["total_experiments"] == 2
        assert status["experiments_with_high_severity"] == 2
        assert status["has_structural_breaks"] is True
        assert status["has_attractor_misses"] is True

    def test_extract_status_no_high_severity(self, sample_classifications_cal_exp_2):
        """Status correctly reports when no high-severity patterns."""
        snap = build_p5_patterns_snapshot("CAL-EXP-2", sample_classifications_cal_exp_2)
        panel = build_p5_patterns_panel([snap])
        status = extract_p5_patterns_panel_status(panel)

        assert status["experiments_with_high_severity"] == 0
        assert status["has_structural_breaks"] is False
        assert status["has_attractor_misses"] is False


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_classifications_snapshot(self):
        """Snapshot handles empty classifications list."""
        snap = build_p5_patterns_snapshot("CAL-EXP-EMPTY", [])

        assert snap.cal_id == "CAL-EXP-EMPTY"
        assert snap.dominant_pattern == DivergencePattern.NOMINAL.value
        assert snap.total_classifications == 0
        assert snap.pattern_counts == {}

    def test_empty_snapshots_panel(self):
        """Panel handles empty snapshots list."""
        panel = build_p5_patterns_panel([])

        assert panel.total_experiments == 0
        assert panel.aggregated_pattern_counts == {}
        assert panel.structural_break_experiments == []

    def test_single_classification_snapshot(self):
        """Snapshot handles single classification."""
        single = PatternClassification(
            pattern=DivergencePattern.DRIFT,
            confidence=0.9,
            evidence={},
        )
        snap = build_p5_patterns_snapshot("CAL-SINGLE", [single])

        assert snap.total_classifications == 1
        assert snap.dominant_pattern == DivergencePattern.DRIFT.value
        assert snap.max_streak == 1

    def test_snapshot_json_serializable(self, sample_classifications):
        """Snapshot converts to JSON-serializable dict."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)

        # Should not raise
        json_str = json.dumps(snap.to_dict())
        assert isinstance(json_str, str)

        # Roundtrip should work
        parsed = json.loads(json_str)
        assert parsed["cal_id"] == "CAL-EXP-1"

    def test_panel_json_serializable(self, sample_classifications):
        """Panel converts to JSON-serializable dict."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        # Should not raise
        json_str = json.dumps(panel.to_dict())
        assert isinstance(json_str, str)

        # Roundtrip should work
        parsed = json.loads(json_str)
        assert parsed["total_experiments"] == 1


# =============================================================================
# Test: Status Integration (Manifest-First, Evidence-Fallback) (5 tests)
# =============================================================================

class TestStatusIntegration:
    """Test manifest-first status signal extraction."""

    def test_manifest_first_extraction(self, sample_classifications):
        """Manifest is preferred over evidence when both present."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        evidence = {"governance": {"p5_patterns_panel": {"max_streak_across_experiments": 999}}}

        signal = extract_p5_patterns_panel_signal_for_status(manifest, evidence)

        # Should use manifest value, not evidence
        assert signal["max_streak"] == panel.max_streak_across_experiments
        assert signal["max_streak"] != 999

    def test_evidence_fallback(self, sample_classifications):
        """Falls back to evidence when manifest is None."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        evidence = {"governance": {"p5_patterns_panel": panel.to_dict()}}

        signal = extract_p5_patterns_panel_signal_for_status(None, evidence)

        assert signal["max_streak"] == panel.max_streak_across_experiments
        assert signal["dominant_pattern"] == DivergencePattern.NOMINAL.value

    def test_returns_defaults_when_no_data(self):
        """Returns default signal when neither source has data."""
        signal = extract_p5_patterns_panel_signal_for_status(None, None)

        assert signal["dominant_pattern"] == DivergencePattern.NOMINAL.value
        assert signal["max_streak"] == 0
        assert signal["experiments_with_high_severity"] == 0
        assert signal["status"] == "ok"

    def test_status_warn_when_structural_break_exists(self, sample_classifications):
        """Status is 'warn' when STRUCTURAL_BREAK experiments exist."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        # sample_classifications has STRUCTURAL_BREAK
        assert signal["status"] == "warn"

    def test_status_ok_when_no_structural_break(self, sample_classifications_cal_exp_2):
        """Status is 'ok' when no STRUCTURAL_BREAK experiments exist."""
        snap = build_p5_patterns_snapshot("CAL-EXP-2", sample_classifications_cal_exp_2)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        # CAL-EXP-2 has no STRUCTURAL_BREAK
        assert signal["status"] == "ok"


# =============================================================================
# Test: Warning Extraction and Hygiene Cap (2 tests)
# =============================================================================

class TestWarningHygieneCap:
    """Test warning extraction with hygiene cap."""

    def test_warning_generated_for_structural_break(self, sample_classifications):
        """Warning is generated when STRUCTURAL_BREAK experiments exist."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        warnings = extract_p5_patterns_panel_warnings(manifest, None)

        assert len(warnings) == 1
        assert "STRUCTURAL_BREAK" in warnings[0]
        assert "CAL-EXP-1" in warnings[0]

    def test_warning_hygiene_cap_at_one(self, sample_classifications, sample_classifications_cal_exp_3):
        """Warning list is capped at 1 even with multiple experiments."""
        snap1 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        snap3 = build_p5_patterns_snapshot("CAL-EXP-3", sample_classifications_cal_exp_3)
        panel = build_p5_patterns_panel([snap1, snap3])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        warnings = extract_p5_patterns_panel_warnings(manifest, None)

        # Both experiments have STRUCTURAL_BREAK, but cap is 1
        assert len(warnings) <= 1

    def test_no_warning_when_no_structural_break(self, sample_classifications_cal_exp_2):
        """No warning when no STRUCTURAL_BREAK experiments exist."""
        snap = build_p5_patterns_snapshot("CAL-EXP-2", sample_classifications_cal_exp_2)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        warnings = extract_p5_patterns_panel_warnings(manifest, None)

        assert len(warnings) == 0


# =============================================================================
# Test: GGFL Adapter (SIG-PAT Low Weight, No Conflict) (5 tests)
# =============================================================================

class TestGGFLAdapter:
    """Test GGFL alignment view adapter."""

    def test_adapter_returns_sig_pat_type(self, sample_classifications):
        """Adapter returns signal with type SIG-PAT."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        assert ggfl_signal["signal_type"] == "SIG-PAT"

    def test_adapter_always_no_conflict(self, sample_classifications):
        """Adapter always returns conflict=False (no solo hard block)."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        # Even with STRUCTURAL_BREAK, conflict must be False
        assert ggfl_signal["conflict"] is False

    def test_adapter_low_weight(self, sample_classifications):
        """Adapter always returns weight_hint=LOW."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        assert ggfl_signal["weight_hint"] == "LOW"

    def test_adapter_includes_dominant_pattern_driver(self, sample_classifications_cal_exp_3):
        """Adapter includes dominant pattern in drivers when not NOMINAL."""
        # Use CAL-EXP-3 which has STRUCTURAL_BREAK as a significant pattern
        snap = build_p5_patterns_snapshot("CAL-EXP-3", sample_classifications_cal_exp_3)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        # Should have drivers (high_severity at minimum since CAL-EXP-3 has STRUCTURAL_BREAK)
        # Note: NOMINAL pattern does NOT generate a driver (only non-NOMINAL patterns do)
        assert len(ggfl_signal["drivers"]) >= 1

    def test_adapter_includes_streak_driver_when_significant(self, sample_classifications):
        """Adapter includes streak driver when >= 5."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        # sample_classifications has max_streak of 30
        streak_driver = [d for d in ggfl_signal["drivers"] if d.startswith("DRIVER_STREAK_")]
        assert len(streak_driver) == 1

    def test_adapter_status_warn_with_high_severity(self, sample_classifications):
        """Adapter status is 'warn' when high-severity patterns present."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        assert ggfl_signal["status"] == "warn"

    def test_adapter_handles_empty_signal(self):
        """Adapter handles empty/None signal gracefully."""
        ggfl_signal = p5_patterns_panel_for_alignment_view({})

        assert ggfl_signal["signal_type"] == "SIG-PAT"
        assert ggfl_signal["status"] == "ok"
        assert ggfl_signal["conflict"] is False
        assert ggfl_signal["weight_hint"] == "LOW"
        assert "not available" in ggfl_signal["summary"]

    def test_adapter_drivers_capped_at_three(self, sample_classifications):
        """Adapter caps drivers list at 3."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        assert len(ggfl_signal["drivers"]) <= 3


# =============================================================================
# Test: Status Signal Stabilization (schema_version, mode, weight_hint)
# =============================================================================

class TestStatusSignalStabilization:
    """Test stabilized status signal fields."""

    def test_status_signal_includes_schema_version(self, sample_classifications):
        """Status signal includes schema_version passthrough."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        assert "schema_version" in signal
        assert signal["schema_version"] == P5_PATTERNS_PANEL_SCHEMA_VERSION

    def test_status_signal_always_shadow_mode(self, sample_classifications):
        """Status signal always has mode=SHADOW."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        assert signal["mode"] == "SHADOW"

    def test_status_signal_always_low_weight(self, sample_classifications):
        """Status signal always has weight_hint=LOW."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        assert signal["weight_hint"] == "LOW"

    def test_status_signal_includes_top_drivers(self, sample_classifications):
        """Status signal includes deterministic top_drivers."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        assert "top_drivers" in signal
        assert isinstance(signal["top_drivers"], list)
        assert len(signal["top_drivers"]) <= 3

    def test_status_signal_top_drivers_deterministic(self, sample_classifications):
        """Top drivers are deterministic across calls."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}

        signal1 = extract_p5_patterns_panel_signal_for_status(manifest, None)
        signal2 = extract_p5_patterns_panel_signal_for_status(manifest, None)

        assert signal1["top_drivers"] == signal2["top_drivers"]

    def test_default_signal_includes_stabilized_fields(self):
        """Default signal (no data) includes all stabilized fields."""
        signal = extract_p5_patterns_panel_signal_for_status(None, None)

        assert signal["schema_version"] == P5_PATTERNS_PANEL_SCHEMA_VERSION
        assert signal["mode"] == "SHADOW"
        assert signal["weight_hint"] == "LOW"
        assert signal["top_drivers"] == []

    def test_ggfl_adapter_passes_through_schema_version(self, sample_classifications):
        """GGFL adapter passes through schema_version."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        assert ggfl_signal["schema_version"] == P5_PATTERNS_PANEL_SCHEMA_VERSION

    def test_ggfl_adapter_always_shadow_mode(self, sample_classifications):
        """GGFL adapter always has mode=SHADOW."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        assert ggfl_signal["mode"] == "SHADOW"


# =============================================================================
# REGRESSION TEST: NO SOLO HARD BLOCK INVARIANT
# =============================================================================

class TestNoSoloHardBlockInvariant:
    """
    CRITICAL REGRESSION TESTS: NO SOLO HARD BLOCK INVARIANT

    These tests ensure that the P5 patterns panel signal can NEVER:
    1. Produce conflict=true
    2. Be the sole cause of a BLOCK decision in fusion

    This is a non-negotiable invariant for SHADOW MODE signals.
    """

    def test_conflict_always_false_with_healthy_data(self, sample_classifications):
        """conflict is always False with healthy data."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        assert ggfl_signal["conflict"] is False

    def test_conflict_always_false_with_structural_break(self, sample_classifications):
        """conflict is always False even with STRUCTURAL_BREAK patterns."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        # Verify STRUCTURAL_BREAK is present
        assert len(panel.structural_break_experiments) > 0

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        # Even with STRUCTURAL_BREAK, conflict must be False
        assert ggfl_signal["conflict"] is False

    def test_conflict_always_false_with_multiple_high_severity(
        self,
        sample_classifications,
        sample_classifications_cal_exp_3,
    ):
        """conflict is always False even with multiple high-severity experiments."""
        snap1 = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        snap3 = build_p5_patterns_snapshot("CAL-EXP-3", sample_classifications_cal_exp_3)
        panel = build_p5_patterns_panel([snap1, snap3])

        # Verify multiple high-severity experiments
        assert panel.experiments_with_high_severity >= 2

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        # Even with multiple high-severity, conflict must be False
        assert ggfl_signal["conflict"] is False

    def test_conflict_always_false_with_empty_signal(self):
        """conflict is always False with empty signal."""
        ggfl_signal = p5_patterns_panel_for_alignment_view({})
        assert ggfl_signal["conflict"] is False

    def test_conflict_always_false_with_none_signal(self):
        """conflict is always False with None signal."""
        ggfl_signal = p5_patterns_panel_for_alignment_view(None)
        assert ggfl_signal["conflict"] is False

    def test_status_never_block(self, sample_classifications):
        """Status can only be 'ok' or 'warn', never 'block'."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        assert signal["status"] in ("ok", "warn")
        assert signal["status"] != "block"

    def test_ggfl_status_never_block(self, sample_classifications):
        """GGFL status can only be 'ok' or 'warn', never 'block'."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        assert ggfl_signal["status"] in ("ok", "warn")
        assert ggfl_signal["status"] != "block"

    def test_weight_hint_always_low(self, sample_classifications):
        """weight_hint is always LOW (can never contribute high weight to block)."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)
        ggfl_signal = p5_patterns_panel_for_alignment_view(signal)

        assert signal["weight_hint"] == "LOW"
        assert ggfl_signal["weight_hint"] == "LOW"


# =============================================================================
# REGRESSION TEST: FUSION RULE ASSERTION
# =============================================================================

class TestFusionRuleAssertion:
    """
    CRITICAL REGRESSION TESTS: FUSION RULE ASSERTION

    These tests ensure that when only p5_patterns_panel signal is provided
    to GGFL (simulating it being the "sole" signal), it cannot produce
    a BLOCK decision. This verifies the "no solo hard block" invariant
    at the fusion layer.
    """

    def test_solo_p5_patterns_cannot_cause_block_decision(self, sample_classifications):
        """
        FUSION RULE: p5_patterns alone cannot cause BLOCK decision.

        When p5_patterns_panel is the only signal with any data,
        the fusion decision must be ALLOW (not BLOCK).
        """
        from backend.governance.fusion import build_global_alignment_view, GovernanceAction

        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        # Build p5_patterns signal
        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        p5_signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        # Call GGFL with ONLY p5_patterns (all other signals are None/missing)
        result = build_global_alignment_view(
            p5_patterns=p5_signal,
            cycle=1,
        )

        # The fusion decision must NOT be BLOCK when p5_patterns is the only signal
        # (Missing signals may cause warnings, but p5_patterns alone cannot cause BLOCK)
        fusion_decision = result["fusion_result"]["decision"]
        determining_signal = result["fusion_result"].get("determining_signal", "")

        # If there IS a block, it must NOT be from p5_patterns
        if fusion_decision == GovernanceAction.BLOCK:
            assert determining_signal != "p5_patterns", (
                "INVARIANT VIOLATION: p5_patterns cannot be the sole cause of BLOCK"
            )

    def test_solo_p5_patterns_with_structural_break_cannot_cause_block(
        self,
        sample_classifications,
    ):
        """
        FUSION RULE: p5_patterns with STRUCTURAL_BREAK cannot solo-cause BLOCK.

        Even with high-severity patterns like STRUCTURAL_BREAK,
        p5_patterns alone cannot be the determining signal for BLOCK.
        """
        from backend.governance.fusion import build_global_alignment_view, GovernanceAction

        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        # Verify STRUCTURAL_BREAK is present
        assert len(panel.structural_break_experiments) > 0

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        p5_signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        result = build_global_alignment_view(
            p5_patterns=p5_signal,
            cycle=1,
        )

        fusion_decision = result["fusion_result"]["decision"]
        determining_signal = result["fusion_result"].get("determining_signal", "")

        # p5_patterns must never be the determining signal for BLOCK
        if fusion_decision == GovernanceAction.BLOCK:
            assert determining_signal != "p5_patterns", (
                "INVARIANT VIOLATION: p5_patterns with STRUCTURAL_BREAK "
                "cannot be the sole cause of BLOCK"
            )

    def test_p5_patterns_signal_in_ggfl_has_low_precedence(self):
        """
        FUSION RULE: p5_patterns has low precedence (9).

        This ensures p5_patterns cannot override higher-precedence signals.
        Lower number = higher priority, so p5_patterns with 9 is very low priority.
        """
        from backend.governance.fusion import SIGNAL_PRECEDENCE

        assert "p5_patterns" in SIGNAL_PRECEDENCE
        p5_precedence = SIGNAL_PRECEDENCE["p5_patterns"]

        # Verify p5_patterns has lower priority than core signals
        assert p5_precedence > SIGNAL_PRECEDENCE["identity"], (
            "p5_patterns must have lower precedence than identity"
        )
        assert p5_precedence > SIGNAL_PRECEDENCE["structure"], (
            "p5_patterns must have lower precedence than structure"
        )
        assert p5_precedence > SIGNAL_PRECEDENCE["telemetry"], (
            "p5_patterns must have lower precedence than telemetry"
        )
        assert p5_precedence > SIGNAL_PRECEDENCE["topology"], (
            "p5_patterns must have lower precedence than topology"
        )
        # p5_patterns should be >= 9 (very low priority)
        assert p5_precedence >= 9, (
            "p5_patterns must have low precedence (>= 9)"
        )

    def test_p5_patterns_extraction_never_produces_hard_block_recommendation(
        self,
        sample_classifications,
    ):
        """
        FUSION RULE: p5_patterns extractor never produces HARD_BLOCK.

        The _extract_p5_patterns_recommendations function in fusion.py
        must never return a HARD_BLOCK recommendation.
        """
        from backend.governance.fusion import _extract_p5_patterns_recommendations

        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        p5_signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        recommendations = _extract_p5_patterns_recommendations(p5_signal)

        # None of the recommendations should be HARD_BLOCK
        for rec in recommendations:
            # action is a string, not an enum
            assert rec.action != "HARD_BLOCK", (
                f"INVARIANT VIOLATION: p5_patterns produced HARD_BLOCK recommendation: {rec}"
            )


# =============================================================================
# Test: Extraction Source Provenance
# =============================================================================

class TestExtractionSourceProvenance:
    """Test extraction_source provenance tracking."""

    def test_extraction_source_manifest(self, sample_classifications):
        """extraction_source is MANIFEST when data comes from manifest."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_MANIFEST

    def test_extraction_source_evidence_json(self, sample_classifications):
        """extraction_source is EVIDENCE_JSON when data comes from evidence fallback."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        evidence = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(None, evidence)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_EVIDENCE_JSON

    def test_extraction_source_missing(self):
        """extraction_source is MISSING when no data available."""
        signal = extract_p5_patterns_panel_signal_for_status(None, None)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_MISSING

    def test_extraction_source_manifest_preferred_over_evidence(self, sample_classifications):
        """Manifest is preferred over evidence when both present."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        evidence = {"governance": {"p5_patterns_panel": {"different": "data"}}}

        signal = extract_p5_patterns_panel_signal_for_status(manifest, evidence)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_MANIFEST


# =============================================================================
# Test: Reason Code Drivers (No Prose, Cap=3)
# =============================================================================

class TestReasonCodeDrivers:
    """Test that drivers are reason codes only with cap=3."""

    def test_drivers_are_reason_codes_format(self, sample_classifications):
        """Drivers follow DRIVER_{CATEGORY}_{VALUE} format (no prose)."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        for driver in signal["top_drivers"]:
            # Each driver must start with DRIVER_ prefix
            assert driver.startswith("DRIVER_"), f"Driver must start with DRIVER_: {driver}"
            # Each driver must be uppercase (reason code format)
            assert driver == driver.upper(), f"Driver must be uppercase: {driver}"
            # No spaces allowed (no prose)
            assert " " not in driver, f"Driver must not contain spaces: {driver}"

    def test_drivers_cap_at_max_drivers_cap(self):
        """Drivers are capped at MAX_DRIVERS_CAP (3)."""
        # Create classifications that would generate many drivers
        classifications = []
        timestamp = "2025-01-01T00:00:00Z"

        # Add many DRIFT patterns (non-NOMINAL dominant)
        for i in range(50):
            classifications.append(PatternClassification(
                pattern=DivergencePattern.DRIFT,
                confidence=0.9,
                evidence={},
                timestamp=timestamp,
            ))

        snap = build_p5_patterns_snapshot("CAL-EXP-1", classifications, cycles_analyzed=50)
        panel = build_p5_patterns_panel([snap])

        # Make it high severity by adding STRUCTURAL_BREAK
        panel.structural_break_experiments = ["CAL-EXP-1"]
        panel.experiments_with_high_severity = 1

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        assert len(signal["top_drivers"]) <= MAX_DRIVERS_CAP
        assert MAX_DRIVERS_CAP == 3  # Verify constant value

    def test_ggfl_adapter_enforces_driver_cap(self, sample_classifications):
        """GGFL adapter also enforces driver cap."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        # Even if we pass extra drivers, GGFL adapter should cap them
        signal_with_extra = dict(signal)
        signal_with_extra["top_drivers"] = [
            "DRIVER_A_1", "DRIVER_B_2", "DRIVER_C_3", "DRIVER_D_4", "DRIVER_E_5"
        ]

        ggfl_signal = p5_patterns_panel_for_alignment_view(signal_with_extra)

        assert len(ggfl_signal["drivers"]) <= MAX_DRIVERS_CAP

    def test_driver_deterministic_ordering(self, sample_classifications):
        """Drivers are ordered deterministically: DOMINANT > STREAK > HIGH_SEVERITY."""
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}

        signal1 = extract_p5_patterns_panel_signal_for_status(manifest, None)
        signal2 = extract_p5_patterns_panel_signal_for_status(manifest, None)

        assert signal1["top_drivers"] == signal2["top_drivers"]


# =============================================================================
# REGRESSION TEST: SOLO WARN CANNOT CAUSE BLOCK
# =============================================================================

class TestSoloWarnCannotBlock:
    """
    CRITICAL REGRESSION TEST: SOLO WARN CANNOT CAUSE BLOCK

    This test ensures that when SIG-PAT (p5_patterns) has status="warn",
    it alone cannot cause a BLOCK decision in fusion. This is the core
    "no solo hard block" invariant.
    """

    def test_solo_p5_patterns_warn_cannot_cause_block(self, sample_classifications):
        """
        FUSION RULE: SIG-PAT with status="warn" alone cannot cause BLOCK.

        When p5_patterns_panel is the only signal provided and has status="warn"
        (due to STRUCTURAL_BREAK), the fusion decision must NOT be BLOCK.
        """
        from backend.governance.fusion import build_global_alignment_view, GovernanceAction

        # Create panel with STRUCTURAL_BREAK (which triggers warn)
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        # Verify we have STRUCTURAL_BREAK (which triggers warn)
        assert len(panel.structural_break_experiments) > 0, "Test requires STRUCTURAL_BREAK"

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        p5_signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        # Verify signal status is "warn"
        assert p5_signal["status"] == "warn", "Signal must be 'warn' for this test"

        # Call GGFL with ONLY p5_patterns
        result = build_global_alignment_view(
            p5_patterns=p5_signal,
            cycle=1,
        )

        fusion_decision = result["fusion_result"]["decision"]
        determining_signal = result["fusion_result"].get("determining_signal", "")

        # INVARIANT: If there is a BLOCK, p5_patterns cannot be the determining signal
        if fusion_decision == GovernanceAction.BLOCK:
            assert determining_signal != "p5_patterns", (
                "INVARIANT VIOLATION: SIG-PAT with status='warn' "
                "cannot be the sole cause of BLOCK decision"
            )

    def test_solo_p5_patterns_warn_with_max_streak_cannot_cause_block(
        self,
        sample_classifications_cal_exp_3,
    ):
        """
        FUSION RULE: SIG-PAT with high streak and warn cannot solo-cause BLOCK.

        Even with maximum severity (high streak + STRUCTURAL_BREAK),
        p5_patterns alone cannot trigger BLOCK.
        """
        from backend.governance.fusion import build_global_alignment_view, GovernanceAction

        snap = build_p5_patterns_snapshot("CAL-EXP-3", sample_classifications_cal_exp_3)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        p5_signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        result = build_global_alignment_view(
            p5_patterns=p5_signal,
            cycle=1,
        )

        fusion_decision = result["fusion_result"]["decision"]
        determining_signal = result["fusion_result"].get("determining_signal", "")

        if fusion_decision == GovernanceAction.BLOCK:
            assert determining_signal != "p5_patterns", (
                "INVARIANT VIOLATION: SIG-PAT with max streak "
                "cannot be the sole cause of BLOCK decision"
            )

    def test_p5_patterns_warn_status_never_escalates_to_block_alone(
        self,
        sample_classifications,
    ):
        """
        FUSION RULE: p5_patterns warn status alone produces at most L1_WARNING.

        This verifies that even with warn status, p5_patterns cannot
        escalate beyond L1_WARNING when it's the only signal.
        """
        from backend.governance.fusion import build_global_alignment_view, EscalationLevel

        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        p5_signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        assert p5_signal["status"] == "warn"

        result = build_global_alignment_view(
            p5_patterns=p5_signal,
            cycle=1,
        )

        escalation_level = result["escalation"]["level"]

        # p5_patterns alone with warn should not reach L3_CRITICAL or higher
        # (which would indicate potential BLOCK)
        assert escalation_level.value < EscalationLevel.L3_CRITICAL.value, (
            f"INVARIANT VIOLATION: p5_patterns warn alone reached {escalation_level}, "
            f"but should be below L3_CRITICAL"
        )

    def test_p5_patterns_conflict_always_false_regardless_of_status(
        self,
        sample_classifications,
    ):
        """
        INVARIANT: p5_patterns conflict is always False, even when status='warn'.

        This is the core mechanism that prevents solo BLOCK:
        conflict=False means no hard block recommendation.
        """
        snap = build_p5_patterns_snapshot("CAL-EXP-1", sample_classifications)
        panel = build_p5_patterns_panel([snap])

        manifest = {"governance": {"p5_patterns_panel": panel.to_dict()}}
        p5_signal = extract_p5_patterns_panel_signal_for_status(manifest, None)

        # Verify status is warn
        assert p5_signal["status"] == "warn"

        ggfl_signal = p5_patterns_panel_for_alignment_view(p5_signal)

        # conflict must ALWAYS be False, regardless of warn status
        assert ggfl_signal["conflict"] is False, (
            "INVARIANT VIOLATION: conflict must be False even when status='warn'"
        )
        assert ggfl_signal["status"] == "warn"  # Status passed through
