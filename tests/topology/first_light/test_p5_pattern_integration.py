"""
End-to-end tests for P5 Pattern Classifier integration with Evidence Pack and GGFL.

Tests verify:
1. Pattern tags file generation from P4 runner
2. Evidence pack detection and manifest integration
3. GGFL signal contribution and conflict detection
4. SHADOW MODE invariants throughout pipeline

See: docs/system_law/GGFL_P5_Pattern_Test_Plan.md
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from backend.topology.first_light.p5_pattern_classifier import (
    TDAPatternClassifier,
    DivergencePattern,
    attach_tda_patterns_to_evidence,
    P5_PATTERN_SCHEMA_VERSION,
)
from backend.topology.first_light.evidence_pack import (
    detect_p5_pattern_tags,
    P5PatternTagsReference,
    P5_PATTERN_TAGS_ARTIFACT,
)
from backend.governance.fusion import (
    build_global_alignment_view,
    EscalationLevel,
    GovernanceAction,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_run_dir():
    """Create temporary run directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def nominal_pattern_tags():
    """Nominal P5 pattern tags."""
    return {
        "schema_version": P5_PATTERN_SCHEMA_VERSION,
        "mode": "SHADOW",
        "timestamp": "2025-01-15T10:00:00Z",
        "run_id": "test-run-001",
        "cycles_analyzed": 100,
        "classification_summary": {
            "final_pattern": "NOMINAL",
            "final_streak": 0,
            "recalibration_triggered": False,
        },
        "signal_extensions": {
            "p5_telemetry": {
                "telemetry_validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.92,
                "divergence_pattern": "NOMINAL",
                "divergence_pattern_streak": 0,
                "recalibration_triggered": False,
            },
            "p5_topology": {
                "attractor_miss_rate": 0.01,
                "twin_omega_alignment": True,
                "transient_tracking_quality": 0.95,
            },
            "p5_replay": {
                "twin_prediction_divergence": 0.02,
                "divergence_bias": 0.005,
                "divergence_variance": 0.008,
            },
        },
        "window_summaries": [
            {"window_start": 0, "window_end": 10, "dominant_pattern": "NOMINAL", "mean_confidence": 0.9},
        ],
        "shadow_mode_invariants": {
            "no_enforcement": True,
            "logged_only": True,
            "observation_only": True,
        },
    }


@pytest.fixture
def structural_break_pattern_tags():
    """STRUCTURAL_BREAK P5 pattern tags for conflict testing."""
    return {
        "schema_version": P5_PATTERN_SCHEMA_VERSION,
        "mode": "SHADOW",
        "timestamp": "2025-01-15T10:00:00Z",
        "run_id": "test-run-002",
        "cycles_analyzed": 100,
        "classification_summary": {
            "final_pattern": "STRUCTURAL_BREAK",
            "final_streak": 3,
            "recalibration_triggered": True,
        },
        "signal_extensions": {
            "p5_telemetry": {
                "telemetry_validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.78,
                "divergence_pattern": "STRUCTURAL_BREAK",
                "divergence_pattern_streak": 3,
                "recalibration_triggered": True,
            },
            "p5_topology": {
                "attractor_miss_rate": 0.35,
                "twin_omega_alignment": False,
                "transient_tracking_quality": 0.30,
            },
            "p5_replay": {
                "twin_prediction_divergence": 0.22,
                "divergence_bias": 0.08,
                "divergence_variance": 0.055,
            },
        },
        "window_summaries": [],
        "shadow_mode_invariants": {
            "no_enforcement": True,
            "logged_only": True,
            "observation_only": True,
        },
    }


# =============================================================================
# Test: Pattern Tags File Generation
# =============================================================================

class TestPatternTagsFileGeneration:
    """Tests for P5 pattern tags file generation."""

    def test_classifier_builds_valid_pattern_tags(self):
        """Verify TDAPatternClassifier produces valid pattern tags structure."""
        classifier = TDAPatternClassifier()

        # Feed some samples
        for _ in range(30):
            classifier.classify(
                delta_p=0.02,
                p_real=0.5,
                p_twin=0.52,
                omega_real=True,
                omega_twin=True,
            )

        # This would be called by runner - simulate it
        p5_telemetry = classifier.get_p5_telemetry_extension(
            validation_status="VALIDATED_REAL",
            validation_confidence=0.9,
        )

        assert p5_telemetry.telemetry_validation_status == "VALIDATED_REAL"
        assert p5_telemetry.validation_confidence == 0.9
        assert p5_telemetry.divergence_pattern is not None

    def test_attach_tda_patterns_includes_governance_key(self):
        """Verify attach_tda_patterns_to_evidence adds governance key."""
        evidence = {"existing": "data"}
        classifier = TDAPatternClassifier()

        result = attach_tda_patterns_to_evidence(evidence, classifier)

        assert "governance" in result
        assert "p5_pattern_classification" in result["governance"]
        assert result["governance"]["p5_pattern_classification"]["mode"] == "SHADOW"

    def test_attach_preserves_existing_governance(self):
        """Verify existing governance data is preserved."""
        evidence = {
            "governance": {
                "other_signal": {"data": "preserved"},
            },
        }
        classifier = TDAPatternClassifier()

        result = attach_tda_patterns_to_evidence(evidence, classifier)

        assert result["governance"]["other_signal"]["data"] == "preserved"
        assert "p5_pattern_classification" in result["governance"]


# =============================================================================
# Test: Evidence Pack Detection
# =============================================================================

class TestEvidencePackDetection:
    """Tests for P5 pattern tags detection in evidence pack."""

    def test_detect_p5_pattern_tags_in_p4_shadow(self, temp_run_dir, nominal_pattern_tags):
        """Verify detection in p4_shadow subdirectory."""
        p4_dir = temp_run_dir / "p4_shadow"
        p4_dir.mkdir()

        tags_path = p4_dir / P5_PATTERN_TAGS_ARTIFACT
        with open(tags_path, "w") as f:
            json.dump(nominal_pattern_tags, f)

        result = detect_p5_pattern_tags(temp_run_dir)

        assert result is not None
        assert isinstance(result, P5PatternTagsReference)
        assert result.final_pattern == "NOMINAL"
        assert result.mode == "SHADOW"
        assert result.shadow_mode_invariants_ok is True

    def test_detect_p5_pattern_tags_in_root(self, temp_run_dir, nominal_pattern_tags):
        """Verify detection in root directory."""
        tags_path = temp_run_dir / P5_PATTERN_TAGS_ARTIFACT
        with open(tags_path, "w") as f:
            json.dump(nominal_pattern_tags, f)

        result = detect_p5_pattern_tags(temp_run_dir)

        assert result is not None
        assert result.final_pattern == "NOMINAL"

    def test_detect_returns_none_when_missing(self, temp_run_dir):
        """Verify None returned when file is missing."""
        result = detect_p5_pattern_tags(temp_run_dir)
        assert result is None

    def test_detect_extracts_structural_break(self, temp_run_dir, structural_break_pattern_tags):
        """Verify correct extraction of STRUCTURAL_BREAK pattern."""
        tags_path = temp_run_dir / P5_PATTERN_TAGS_ARTIFACT
        with open(tags_path, "w") as f:
            json.dump(structural_break_pattern_tags, f)

        result = detect_p5_pattern_tags(temp_run_dir)

        assert result is not None
        assert result.final_pattern == "STRUCTURAL_BREAK"
        assert result.final_streak == 3
        assert result.recalibration_triggered is True

    def test_detect_validates_shadow_mode_invariants(self, temp_run_dir, nominal_pattern_tags):
        """Verify shadow mode invariants are validated."""
        # Create tags with missing invariants
        broken_tags = dict(nominal_pattern_tags)
        broken_tags["shadow_mode_invariants"] = {
            "no_enforcement": True,
            "logged_only": False,  # Missing invariant
            "observation_only": True,
        }

        tags_path = temp_run_dir / P5_PATTERN_TAGS_ARTIFACT
        with open(tags_path, "w") as f:
            json.dump(broken_tags, f)

        result = detect_p5_pattern_tags(temp_run_dir)

        assert result is not None
        assert result.shadow_mode_invariants_ok is False


# =============================================================================
# Test: GGFL Integration
# =============================================================================

class TestGGFLIntegration:
    """Tests for P5 patterns integration with GGFL."""

    def test_ggfl_accepts_p5_patterns_signal(self):
        """Verify GGFL accepts p5_patterns as a signal."""
        result = build_global_alignment_view(
            p5_patterns={
                "final_pattern": "NOMINAL",
                "final_streak": 0,
            },
            cycle=100,
        )

        assert "signals" in result
        assert "p5_patterns" in result["signals"]

    def test_nominal_pattern_produces_allow(self):
        """Verify NOMINAL pattern produces ALLOW decision."""
        result = build_global_alignment_view(
            p5_patterns={
                "final_pattern": "NOMINAL",
                "final_streak": 0,
            },
            identity={
                "block_hash_valid": True,
                "chain_continuous": True,
            },
            cycle=100,
        )

        assert result["fusion_result"]["decision"] == "ALLOW"

    def test_structural_break_alone_does_not_hard_block(self):
        """Verify STRUCTURAL_BREAK alone produces soft block, not hard block."""
        result = build_global_alignment_view(
            p5_patterns={
                "final_pattern": "STRUCTURAL_BREAK",
                "final_streak": 3,
            },
            # No other signals that would conflict
            identity={
                "block_hash_valid": True,
                "chain_continuous": True,
            },
            structure={
                "dag_coherent": True,
                "min_cut_capacity": 0.5,  # Above threshold - no conflict
            },
            cycle=100,
        )

        # Should NOT be hard block from p5_patterns alone
        assert result["fusion_result"]["is_hard"] is False

    def test_structural_break_with_dag_tension_creates_conflict(self):
        """Verify CSC-P5-001: STRUCTURAL_BREAK + low min_cut triggers conflict."""
        result = build_global_alignment_view(
            p5_patterns={
                "final_pattern": "STRUCTURAL_BREAK",
                "final_streak": 3,
            },
            structure={
                "dag_coherent": True,
                "min_cut_capacity": 0.15,  # Below 0.2 threshold
            },
            cycle=100,
        )

        # Should detect CSC-P5-001 conflict
        conflict_ids = [c["rule_id"] for c in result["conflict_detections"]]
        assert "CSC-P5-001" in conflict_ids

    def test_attractor_miss_with_safe_omega_creates_conflict(self):
        """Verify CSC-P5-003: ATTRACTOR_MISS while in safe region triggers conflict."""
        result = build_global_alignment_view(
            p5_patterns={
                "final_pattern": "ATTRACTOR_MISS",
                "final_streak": 5,
            },
            topology={
                "within_omega": True,
                "p5_twin": {
                    "attractor_miss_rate": 0.25,  # Above 0.2 threshold
                },
            },
            cycle=100,
        )

        conflict_ids = [c["rule_id"] for c in result["conflict_detections"]]
        assert "CSC-P5-003" in conflict_ids

    def test_drift_pattern_produces_warning(self):
        """Verify DRIFT pattern produces WARNING escalation."""
        result = build_global_alignment_view(
            p5_patterns={
                "final_pattern": "DRIFT",
                "final_streak": 3,
            },
            cycle=100,
        )

        # Should have p5_patterns recommendation with WARNING
        p5_recs = [r for r in result["recommendations"] if r["signal_id"] == "p5_patterns"]
        assert len(p5_recs) > 0
        assert any(r["action"] == "WARNING" for r in p5_recs)

    def test_p5_patterns_in_signal_summary(self):
        """Verify p5_patterns appears in signal summary."""
        result = build_global_alignment_view(
            p5_patterns={
                "final_pattern": "DRIFT",
                "final_streak": 2,
            },
            cycle=100,
        )

        assert "signal_summary" in result
        signal_summary = result["signal_summary"]
        assert "p5_patterns" in signal_summary


# =============================================================================
# Test: SHADOW MODE Contract
# =============================================================================

class TestShadowModeContract:
    """Tests verifying SHADOW MODE contract is enforced throughout pipeline."""

    def test_pattern_tags_always_have_shadow_mode(self, temp_run_dir, nominal_pattern_tags):
        """Verify pattern tags always include mode=SHADOW."""
        tags_path = temp_run_dir / P5_PATTERN_TAGS_ARTIFACT
        with open(tags_path, "w") as f:
            json.dump(nominal_pattern_tags, f)

        result = detect_p5_pattern_tags(temp_run_dir)

        assert result.mode == "SHADOW"

    def test_ggfl_output_always_shadow_mode(self):
        """Verify GGFL output always includes shadow mode marker."""
        result = build_global_alignment_view(
            p5_patterns={
                "final_pattern": "STRUCTURAL_BREAK",
                "final_streak": 5,
            },
            cycle=100,
        )

        assert result["mode"] == "shadow"

    def test_p5_patterns_never_solo_hard_block(self):
        """Verify p5_patterns signal alone NEVER produces HARD_BLOCK."""
        # Test all pattern types
        patterns = [
            ("DRIFT", 10),
            ("NOISE_AMPLIFICATION", 10),
            ("PHASE_LAG", 15),
            ("ATTRACTOR_MISS", 10),
            ("TRANSIENT_MISS", 10),
            ("STRUCTURAL_BREAK", 10),
        ]

        for pattern, streak in patterns:
            result = build_global_alignment_view(
                p5_patterns={
                    "final_pattern": pattern,
                    "final_streak": streak,
                },
                # All other signals healthy
                identity={
                    "block_hash_valid": True,
                    "chain_continuous": True,
                },
                structure={
                    "dag_coherent": True,
                    "min_cut_capacity": 0.5,
                },
                telemetry={
                    "lean_healthy": True,
                    "db_healthy": True,
                    "redis_healthy": True,
                },
                cycle=100,
            )

            # P5 patterns alone should NEVER cause hard block
            assert result["fusion_result"]["is_hard"] is False, (
                f"P5 pattern {pattern} caused solo hard block"
            )


# =============================================================================
# Test: Escalation Levels
# =============================================================================

class TestP5PatternsEscalation:
    """Tests for P5 patterns contribution to escalation levels."""

    def test_warning_patterns_contribute_to_l1(self):
        """Verify warning-level patterns contribute to L1_WARNING."""
        result = build_global_alignment_view(
            p5_patterns={
                "final_pattern": "DRIFT",
                "final_streak": 3,
            },
            cycle=100,
        )

        # With only p5_patterns WARNING, escalation should be L1
        assert result["escalation"]["level"] >= EscalationLevel.L1_WARNING

    def test_blocking_patterns_contribute_to_escalation(self):
        """Verify blocking patterns contribute to higher escalation."""
        result = build_global_alignment_view(
            p5_patterns={
                "final_pattern": "STRUCTURAL_BREAK",
                "final_streak": 5,
            },
            # Add another signal that also blocks
            telemetry={
                "lean_healthy": True,
                "db_healthy": True,
                "error_rate": 0.15,  # Elevated - adds warning
            },
            cycle=100,
        )

        # With p5 block + another warning, escalation should reflect combination
        assert result["escalation"]["level"] >= EscalationLevel.L1_WARNING


# =============================================================================
# Test: Full Pipeline Integration
# =============================================================================

class TestFullPipelineIntegration:
    """End-to-end tests for full P5 pattern pipeline."""

    def test_pattern_tags_to_ggfl_pipeline(self, temp_run_dir, structural_break_pattern_tags):
        """Test full pipeline: tags file → detection → GGFL integration."""
        # Step 1: Write pattern tags file
        p4_dir = temp_run_dir / "p4_shadow"
        p4_dir.mkdir()
        tags_path = p4_dir / P5_PATTERN_TAGS_ARTIFACT
        with open(tags_path, "w") as f:
            json.dump(structural_break_pattern_tags, f)

        # Step 2: Detect pattern tags
        tags_ref = detect_p5_pattern_tags(temp_run_dir)
        assert tags_ref is not None
        assert tags_ref.final_pattern == "STRUCTURAL_BREAK"

        # Step 3: Build p5_patterns signal from tags
        p5_signal = {
            "final_pattern": tags_ref.final_pattern,
            "final_streak": tags_ref.final_streak,
            "recalibration_triggered": tags_ref.recalibration_triggered,
        }

        # Step 4: Feed to GGFL with conflicting structure signal
        result = build_global_alignment_view(
            p5_patterns=p5_signal,
            structure={
                "dag_coherent": True,
                "min_cut_capacity": 0.15,  # Triggers CSC-P5-001
            },
            cycle=100,
        )

        # Step 5: Verify conflict detected
        conflict_ids = [c["rule_id"] for c in result["conflict_detections"]]
        assert "CSC-P5-001" in conflict_ids

        # Step 6: Verify SHADOW MODE maintained
        assert result["mode"] == "shadow"
        assert result["fusion_result"]["is_hard"] is False  # P5 can't solo hard block
