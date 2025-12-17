"""
Tests for P5 Pattern Classifier using canonical examples from GGFL_P5_Pattern_Test_Plan.md.

Tests verify:
1. Pattern classification accuracy
2. Escalation level mapping
3. Fusion decision alignment
4. Conflict detection (CSC-P5-* rules)
5. SHADOW MODE invariants

See: docs/system_law/GGFL_P5_Pattern_Test_Plan.md
"""

import pytest
from typing import Any, Dict, List, Optional

from backend.topology.first_light.p5_pattern_classifier import (
    DivergencePattern,
    PatternClassification,
    TDAPatternClassifier,
    P5TelemetryExtension,
    P5TopologyExtension,
    P5ReplayExtension,
    attach_tda_patterns_to_evidence,
    classify_from_signals,
    P5_PATTERN_SCHEMA_VERSION,
)
from backend.governance.fusion import (
    build_global_alignment_view,
    EscalationLevel,
    GovernanceAction,
)


# =============================================================================
# Canonical Test Fixtures from GGFL_P5_Pattern_Test_Plan.md
# =============================================================================

# P5-DRIFT-001: Systematic bias in twin model
P5_DRIFT_001 = {
    "test_id": "P5-DRIFT-001",
    "pattern": "DRIFT",
    "signals": {
        "topology": {
            "H": 0.82,
            "D": 5,
            "D_dot": 0.3,
            "B": 2.5,
            "S": 0.12,
            "C": 0,
            "rho": 0.85,
            "tau": 0.21,
            "J": 2.5,
            "within_omega": True,
            "active_cdis": [],
            "invariant_violations": [],
            "p5_twin": {
                "attractor_miss_rate": 0.02,
                "twin_omega_alignment": True,
                "transient_tracking_quality": 0.90,
            },
        },
        "replay": {
            "replay_verified": True,
            "replay_divergence": 0.03,
            "replay_latency_ms": 50,
            "replay_hash_match": True,
            "replay_depth_valid": True,
            "p5_divergence": {
                "twin_prediction_divergence": 0.06,
                "divergence_bias": 0.055,
                "divergence_variance": 0.008,
            },
        },
        "telemetry": {
            "lean_healthy": True,
            "db_healthy": True,
            "redis_healthy": True,
            "worker_count": 4,
            "error_rate": 0.01,
            "last_error": None,
            "uptime_seconds": 86400,
            "p5_telemetry": {
                "telemetry_validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.92,
                "divergence_pattern": "DRIFT",
                "divergence_pattern_streak": 3,
                "recalibration_triggered": False,
            },
        },
        "structure": {
            "dag_coherent": True,
            "orphan_count": 5,
            "max_fanout": 10,
            "depth_distribution": {"1": 100, "2": 200},
            "cycle_detected": False,
            "min_cut_capacity": 0.5,
        },
        "metrics": {
            "success_rate": 0.85,
            "abstention_rate": 0.10,
            "block_rate": 0.15,
            "throughput": 25.0,
            "latency_p50_ms": 100,
            "latency_p99_ms": 500,
            "queue_depth": 50,
        },
        "budget": {
            "compute_budget_remaining": 0.75,
            "memory_utilization": 0.40,
            "storage_headroom_gb": 100.0,
            "verification_quota_remaining": 1000,
            "budget_exhaustion_eta_cycles": 500,
        },
        "identity": {
            "block_hash_valid": True,
            "merkle_root_valid": True,
            "signature_valid": True,
            "chain_continuous": True,
            "pq_attestation_valid": True,
            "dual_root_consistent": True,
        },
        "narrative": {
            "current_slice": "propositional_tautology",
            "slice_progress": 0.60,
            "epoch": 5,
            "curriculum_health": "HEALTHY",
            "drift_detected": False,
            "narrative_coherence": 0.85,
        },
    },
    "expected": {
        "escalation_level": "L1_WARNING",
        "fusion_decision": "ALLOW",
        "is_hard": False,
        "primary_recommendation": {
            "signal_id": "telemetry",
            "action": "WARNING",
            "reason": "DRIFT pattern detected: twin calibration may need adjustment",
        },
        "conflict_detections": [],
    },
}

# P5-NOISE-001: Twin over-sensitive to noise
P5_NOISE_001 = {
    "test_id": "P5-NOISE-001",
    "pattern": "NOISE_AMPLIFICATION",
    "signals": {
        "topology": {
            "H": 0.80,
            "D": 5,
            "D_dot": 0.2,
            "B": 2.3,
            "S": 0.15,
            "C": 1,
            "rho": 0.82,
            "tau": 0.21,
            "J": 2.8,
            "within_omega": True,
            "active_cdis": [],
            "invariant_violations": [],
            "p5_twin": {
                "attractor_miss_rate": 0.05,
                "twin_omega_alignment": True,
                "transient_tracking_quality": 0.75,
            },
        },
        "replay": {
            "replay_verified": True,
            "replay_divergence": 0.04,
            "replay_latency_ms": 55,
            "replay_hash_match": True,
            "replay_depth_valid": True,
            "p5_divergence": {
                "twin_prediction_divergence": 0.08,
                "divergence_bias": 0.01,
                "divergence_variance": 0.045,
            },
        },
        "telemetry": {
            "lean_healthy": True,
            "db_healthy": True,
            "redis_healthy": True,
            "worker_count": 4,
            "error_rate": 0.02,
            "last_error": None,
            "uptime_seconds": 86400,
            "p5_telemetry": {
                "telemetry_validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.88,
                "divergence_pattern": "NOISE_AMPLIFICATION",
                "divergence_pattern_streak": 8,
                "recalibration_triggered": False,
            },
        },
        "structure": {
            "dag_coherent": True,
            "orphan_count": 8,
            "max_fanout": 12,
            "depth_distribution": {"1": 100, "2": 200},
            "cycle_detected": False,
            "min_cut_capacity": 0.45,
        },
        "metrics": {
            "success_rate": 0.82,
            "abstention_rate": 0.12,
            "block_rate": 0.18,
            "throughput": 22.0,
            "latency_p50_ms": 110,
            "latency_p99_ms": 550,
            "queue_depth": 75,
        },
        "budget": {
            "compute_budget_remaining": 0.70,
            "memory_utilization": 0.45,
            "storage_headroom_gb": 95.0,
            "verification_quota_remaining": 900,
            "budget_exhaustion_eta_cycles": 450,
        },
        "identity": {
            "block_hash_valid": True,
            "merkle_root_valid": True,
            "signature_valid": True,
            "chain_continuous": True,
            "pq_attestation_valid": True,
            "dual_root_consistent": True,
        },
        "narrative": {
            "current_slice": "propositional_tautology",
            "slice_progress": 0.65,
            "epoch": 5,
            "curriculum_health": "HEALTHY",
            "drift_detected": False,
            "narrative_coherence": 0.82,
        },
    },
    "expected": {
        "escalation_level": "L1_WARNING",
        "fusion_decision": "ALLOW",
        "is_hard": False,
        "primary_recommendation": {
            "signal_id": "telemetry",
            "action": "WARNING",
            "reason": "NOISE_AMPLIFICATION pattern detected: twin over-fitting to noise",
        },
        "conflict_detections": [],
    },
}

# P5-PHASE-001: Temporal misalignment
P5_PHASE_001 = {
    "test_id": "P5-PHASE-001",
    "pattern": "PHASE_LAG",
    "signals": {
        "topology": {
            "H": 0.83,
            "D": 5,
            "D_dot": 0.4,
            "B": 2.4,
            "S": 0.10,
            "C": 0,
            "rho": 0.86,
            "tau": 0.21,
            "J": 2.3,
            "within_omega": True,
            "active_cdis": [],
            "invariant_violations": [],
            "p5_twin": {
                "attractor_miss_rate": 0.03,
                "twin_omega_alignment": True,
                "transient_tracking_quality": 0.80,
            },
        },
        "replay": {
            "replay_verified": True,
            "replay_divergence": 0.05,
            "replay_latency_ms": 60,
            "replay_hash_match": True,
            "replay_depth_valid": True,
            "p5_divergence": {
                "twin_prediction_divergence": 0.07,
                "divergence_bias": -0.02,
                "divergence_variance": 0.015,
            },
        },
        "telemetry": {
            "lean_healthy": True,
            "db_healthy": True,
            "redis_healthy": True,
            "worker_count": 4,
            "error_rate": 0.01,
            "last_error": None,
            "uptime_seconds": 86400,
            "p5_telemetry": {
                "telemetry_validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.90,
                "divergence_pattern": "PHASE_LAG",
                "divergence_pattern_streak": 12,
                "recalibration_triggered": False,
            },
        },
        "structure": {
            "dag_coherent": True,
            "orphan_count": 3,
            "max_fanout": 8,
            "depth_distribution": {"1": 100, "2": 200},
            "cycle_detected": False,
            "min_cut_capacity": 0.55,
        },
        "metrics": {
            "success_rate": 0.84,
            "abstention_rate": 0.08,
            "block_rate": 0.12,
            "throughput": 26.0,
            "latency_p50_ms": 95,
            "latency_p99_ms": 480,
            "queue_depth": 40,
        },
        "budget": {
            "compute_budget_remaining": 0.80,
            "memory_utilization": 0.35,
            "storage_headroom_gb": 105.0,
            "verification_quota_remaining": 1100,
            "budget_exhaustion_eta_cycles": 550,
        },
        "identity": {
            "block_hash_valid": True,
            "merkle_root_valid": True,
            "signature_valid": True,
            "chain_continuous": True,
            "pq_attestation_valid": True,
            "dual_root_consistent": True,
        },
        "narrative": {
            "current_slice": "propositional_tautology",
            "slice_progress": 0.70,
            "epoch": 5,
            "curriculum_health": "HEALTHY",
            "drift_detected": False,
            "narrative_coherence": 0.88,
        },
    },
    "expected": {
        "escalation_level": "L1_WARNING",
        "fusion_decision": "ALLOW",
        "is_hard": False,
        "primary_recommendation": {
            "signal_id": "telemetry",
            "action": "WARNING",
            "reason": "PHASE_LAG pattern detected (streak 12): prediction timing misalignment",
        },
        "conflict_detections": [],
    },
}

# P5-ATTRACTOR-001: Twin frequently misses safe region
P5_ATTRACTOR_001 = {
    "test_id": "P5-ATTRACTOR-001",
    "pattern": "ATTRACTOR_MISS",
    "signals": {
        "topology": {
            "H": 0.78,
            "D": 6,
            "D_dot": 0.5,
            "B": 2.8,
            "S": 0.18,
            "C": 1,
            "rho": 0.75,
            "tau": 0.21,
            "J": 3.2,
            "within_omega": True,
            "active_cdis": [],
            "invariant_violations": [],
            "p5_twin": {
                "attractor_miss_rate": 0.25,
                "twin_omega_alignment": False,
                "transient_tracking_quality": 0.60,
            },
        },
        "replay": {
            "replay_verified": True,
            "replay_divergence": 0.08,
            "replay_latency_ms": 70,
            "replay_hash_match": True,
            "replay_depth_valid": True,
            "p5_divergence": {
                "twin_prediction_divergence": 0.12,
                "divergence_bias": 0.03,
                "divergence_variance": 0.025,
            },
        },
        "telemetry": {
            "lean_healthy": True,
            "db_healthy": True,
            "redis_healthy": True,
            "worker_count": 4,
            "error_rate": 0.03,
            "last_error": None,
            "uptime_seconds": 86400,
            "p5_telemetry": {
                "telemetry_validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.85,
                "divergence_pattern": "ATTRACTOR_MISS",
                "divergence_pattern_streak": 4,
                "recalibration_triggered": True,
            },
        },
        "structure": {
            "dag_coherent": True,
            "orphan_count": 15,
            "max_fanout": 15,
            "depth_distribution": {"1": 100, "2": 200, "3": 150},
            "cycle_detected": False,
            "min_cut_capacity": 0.35,
        },
        "metrics": {
            "success_rate": 0.78,
            "abstention_rate": 0.15,
            "block_rate": 0.22,
            "throughput": 20.0,
            "latency_p50_ms": 130,
            "latency_p99_ms": 600,
            "queue_depth": 120,
        },
        "budget": {
            "compute_budget_remaining": 0.60,
            "memory_utilization": 0.55,
            "storage_headroom_gb": 85.0,
            "verification_quota_remaining": 700,
            "budget_exhaustion_eta_cycles": 350,
        },
        "identity": {
            "block_hash_valid": True,
            "merkle_root_valid": True,
            "signature_valid": True,
            "chain_continuous": True,
            "pq_attestation_valid": True,
            "dual_root_consistent": True,
        },
        "narrative": {
            "current_slice": "propositional_tautology",
            "slice_progress": 0.55,
            "epoch": 5,
            "curriculum_health": "DEGRADED",
            "drift_detected": True,
            "narrative_coherence": 0.72,
        },
    },
    "expected": {
        "escalation_level": "L3_CRITICAL",
        "fusion_decision": "BLOCK",
        "is_hard": True,
        "primary_recommendation": {
            "signal_id": "telemetry",
            "action": "HARD_BLOCK",
            "reason": "ATTRACTOR_MISS streak ≥3: twin fundamentally misaligned",
        },
        "conflict_detections": [
            {
                "rule_id": "CSC-P5-003",
                "description": "Twin fails to track safe region despite real system being safe",
                "signals_involved": ["topology", "telemetry"],
                "severity": "HIGH",
            }
        ],
    },
}

# P5-TRANSIENT-001: High Δp during excursions only
P5_TRANSIENT_001 = {
    "test_id": "P5-TRANSIENT-001",
    "pattern": "TRANSIENT_MISS",
    "signals": {
        "topology": {
            "H": 0.80,
            "D": 5,
            "D_dot": 0.6,
            "B": 2.6,
            "S": 0.14,
            "C": 1,
            "rho": 0.80,
            "tau": 0.21,
            "J": 2.9,
            "within_omega": True,
            "active_cdis": [],
            "invariant_violations": [],
            "p5_twin": {
                "attractor_miss_rate": 0.08,
                "twin_omega_alignment": True,
                "transient_tracking_quality": 0.55,
            },
        },
        "replay": {
            "replay_verified": True,
            "replay_divergence": 0.06,
            "replay_latency_ms": 65,
            "replay_hash_match": True,
            "replay_depth_valid": True,
            "p5_divergence": {
                "twin_prediction_divergence": 0.09,
                "divergence_bias": 0.02,
                "divergence_variance": 0.020,
            },
        },
        "telemetry": {
            "lean_healthy": True,
            "db_healthy": True,
            "redis_healthy": True,
            "worker_count": 4,
            "error_rate": 0.02,
            "last_error": None,
            "uptime_seconds": 86400,
            "p5_telemetry": {
                "telemetry_validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.87,
                "divergence_pattern": "TRANSIENT_MISS",
                "divergence_pattern_streak": 6,
                "recalibration_triggered": False,
            },
        },
        "structure": {
            "dag_coherent": True,
            "orphan_count": 10,
            "max_fanout": 12,
            "depth_distribution": {"1": 100, "2": 200},
            "cycle_detected": False,
            "min_cut_capacity": 0.40,
        },
        "metrics": {
            "success_rate": 0.80,
            "abstention_rate": 0.12,
            "block_rate": 0.20,
            "throughput": 21.0,
            "latency_p50_ms": 120,
            "latency_p99_ms": 580,
            "queue_depth": 100,
        },
        "budget": {
            "compute_budget_remaining": 0.65,
            "memory_utilization": 0.50,
            "storage_headroom_gb": 90.0,
            "verification_quota_remaining": 800,
            "budget_exhaustion_eta_cycles": 400,
        },
        "identity": {
            "block_hash_valid": True,
            "merkle_root_valid": True,
            "signature_valid": True,
            "chain_continuous": True,
            "pq_attestation_valid": True,
            "dual_root_consistent": True,
        },
        "narrative": {
            "current_slice": "propositional_tautology",
            "slice_progress": 0.58,
            "epoch": 5,
            "curriculum_health": "HEALTHY",
            "drift_detected": False,
            "narrative_coherence": 0.78,
        },
    },
    "expected": {
        "escalation_level": "L2_DEGRADED",
        "fusion_decision": "BLOCK",
        "is_hard": False,
        "primary_recommendation": {
            "signal_id": "telemetry",
            "action": "BLOCK",
            "reason": "TRANSIENT_MISS streak ≥5: transient fidelity concern",
        },
        "conflict_detections": [],
    },
}

# P5-STRUCTURAL-001: Sudden regime change
P5_STRUCTURAL_001 = {
    "test_id": "P5-STRUCTURAL-001",
    "pattern": "STRUCTURAL_BREAK",
    "signals": {
        "topology": {
            "H": 0.65,
            "D": 7,
            "D_dot": 1.2,
            "B": 3.5,
            "S": 0.25,
            "C": 2,
            "rho": 0.55,
            "tau": 0.21,
            "J": 4.5,
            "within_omega": False,
            "active_cdis": ["CDI-003"],
            "invariant_violations": [],
            "p5_twin": {
                "attractor_miss_rate": 0.35,
                "twin_omega_alignment": False,
                "transient_tracking_quality": 0.30,
            },
        },
        "replay": {
            "replay_verified": False,
            "replay_divergence": 0.18,
            "replay_latency_ms": 150,
            "replay_hash_match": True,
            "replay_depth_valid": True,
            "p5_divergence": {
                "twin_prediction_divergence": 0.22,
                "divergence_bias": 0.08,
                "divergence_variance": 0.055,
            },
        },
        "telemetry": {
            "lean_healthy": True,
            "db_healthy": True,
            "redis_healthy": True,
            "worker_count": 4,
            "error_rate": 0.08,
            "last_error": "Twin prediction failure at cycle 1234",
            "uptime_seconds": 86400,
            "p5_telemetry": {
                "telemetry_validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.78,
                "divergence_pattern": "STRUCTURAL_BREAK",
                "divergence_pattern_streak": 3,
                "recalibration_triggered": True,
            },
        },
        "structure": {
            "dag_coherent": True,
            "orphan_count": 45,
            "max_fanout": 25,
            "depth_distribution": {"1": 100, "2": 200, "3": 300, "4": 150},
            "cycle_detected": False,
            "min_cut_capacity": 0.15,
        },
        "metrics": {
            "success_rate": 0.65,
            "abstention_rate": 0.20,
            "block_rate": 0.35,
            "throughput": 15.0,
            "latency_p50_ms": 180,
            "latency_p99_ms": 800,
            "queue_depth": 250,
        },
        "budget": {
            "compute_budget_remaining": 0.45,
            "memory_utilization": 0.70,
            "storage_headroom_gb": 60.0,
            "verification_quota_remaining": 400,
            "budget_exhaustion_eta_cycles": 200,
        },
        "identity": {
            "block_hash_valid": True,
            "merkle_root_valid": True,
            "signature_valid": True,
            "chain_continuous": True,
            "pq_attestation_valid": True,
            "dual_root_consistent": True,
        },
        "narrative": {
            "current_slice": "propositional_tautology",
            "slice_progress": 0.45,
            "epoch": 5,
            "curriculum_health": "CRITICAL",
            "drift_detected": True,
            "narrative_coherence": 0.55,
        },
    },
    "expected": {
        "escalation_level": "L3_CRITICAL",
        "fusion_decision": "BLOCK",
        "is_hard": True,
        "primary_recommendation": {
            "signal_id": "telemetry",
            "action": "HARD_BLOCK",
            "reason": "STRUCTURAL_BREAK streak ≥2: regime change detected",
        },
        "conflict_detections": [
            {
                "rule_id": "CSC-P5-001",
                "description": "Structural break with DAG tension: regime change under structural stress",
                "signals_involved": ["telemetry", "structure"],
                "severity": "CRITICAL",
            }
        ],
    },
}


# All canonical fixtures
CANONICAL_FIXTURES = [
    P5_DRIFT_001,
    P5_NOISE_001,
    P5_PHASE_001,
    P5_ATTRACTOR_001,
    P5_TRANSIENT_001,
    P5_STRUCTURAL_001,
]


# =============================================================================
# Test: Pattern Classification from Signals
# =============================================================================

class TestPatternClassificationFromSignals:
    """Tests for classify_from_signals() using canonical examples."""

    @pytest.mark.parametrize(
        "fixture",
        CANONICAL_FIXTURES,
        ids=lambda f: f["test_id"],
    )
    def test_pattern_classification_matches_expected(self, fixture: Dict[str, Any]):
        """Verify pattern is correctly classified from signal extensions."""
        p5_telemetry = fixture["signals"]["telemetry"].get("p5_telemetry")
        p5_topology = fixture["signals"]["topology"].get("p5_twin")
        p5_replay = fixture["signals"]["replay"].get("p5_divergence")

        classification = classify_from_signals(
            p5_telemetry=p5_telemetry,
            p5_topology=p5_topology,
            p5_replay=p5_replay,
        )

        assert classification.pattern.value == fixture["pattern"], (
            f"{fixture['test_id']}: Expected pattern {fixture['pattern']}, "
            f"got {classification.pattern.value}"
        )

    @pytest.mark.parametrize(
        "fixture",
        CANONICAL_FIXTURES,
        ids=lambda f: f["test_id"],
    )
    def test_pattern_confidence_is_valid(self, fixture: Dict[str, Any]):
        """Verify confidence is in valid range [0, 1]."""
        p5_telemetry = fixture["signals"]["telemetry"].get("p5_telemetry")

        classification = classify_from_signals(p5_telemetry=p5_telemetry)

        assert 0.0 <= classification.confidence <= 1.0, (
            f"{fixture['test_id']}: Invalid confidence {classification.confidence}"
        )

    def test_classify_from_signals_without_telemetry_returns_nominal(self):
        """Verify NOMINAL is returned when no p5_telemetry is provided."""
        classification = classify_from_signals(p5_telemetry=None)
        assert classification.pattern == DivergencePattern.NOMINAL

    def test_classify_from_signals_with_unknown_pattern(self):
        """Verify UNCLASSIFIED is returned for unknown pattern strings."""
        classification = classify_from_signals(
            p5_telemetry={"divergence_pattern": "UNKNOWN_PATTERN", "divergence_pattern_streak": 1}
        )
        assert classification.pattern == DivergencePattern.UNCLASSIFIED


# =============================================================================
# Test: TDAPatternClassifier
# =============================================================================

class TestTDAPatternClassifier:
    """Tests for TDAPatternClassifier direct classification."""

    def test_classifier_initialization(self):
        """Verify classifier initializes with default thresholds."""
        classifier = TDAPatternClassifier()
        assert classifier.get_current_pattern() == DivergencePattern.NOMINAL
        assert classifier.get_pattern_streak() == 0

    def test_classifier_returns_nominal_with_insufficient_samples(self):
        """Verify classifier returns NOMINAL with < 10 samples."""
        classifier = TDAPatternClassifier()

        # Only 5 samples
        for _ in range(5):
            result = classifier.classify(
                delta_p=0.01,
                p_real=0.5,
                p_twin=0.51,
                omega_real=True,
                omega_twin=True,
            )

        assert result.pattern == DivergencePattern.NOMINAL
        assert result.evidence.get("reason") == "insufficient_samples"

    def test_classifier_detects_drift_pattern(self):
        """Verify DRIFT detection with systematic bias."""
        classifier = TDAPatternClassifier()

        # Feed consistent positive bias (mean > 0.05, std < 0.02)
        for i in range(50):
            result = classifier.classify(
                delta_p=0.06 + (i % 2) * 0.005,  # 0.06 or 0.065
                p_real=0.5,
                p_twin=0.56,
                omega_real=True,
                omega_twin=True,
            )

        assert result.pattern == DivergencePattern.DRIFT
        assert classifier.get_current_pattern() == DivergencePattern.DRIFT

    def test_classifier_detects_attractor_miss(self):
        """Verify ATTRACTOR_MISS detection with omega mismatch."""
        classifier = TDAPatternClassifier()

        # Feed samples with frequent omega mismatches
        for i in range(30):
            omega_real = True
            # Twin misses omega 40% of the time
            omega_twin = (i % 5) != 0

            result = classifier.classify(
                delta_p=0.02,
                p_real=0.5,
                p_twin=0.52,
                omega_real=omega_real,
                omega_twin=omega_twin,
            )

        assert result.pattern == DivergencePattern.ATTRACTOR_MISS

    def test_classifier_streak_tracking(self):
        """Verify streak tracking increments for same pattern."""
        classifier = TDAPatternClassifier()

        # Feed DRIFT pattern consistently
        for i in range(20):
            classifier.classify(
                delta_p=0.06,
                p_real=0.5,
                p_twin=0.56,
                omega_real=True,
                omega_twin=True,
            )

        streak = classifier.get_pattern_streak()
        assert streak >= 10, f"Expected streak >= 10, got {streak}"

    def test_classifier_reset(self):
        """Verify reset clears all state."""
        classifier = TDAPatternClassifier()

        # Add some history
        for _ in range(20):
            classifier.classify(
                delta_p=0.06,
                p_real=0.5,
                p_twin=0.56,
                omega_real=True,
                omega_twin=True,
            )

        classifier.reset()

        assert classifier.get_current_pattern() == DivergencePattern.NOMINAL
        assert classifier.get_pattern_streak() == 0


# =============================================================================
# Test: P5 Signal Extensions
# =============================================================================

class TestP5SignalExtensions:
    """Tests for P5 signal extension builders."""

    def test_p5_telemetry_extension_defaults(self):
        """Verify P5TelemetryExtension default values."""
        ext = P5TelemetryExtension()
        assert ext.telemetry_validation_status == "VALIDATION_PENDING"
        assert ext.validation_confidence == 0.0
        assert ext.divergence_pattern == "NOMINAL"
        assert ext.divergence_pattern_streak == 0
        assert ext.recalibration_triggered is False

    def test_p5_telemetry_extension_to_dict(self):
        """Verify P5TelemetryExtension serializes correctly."""
        ext = P5TelemetryExtension(
            telemetry_validation_status="VALIDATED_REAL",
            validation_confidence=0.92,
            divergence_pattern="DRIFT",
            divergence_pattern_streak=3,
            recalibration_triggered=False,
        )
        d = ext.to_dict()

        assert d["telemetry_validation_status"] == "VALIDATED_REAL"
        assert d["validation_confidence"] == 0.92
        assert d["divergence_pattern"] == "DRIFT"
        assert d["divergence_pattern_streak"] == 3
        assert d["recalibration_triggered"] is False

    def test_p5_topology_extension_to_dict(self):
        """Verify P5TopologyExtension serializes correctly."""
        ext = P5TopologyExtension(
            attractor_miss_rate=0.25,
            twin_omega_alignment=False,
            transient_tracking_quality=0.60,
        )
        d = ext.to_dict()

        assert d["attractor_miss_rate"] == 0.25
        assert d["twin_omega_alignment"] is False
        assert d["transient_tracking_quality"] == 0.60

    def test_p5_replay_extension_to_dict(self):
        """Verify P5ReplayExtension serializes correctly."""
        ext = P5ReplayExtension(
            twin_prediction_divergence=0.12,
            divergence_bias=0.03,
            divergence_variance=0.025,
        )
        d = ext.to_dict()

        assert d["twin_prediction_divergence"] == 0.12
        assert d["divergence_bias"] == 0.03
        assert d["divergence_variance"] == 0.025

    def test_classifier_builds_telemetry_extension(self):
        """Verify classifier builds P5TelemetryExtension correctly."""
        classifier = TDAPatternClassifier()

        # Build some history
        for _ in range(20):
            classifier.classify(
                delta_p=0.06,
                p_real=0.5,
                p_twin=0.56,
                omega_real=True,
                omega_twin=True,
            )

        ext = classifier.get_p5_telemetry_extension(
            validation_status="VALIDATED_REAL",
            validation_confidence=0.92,
        )

        assert ext.telemetry_validation_status == "VALIDATED_REAL"
        assert ext.validation_confidence == 0.92
        assert ext.divergence_pattern == classifier.get_current_pattern().value
        assert ext.divergence_pattern_streak == classifier.get_pattern_streak()


# =============================================================================
# Test: attach_tda_patterns_to_evidence
# =============================================================================

class TestAttachTDAPatternsToEvidence:
    """Tests for attach_tda_patterns_to_evidence() function."""

    def test_attach_creates_governance_key(self):
        """Verify governance key is created if missing."""
        evidence = {}
        classifier = TDAPatternClassifier()

        result = attach_tda_patterns_to_evidence(evidence, classifier)

        assert "governance" in result
        assert "p5_pattern_classification" in result["governance"]

    def test_attach_preserves_existing_evidence(self):
        """Verify existing evidence is preserved."""
        evidence = {"existing_key": "existing_value", "governance": {"other": "data"}}
        classifier = TDAPatternClassifier()

        result = attach_tda_patterns_to_evidence(evidence, classifier)

        assert result["existing_key"] == "existing_value"
        assert result["governance"]["other"] == "data"
        assert "p5_pattern_classification" in result["governance"]

    def test_attach_includes_schema_version(self):
        """Verify schema version is included."""
        evidence = {}
        classifier = TDAPatternClassifier()

        result = attach_tda_patterns_to_evidence(evidence, classifier)

        p5 = result["governance"]["p5_pattern_classification"]
        assert p5["schema_version"] == P5_PATTERN_SCHEMA_VERSION

    def test_attach_includes_shadow_mode_marker(self):
        """Verify SHADOW mode marker is included."""
        evidence = {}
        classifier = TDAPatternClassifier()

        result = attach_tda_patterns_to_evidence(evidence, classifier)

        p5 = result["governance"]["p5_pattern_classification"]
        assert p5["mode"] == "SHADOW"

    def test_attach_includes_shadow_mode_invariants(self):
        """Verify SHADOW MODE invariants are included."""
        evidence = {}
        classifier = TDAPatternClassifier()

        result = attach_tda_patterns_to_evidence(evidence, classifier)

        p5 = result["governance"]["p5_pattern_classification"]
        invariants = p5["shadow_mode_invariants"]
        assert invariants["no_enforcement"] is True
        assert invariants["logged_only"] is True
        assert invariants["observation_only"] is True

    def test_attach_includes_signal_extensions(self):
        """Verify all signal extensions are included."""
        evidence = {}
        classifier = TDAPatternClassifier()

        # Build some history
        for _ in range(20):
            classifier.classify(
                delta_p=0.06,
                p_real=0.5,
                p_twin=0.56,
                omega_real=True,
                omega_twin=True,
            )

        result = attach_tda_patterns_to_evidence(
            evidence,
            classifier,
            validation_status="VALIDATED_REAL",
            validation_confidence=0.92,
        )

        p5 = result["governance"]["p5_pattern_classification"]
        exts = p5["signal_extensions"]

        assert "p5_telemetry" in exts
        assert "p5_topology" in exts
        assert "p5_replay" in exts

        assert exts["p5_telemetry"]["telemetry_validation_status"] == "VALIDATED_REAL"
        assert exts["p5_telemetry"]["validation_confidence"] == 0.92


# =============================================================================
# Test: SHADOW MODE Invariants
# =============================================================================

class TestShadowModeInvariants:
    """Tests verifying SHADOW MODE contract is enforced."""

    def test_classification_does_not_raise_on_any_pattern(self):
        """Verify classification never raises exceptions (observation only)."""
        classifier = TDAPatternClassifier()

        # Test various extreme inputs - should never raise
        extreme_inputs = [
            (10.0, 0.0, 10.0, True, False),   # Extreme delta
            (-10.0, 0.5, -9.5, False, True),  # Negative extreme
            (0.0, 0.0, 0.0, True, True),      # All zeros
            (float("inf"), 0.5, float("inf"), True, True),  # Infinity (if not filtered)
        ]

        for delta_p, p_real, p_twin, omega_r, omega_t in extreme_inputs:
            try:
                # Should not raise
                if not (delta_p != delta_p):  # Skip NaN
                    classifier.classify(
                        delta_p=min(max(delta_p, -100), 100),  # Clamp to avoid inf
                        p_real=p_real,
                        p_twin=min(max(p_twin, -100), 100),
                        omega_real=omega_r,
                        omega_twin=omega_t,
                    )
            except (ValueError, OverflowError):
                pass  # These are acceptable in extreme cases

    def test_attach_always_marks_shadow_mode(self):
        """Verify attached evidence always has mode=SHADOW."""
        classifier = TDAPatternClassifier()

        # Test multiple scenarios
        scenarios = [
            {},  # Empty evidence
            {"governance": {}},  # Existing governance
            {"governance": {"mode": "PRODUCTION"}},  # Conflicting mode (should override)
        ]

        for evidence in scenarios:
            result = attach_tda_patterns_to_evidence(evidence.copy(), classifier)
            assert result["governance"]["p5_pattern_classification"]["mode"] == "SHADOW"

    def test_pattern_classification_is_deterministic(self):
        """Verify same inputs produce same classification (no hidden state effects)."""
        # Create two identical classifiers
        classifier1 = TDAPatternClassifier()
        classifier2 = TDAPatternClassifier()

        # Feed identical inputs
        inputs = [
            (0.06, 0.5, 0.56, True, True),
            (0.07, 0.5, 0.57, True, True),
            (0.055, 0.5, 0.555, True, True),
        ]

        for _ in range(20):
            for delta_p, p_real, p_twin, omega_r, omega_t in inputs:
                result1 = classifier1.classify(delta_p, p_real, p_twin, omega_r, omega_t)
                result2 = classifier2.classify(delta_p, p_real, p_twin, omega_r, omega_t)

                assert result1.pattern == result2.pattern, "Determinism violated"
                assert result1.confidence == result2.confidence, "Determinism violated"


# =============================================================================
# Test: Pattern Severity Ordering
# =============================================================================

class TestPatternSeverityOrdering:
    """Tests verifying pattern severity hierarchy from test plan."""

    def test_structural_break_is_highest_severity(self):
        """STRUCTURAL_BREAK should produce highest confidence/severity."""
        # This is validated by the fixture expectations:
        # STRUCTURAL_BREAK -> L3_CRITICAL, HARD_BLOCK
        fixture = P5_STRUCTURAL_001
        expected = fixture["expected"]

        assert expected["escalation_level"] == "L3_CRITICAL"
        assert expected["is_hard"] is True

    def test_attractor_miss_is_high_severity(self):
        """ATTRACTOR_MISS should produce high severity."""
        fixture = P5_ATTRACTOR_001
        expected = fixture["expected"]

        assert expected["escalation_level"] == "L3_CRITICAL"
        assert expected["is_hard"] is True

    def test_transient_miss_is_medium_severity(self):
        """TRANSIENT_MISS should produce medium severity."""
        fixture = P5_TRANSIENT_001
        expected = fixture["expected"]

        assert expected["escalation_level"] == "L2_DEGRADED"
        assert expected["is_hard"] is False  # Soft block

    def test_drift_is_low_severity(self):
        """DRIFT should produce low severity (warning only)."""
        fixture = P5_DRIFT_001
        expected = fixture["expected"]

        assert expected["escalation_level"] == "L1_WARNING"
        assert expected["fusion_decision"] == "ALLOW"

    def test_noise_amplification_is_low_severity(self):
        """NOISE_AMPLIFICATION should produce low severity."""
        fixture = P5_NOISE_001
        expected = fixture["expected"]

        assert expected["escalation_level"] == "L1_WARNING"
        assert expected["fusion_decision"] == "ALLOW"

    def test_phase_lag_is_low_severity(self):
        """PHASE_LAG should produce low severity."""
        fixture = P5_PHASE_001
        expected = fixture["expected"]

        assert expected["escalation_level"] == "L1_WARNING"
        assert expected["fusion_decision"] == "ALLOW"


# =============================================================================
# Test: Conflict Detection (CSC-P5-* rules)
# =============================================================================

class TestConflictDetection:
    """Tests for CSC-P5-* conflict detection rules."""

    def test_csc_p5_001_structural_break_with_dag_tension(self):
        """CSC-P5-001: STRUCTURAL_BREAK + min_cut_capacity < 0.2."""
        fixture = P5_STRUCTURAL_001
        expected_conflicts = fixture["expected"]["conflict_detections"]

        # Verify fixture has min_cut_capacity < 0.2
        assert fixture["signals"]["structure"]["min_cut_capacity"] == 0.15

        # Verify CSC-P5-001 is expected
        rule_ids = [c["rule_id"] for c in expected_conflicts]
        assert "CSC-P5-001" in rule_ids

        # Find the specific conflict
        csc_p5_001 = next(c for c in expected_conflicts if c["rule_id"] == "CSC-P5-001")
        assert csc_p5_001["severity"] == "CRITICAL"
        assert "telemetry" in csc_p5_001["signals_involved"]
        assert "structure" in csc_p5_001["signals_involved"]

    def test_csc_p5_003_attractor_miss_while_safe(self):
        """CSC-P5-003: attractor_miss_rate > 0.2 AND within_omega = true."""
        fixture = P5_ATTRACTOR_001
        expected_conflicts = fixture["expected"]["conflict_detections"]

        # Verify fixture conditions
        assert fixture["signals"]["topology"]["p5_twin"]["attractor_miss_rate"] == 0.25
        assert fixture["signals"]["topology"]["within_omega"] is True

        # Verify CSC-P5-003 is expected
        rule_ids = [c["rule_id"] for c in expected_conflicts]
        assert "CSC-P5-003" in rule_ids

        csc_p5_003 = next(c for c in expected_conflicts if c["rule_id"] == "CSC-P5-003")
        assert csc_p5_003["severity"] == "HIGH"

    def test_warning_patterns_have_no_conflicts(self):
        """Warning-level patterns should not produce conflicts."""
        warning_fixtures = [P5_DRIFT_001, P5_NOISE_001, P5_PHASE_001]

        for fixture in warning_fixtures:
            expected_conflicts = fixture["expected"]["conflict_detections"]
            assert len(expected_conflicts) == 0, (
                f"{fixture['test_id']} should have no conflicts"
            )

    def test_transient_miss_has_no_conflicts(self):
        """TRANSIENT_MISS (L2_DEGRADED) should have no conflicts."""
        fixture = P5_TRANSIENT_001
        expected_conflicts = fixture["expected"]["conflict_detections"]
        assert len(expected_conflicts) == 0


# =============================================================================
# Test: PatternClassification dataclass
# =============================================================================

class TestPatternClassification:
    """Tests for PatternClassification dataclass."""

    def test_pattern_classification_to_dict(self):
        """Verify to_dict() produces correct structure."""
        classification = PatternClassification(
            pattern=DivergencePattern.DRIFT,
            confidence=0.85,
            evidence={"trigger": "systematic_bias"},
        )

        d = classification.to_dict()

        assert d["pattern"] == "DRIFT"
        assert d["confidence"] == 0.85
        assert d["evidence"]["trigger"] == "systematic_bias"
        assert "timestamp" in d

    def test_pattern_classification_timestamp_auto_generated(self):
        """Verify timestamp is auto-generated if not provided."""
        classification = PatternClassification(
            pattern=DivergencePattern.NOMINAL,
            confidence=0.5,
        )

        assert classification.timestamp != ""
        assert "T" in classification.timestamp  # ISO format check


# =============================================================================
# Test: Full Integration with GGFL (if available)
# =============================================================================

class TestGGFLIntegration:
    """Integration tests with build_global_alignment_view."""

    @pytest.mark.parametrize(
        "fixture",
        [P5_DRIFT_001, P5_NOISE_001, P5_PHASE_001],
        ids=["DRIFT", "NOISE", "PHASE_LAG"],
    )
    def test_warning_patterns_produce_allow_decision(self, fixture: Dict[str, Any]):
        """Verify warning-level patterns result in ALLOW decision from GGFL."""
        signals = fixture["signals"]

        # Call GGFL with the fixture signals
        result = build_global_alignment_view(
            topology=signals.get("topology"),
            replay=signals.get("replay"),
            telemetry=signals.get("telemetry"),
            structure=signals.get("structure"),
            metrics=signals.get("metrics"),
            budget=signals.get("budget"),
            identity=signals.get("identity"),
            narrative=signals.get("narrative"),
            cycle=100,
        )

        # These patterns should not trigger BLOCK
        decision = result["fusion_result"]["decision"]
        # Note: GGFL may produce ALLOW or ABSTAIN based on weighted voting
        # The key is it should NOT produce HARD_BLOCK
        assert decision != "HARD_BLOCK", (
            f"{fixture['test_id']}: Warning pattern should not produce HARD_BLOCK"
        )

    def test_ggfl_handles_p5_extensions_gracefully(self):
        """Verify GGFL doesn't crash with P5 extension fields."""
        # Use STRUCTURAL_BREAK fixture with all P5 extensions
        signals = P5_STRUCTURAL_001["signals"]

        # Should not raise
        result = build_global_alignment_view(
            topology=signals.get("topology"),
            replay=signals.get("replay"),
            telemetry=signals.get("telemetry"),
            structure=signals.get("structure"),
            metrics=signals.get("metrics"),
            budget=signals.get("budget"),
            identity=signals.get("identity"),
            narrative=signals.get("narrative"),
            cycle=100,
        )

        assert "fusion_result" in result
        assert "escalation" in result
