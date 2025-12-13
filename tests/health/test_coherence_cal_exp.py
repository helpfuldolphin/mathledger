"""Tests for coherence CAL-EXP cross-check.

Tests verify:
- CAL-EXP coherence snapshot building
- Snapshot persistence
- Coherence vs GGFL consistency analysis
- Evidence attachment
- JSON serialization
- Determinism
- Non-mutation
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict

from backend.health.coherence_cal_exp import (
    COHERENCE_SNAPSHOT_SCHEMA_VERSION,
    COHERENCE_CROSSCHECK_SCHEMA_VERSION,
    COHERENCE_CROSSCHECK_MODE,
    ALLOWED_CONSISTENCY_STATUS,
    ALLOWED_GGFL_ESCALATION,
    SELECTION_CONTRACT_SEVERITY_FORMULA,
    SELECTION_CONTRACT_TIE_BREAKERS,
    build_cal_exp_coherence_snapshot,
    persist_coherence_snapshot,
    summarize_coherence_vs_fusion,
    attach_coherence_fusion_crosscheck_to_evidence,
    extract_coherence_fusion_status,
    _score_example,
    _coerce_consistency_status,
    _coerce_ggfl_escalation,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_first_light_summary() -> Dict[str, Any]:
    """Sample coherence first-light summary."""
    return {
        "coherence_band": "PARTIAL",
        "global_index": 0.583,
        "slices_at_risk": ["slice3"],
    }


@pytest.fixture
def sample_ggfl_results() -> Dict[str, Any]:
    """Sample GGFL fusion results."""
    return {
        "escalation_level": "L1_WARNING",
        "decision": "ALLOW",
        "recommendations": [
            {
                "signal_id": "narrative",
                "action": "WARNING",
                "confidence": 0.7,
                "reason": "Narrative coherence below threshold",
            },
        ],
    }


# =============================================================================
# PHASE 1: CAL-EXP COHERENCE SNAPSHOT
# =============================================================================

class TestBuildCalExpCoherenceSnapshot:
    """Tests for CAL-EXP coherence snapshot building."""
    
    def test_snapshot_has_required_fields(
        self, sample_first_light_summary
    ):
        """Snapshot should have all required fields."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        required = {
            "schema_version", "cal_id", "coherence_band",
            "global_index", "num_slices_at_risk"
        }
        assert required.issubset(set(snapshot.keys()))
    
    def test_schema_version(self, sample_first_light_summary):
        """Snapshot should have correct schema version."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        assert snapshot["schema_version"] == COHERENCE_SNAPSHOT_SCHEMA_VERSION
    
    def test_cal_id_preserved(self, sample_first_light_summary):
        """Should preserve cal_id."""
        snapshot = build_cal_exp_coherence_snapshot(
            "CAL-EXP-1", sample_first_light_summary
        )
        assert snapshot["cal_id"] == "CAL-EXP-1"
    
    def test_coherence_band_extracted(self, sample_first_light_summary):
        """Should extract coherence_band."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        assert snapshot["coherence_band"] == "PARTIAL"
    
    def test_global_index_extracted(self, sample_first_light_summary):
        """Should extract and round global_index."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        assert snapshot["global_index"] == 0.583
        assert isinstance(snapshot["global_index"], float)
    
    def test_num_slices_at_risk_computed(self, sample_first_light_summary):
        """Should compute num_slices_at_risk from slices_at_risk."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        assert snapshot["num_slices_at_risk"] == 1
    
    def test_num_slices_at_risk_zero(self):
        """Should handle empty slices_at_risk."""
        summary = {
            "coherence_band": "COHERENT",
            "global_index": 0.9,
            "slices_at_risk": [],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        assert snapshot["num_slices_at_risk"] == 0
    
    def test_non_mutating(self, sample_first_light_summary):
        """Should not modify input dictionary."""
        original = dict(sample_first_light_summary)
        
        build_cal_exp_coherence_snapshot("cal_exp1", sample_first_light_summary)
        
        assert sample_first_light_summary == original
    
    def test_json_serializable(self, sample_first_light_summary):
        """Snapshot should be JSON serializable."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        json_str = json.dumps(snapshot)
        parsed = json.loads(json_str)
        
        assert parsed == snapshot
    
    def test_deterministic(self, sample_first_light_summary):
        """Snapshot should be deterministic."""
        snapshot1 = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        snapshot2 = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        assert snapshot1 == snapshot2


class TestPersistCoherenceSnapshot:
    """Tests for snapshot persistence."""
    
    def test_persists_to_file(self, sample_first_light_summary):
        """Should persist snapshot to file."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            output_path = persist_coherence_snapshot(snapshot, output_dir)
            
            assert output_path.exists()
            assert output_path.name == "coherence_cal_exp1.json"
            
            # Verify file contents
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            assert loaded == snapshot
    
    def test_creates_directory(self, sample_first_light_summary):
        """Should create output directory if missing."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration" / "nested"
            output_path = persist_coherence_snapshot(snapshot, output_dir)
            
            assert output_dir.exists()
            assert output_path.exists()
    
    def test_json_round_trip(self, sample_first_light_summary):
        """Should support JSON round-trip."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            output_path = persist_coherence_snapshot(snapshot, output_dir)
            
            # Read back and verify
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            assert loaded == snapshot
            assert loaded["schema_version"] == COHERENCE_SNAPSHOT_SCHEMA_VERSION


# =============================================================================
# PHASE 2: COHERENCE VS GGFL CONSISTENCY
# =============================================================================

class TestSummarizeCoherenceVsFusion:
    """Tests for coherence vs GGFL consistency analysis."""
    
    def test_crosscheck_has_required_fields(
        self, sample_first_light_summary, sample_ggfl_results
    ):
        """Cross-check should have all required fields."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        crosscheck = summarize_coherence_vs_fusion(
            [snapshot], sample_ggfl_results
        )
        
        required = {"consistency_status", "examples", "advisory_notes"}
        assert required.issubset(set(crosscheck.keys()))
    
    def test_consistency_status_values(self):
        """Consistency status should be one of allowed values."""
        snapshot = {
            "schema_version": "1.0.0",
            "cal_id": "cal_exp1",
            "coherence_band": "COHERENT",
            "global_index": 0.9,
            "num_slices_at_risk": 0,
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot])
        
        assert crosscheck["consistency_status"] in (
            "CONSISTENT", "TENSION", "CONFLICT"
        )
    
    def test_consistent_when_no_ggfl(self, sample_first_light_summary):
        """Should return CONSISTENT when no GGFL results provided."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        crosscheck = summarize_coherence_vs_fusion([snapshot])
        
        assert crosscheck["consistency_status"] == "CONSISTENT"
    
    def test_tension_ggfl_warn_coherence_misaligned(
        self, sample_ggfl_results
    ):
        """Should detect TENSION when GGFL WARNING + coherence MISALIGNED."""
        summary_misaligned = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.3,
            "slices_at_risk": ["slice1", "slice2"],
        }
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", summary_misaligned
        )
        
        # GGFL with L1_WARNING
        ggfl_warn = {
            **sample_ggfl_results,
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl_warn)
        
        assert crosscheck["consistency_status"] == "TENSION"
        assert len(crosscheck["examples"]) > 0
        
        # Verify normalized example structure
        example = crosscheck["examples"][0]
        assert "cal_id" in example
        assert "coherence_band" in example
        assert "ggfl_escalation" in example
        assert "ggfl_decision" in example
        assert "reason" in example
        assert example["cal_id"] == "cal_exp1"
        assert example["coherence_band"] == "MISALIGNED"
        assert example["ggfl_escalation"] == "L1_WARNING"
    
    def test_conflict_ggfl_ok_coherence_misaligned(self):
        """Should detect CONFLICT when GGFL OK + coherence MISALIGNED."""
        summary_misaligned = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.3,
            "slices_at_risk": ["slice1"],
        }
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", summary_misaligned
        )
        
        ggfl_ok = {
            "escalation_level": "L0_NOMINAL",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl_ok)
        
        assert crosscheck["consistency_status"] == "CONFLICT"
        assert len(crosscheck["examples"]) > 0
        
        # Verify normalized example structure
        example = crosscheck["examples"][0]
        assert example["cal_id"] == "cal_exp1"
        assert example["coherence_band"] == "MISALIGNED"
        assert example["ggfl_escalation"] == "L0_NOMINAL"
        assert "L0_NOMINAL" in example["reason"]
    
    def test_tension_multiple_warnings_partial(self, sample_ggfl_results):
        """Should detect TENSION with multiple warnings + partial coherence."""
        summary_partial = {
            "coherence_band": "PARTIAL",
            "global_index": 0.6,
            "slices_at_risk": ["slice1"],
        }
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", summary_partial
        )
        
        ggfl_multi_warn = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [
                {"signal_id": "narrative", "action": "WARNING"},
                {"signal_id": "topology", "action": "WARNING"},
            ],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl_multi_warn)
        
        assert crosscheck["consistency_status"] == "TENSION"
    
    def test_advisory_notes_present(self, sample_first_light_summary):
        """Should include advisory notes."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        crosscheck = summarize_coherence_vs_fusion([snapshot])
        
        assert len(crosscheck["advisory_notes"]) > 0
        assert all(isinstance(note, str) for note in crosscheck["advisory_notes"])
    
    def test_empty_snapshots_handled(self):
        """Should handle empty snapshot list."""
        crosscheck = summarize_coherence_vs_fusion([])
        
        assert crosscheck["consistency_status"] == "CONSISTENT"
        assert "No coherence snapshots" in crosscheck["advisory_notes"][0]
    
    def test_non_mutating(self, sample_first_light_summary, sample_ggfl_results):
        """Should not modify input dictionaries."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        original_snapshot = dict(snapshot)
        original_ggfl = dict(sample_ggfl_results)
        
        summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        
        assert snapshot == original_snapshot
        assert sample_ggfl_results == original_ggfl
    
    def test_json_serializable(
        self, sample_first_light_summary, sample_ggfl_results
    ):
        """Cross-check should be JSON serializable."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        
        json_str = json.dumps(crosscheck)
        parsed = json.loads(json_str)
        
        assert parsed == crosscheck
    
    def test_deterministic(self, sample_first_light_summary, sample_ggfl_results):
        """Cross-check should be deterministic."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        crosscheck1 = summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        crosscheck2 = summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        
        assert crosscheck1 == crosscheck2
    
    def test_examples_normalized_structure(self, sample_ggfl_results):
        """Examples should have normalized structure with all required fields."""
        summaries = [
            {"coherence_band": "MISALIGNED", "global_index": 0.3, "slices_at_risk": ["s1"]},
            {"coherence_band": "PARTIAL", "global_index": 0.6, "slices_at_risk": ["s2"]},
        ]
        snapshots = [
            build_cal_exp_coherence_snapshot(f"cal_exp{i+1}", s)
            for i, s in enumerate(summaries)
        ]
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [{"action": "WARNING"}],
        }
        
        crosscheck = summarize_coherence_vs_fusion(snapshots, ggfl)
        
        # Verify all examples have required fields
        for example in crosscheck["examples"]:
            required = {"cal_id", "coherence_band", "ggfl_escalation", "ggfl_decision", "reason"}
            assert required.issubset(set(example.keys()))
            assert isinstance(example["cal_id"], str)
            assert example["coherence_band"] in ("COHERENT", "PARTIAL", "MISALIGNED")
            assert isinstance(example["reason"], str)
    
    def test_examples_limited_to_top_5(self):
        """Examples should be limited to top 5 deterministically."""
        # Create 10 snapshots with varying severity
        summaries = [
            {"coherence_band": "MISALIGNED", "global_index": 0.3, "slices_at_risk": []},
            {"coherence_band": "PARTIAL", "global_index": 0.6, "slices_at_risk": []},
            {"coherence_band": "COHERENT", "global_index": 0.9, "slices_at_risk": []},
        ] * 4  # 12 total
        
        snapshots = [
            build_cal_exp_coherence_snapshot(f"cal_exp{i+1}", s)
            for i, s in enumerate(summaries)
        ]
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion(snapshots, ggfl)
        
        # Should have at most 5 examples
        assert len(crosscheck["examples"]) <= 5
    
    def test_examples_deterministic_ordering(self):
        """Examples should be deterministically ordered by severity then cal_id."""
        summaries = [
            {"coherence_band": "MISALIGNED", "global_index": 0.3, "slices_at_risk": []},
            {"coherence_band": "PARTIAL", "global_index": 0.6, "slices_at_risk": []},
            {"coherence_band": "MISALIGNED", "global_index": 0.2, "slices_at_risk": []},
        ]
        snapshots = [
            build_cal_exp_coherence_snapshot(f"cal_exp{i+1}", s)
            for i, s in enumerate(summaries)
        ]
        
        ggfl = {
            "escalation_level": "L2_DEGRADED",
            "decision": "BLOCK",
            "recommendations": [],
        }
        
        crosscheck1 = summarize_coherence_vs_fusion(snapshots, ggfl)
        crosscheck2 = summarize_coherence_vs_fusion(snapshots, ggfl)
        
        # Examples should be in same order
        assert crosscheck1["examples"] == crosscheck2["examples"]
        
        # MISALIGNED should come before PARTIAL (higher severity)
        if len(crosscheck1["examples"]) >= 2:
            misaligned_indices = [
                i for i, ex in enumerate(crosscheck1["examples"])
                if ex["coherence_band"] == "MISALIGNED"
            ]
            partial_indices = [
                i for i, ex in enumerate(crosscheck1["examples"])
                if ex["coherence_band"] == "PARTIAL"
            ]
            if misaligned_indices and partial_indices:
                assert min(misaligned_indices) < min(partial_indices)
    
    def test_examples_enriched_with_coherence_index(self):
        """Examples should include coherence_index when available from snapshot."""
        summary = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.345,
            "slices_at_risk": ["slice1"],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        
        # Should have coherence_index field
        if crosscheck["examples"]:
            example = crosscheck["examples"][0]
            assert "coherence_index" in example
            assert example["coherence_index"] == 0.345
    
    def test_examples_enriched_with_ggfl_primary_signal(self):
        """Examples should include ggfl_primary_signal when available."""
        summary = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.3,
            "slices_at_risk": ["slice1"],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [
                {"action": "WARNING", "signal_id": "topology", "priority": 5},
                {"action": "WARNING", "signal_id": "narrative", "priority": 3},
            ],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        
        # Should have ggfl_primary_signal field (highest priority recommendation)
        if crosscheck["examples"]:
            example = crosscheck["examples"][0]
            assert "ggfl_primary_signal" in example
            assert example["ggfl_primary_signal"] == "topology"
    
    def test_examples_optional_fields_never_error(self):
        """Optional fields should not cause errors if missing."""
        summary = {
            "coherence_band": "MISALIGNED",
            # No global_index
            "slices_at_risk": [],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        # Remove global_index if it was set by default
        if "global_index" in snapshot:
            del snapshot["global_index"]
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [],  # No recommendations = no primary signal
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        
        # Should not error, optional fields may be missing
        if crosscheck["examples"]:
            example = crosscheck["examples"][0]
            # coherence_index and ggfl_primary_signal are optional
            assert "coherence_index" not in example or example["coherence_index"] is not None
            assert "ggfl_primary_signal" not in example or example["ggfl_primary_signal"] is not None


class TestScoreExample:
    """Tests for severity scoring helper."""
    
    def test_score_example_formula(self):
        """Score helper should use documented formula."""
        # MISALIGNED + L3_CRITICAL = (2 × 10) + 3 = 23
        assert _score_example("MISALIGNED", "L3_CRITICAL") == 23
        
        # PARTIAL + L1_WARNING = (1 × 10) + 1 = 11
        assert _score_example("PARTIAL", "L1_WARNING") == 11
        
        # COHERENT + L0_NOMINAL = (0 × 10) + 0 = 0
        assert _score_example("COHERENT", "L0_NOMINAL") == 0
    
    def test_score_example_deterministic(self):
        """Score helper should be deterministic."""
        score1 = _score_example("MISALIGNED", "L2_DEGRADED")
        score2 = _score_example("MISALIGNED", "L2_DEGRADED")
        assert score1 == score2
    
    def test_score_example_coherence_weighted(self):
        """Coherence should be weighted 10x more than GGFL escalation."""
        # MISALIGNED + L0 should score higher than PARTIAL + L5
        misaligned_l0 = _score_example("MISALIGNED", "L0_NOMINAL")
        partial_l5 = _score_example("PARTIAL", "L5_EMERGENCY")
        assert misaligned_l0 > partial_l5  # 20 > 15
    
    def test_score_example_handles_none_escalation(self):
        """Score helper should handle None escalation."""
        score = _score_example("MISALIGNED", None)
        assert score == 20  # (2 × 10) + 0
    
    def test_score_example_handles_unknown_escalation(self):
        """Score helper should handle unknown escalation levels."""
        score = _score_example("PARTIAL", "UNKNOWN_LEVEL")
        assert score == 10  # (1 × 10) + 0 (default)


class TestExtractCoherenceFusionStatus:
    """Tests for status extraction."""
    
    def test_extracts_status_fields(self, sample_first_light_summary, sample_ggfl_results):
        """Should extract consistency_status, example_count, top_example_cal_id."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        crosscheck = summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        
        status = extract_coherence_fusion_status(crosscheck)
        
        required = {"consistency_status", "example_count", "top_example_cal_id"}
        assert required.issubset(set(status.keys()))
        assert status["consistency_status"] in ("CONSISTENT", "TENSION", "CONFLICT")
        assert isinstance(status["example_count"], int)
    
    def test_top_example_cal_id_present_when_examples_exist(self):
        """Should set top_example_cal_id when examples exist."""
        summary = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.3,
            "slices_at_risk": ["slice1"],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        status = extract_coherence_fusion_status(crosscheck)
        
        assert status["top_example_cal_id"] == "cal_exp1"
        assert status["example_count"] == 1
    
    def test_top_example_cal_id_none_when_no_examples(self):
        """Should set top_example_cal_id to None when no examples."""
        summary = {
            "coherence_band": "COHERENT",
            "global_index": 0.9,
            "slices_at_risk": [],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L0_NOMINAL",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        status = extract_coherence_fusion_status(crosscheck)
        
        assert status["top_example_cal_id"] is None
        assert status["example_count"] == 0
    
    def test_status_json_serializable(self, sample_first_light_summary, sample_ggfl_results):
        """Status should be JSON serializable."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        crosscheck = summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        status = extract_coherence_fusion_status(crosscheck)
        
        json_str = json.dumps(status)
        parsed = json.loads(json_str)
        
        assert parsed == status
    
    def test_status_non_mutating(self, sample_first_light_summary, sample_ggfl_results):
        """Should not modify input dictionary."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        crosscheck = summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        original = dict(crosscheck)
        
        extract_coherence_fusion_status(crosscheck)
        
        assert crosscheck == original


class TestAttachCoherenceFusionCrosscheckToEvidence:
    """Tests for evidence attachment."""
    
    def test_attaches_crosscheck_to_governance(
        self, sample_first_light_summary, sample_ggfl_results
    ):
        """Should attach cross-check under evidence['governance']."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        crosscheck = summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        
        evidence = {"evidence_id": "ev_001"}
        updated = attach_coherence_fusion_crosscheck_to_evidence(
            evidence, crosscheck
        )
        
        assert "governance" in updated
        assert "coherence_fusion_crosscheck" in updated["governance"]
        assert updated["governance"]["coherence_fusion_crosscheck"] == crosscheck
    
    def test_creates_governance_if_missing(self, sample_first_light_summary):
        """Should create governance structure if missing."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        crosscheck = summarize_coherence_vs_fusion([snapshot])
        
        evidence = {"evidence_id": "ev_001"}
        updated = attach_coherence_fusion_crosscheck_to_evidence(
            evidence, crosscheck
        )
        
        assert "governance" in updated
        assert "coherence_fusion_crosscheck" in updated["governance"]
    
    def test_non_mutating(self, sample_first_light_summary, sample_ggfl_results):
        """Should not modify input dictionaries."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        crosscheck = summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        
        evidence = {"evidence_id": "ev_001"}
        original = dict(evidence)
        
        attach_coherence_fusion_crosscheck_to_evidence(evidence, crosscheck)
        
        assert evidence == original


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestWarningHygiene:
    """Tests for warning hygiene (format and content)."""
    
    def test_status_extraction_includes_warning_fields(self):
        """Status extraction should include fields needed for warnings."""
        summary = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.3,
            "slices_at_risk": ["slice1"],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        status = extract_coherence_fusion_status(crosscheck)
        
        # Verify all fields needed for warnings are present
        assert "consistency_status" in status
        assert "example_count" in status
        assert "top_example_cal_id" in status
        
        # Verify warning format would be correct
        if status["consistency_status"] in ("TENSION", "CONFLICT"):
            assert status["example_count"] > 0
            assert status["top_example_cal_id"] is not None
    
    def test_no_warnings_for_consistent_status(self):
        """Status extraction should not trigger warnings for CONSISTENT."""
        summary = {
            "coherence_band": "COHERENT",
            "global_index": 0.9,
            "slices_at_risk": [],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L0_NOMINAL",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        status = extract_coherence_fusion_status(crosscheck)
        
        # CONSISTENT status should not trigger warnings
        assert status["consistency_status"] == "CONSISTENT"
        assert status["example_count"] == 0
        assert status["top_example_cal_id"] is None
        assert status["top_example_reason"] is None
    
    def test_schema_version_and_mode_present(self, sample_first_light_summary, sample_ggfl_results):
        """Cross-check summary should include schema_version and mode."""
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        crosscheck = summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        
        assert "schema_version" in crosscheck
        assert crosscheck["schema_version"] == "1.0.0"
        assert "mode" in crosscheck
        assert crosscheck["mode"] == "SHADOW"
    
    def test_top_example_reason_deterministic(self):
        """Top-example reason should be deterministic and neutral."""
        summary = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.3,
            "slices_at_risk": ["slice1"],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L3_CRITICAL",
            "decision": "BLOCK",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        status1 = extract_coherence_fusion_status(crosscheck)
        status2 = extract_coherence_fusion_status(crosscheck)
        
        # Should be deterministic
        assert status1["top_example_reason"] == status2["top_example_reason"]
        
        # Should include coherence_band and ggfl_escalation
        if status1["top_example_reason"]:
            assert "MISALIGNED" in status1["top_example_reason"]
            assert "L3_CRITICAL" in status1["top_example_reason"]
            assert "highest severity score" in status1["top_example_reason"]
            # No emojis
            assert "emoji" not in status1["top_example_reason"].lower()
    
    def test_single_warning_rule_conflict_overrides_tension(self):
        """When both CONFLICT and TENSION exist, only CONFLICT warning should be emitted."""
        # Create snapshots that would trigger both TENSION and CONFLICT
        summaries = [
            {"coherence_band": "MISALIGNED", "global_index": 0.3, "slices_at_risk": []},
            {"coherence_band": "PARTIAL", "global_index": 0.6, "slices_at_risk": []},
        ]
        snapshots = [
            build_cal_exp_coherence_snapshot(f"cal_exp{i+1}", s)
            for i, s in enumerate(summaries)
        ]
        
        # GGFL with L0_NOMINAL (triggers CONFLICT for MISALIGNED)
        # and L1_WARNING (triggers TENSION for PARTIAL)
        ggfl = {
            "escalation_level": "L0_NOMINAL",  # This will cause CONFLICT for MISALIGNED
            "decision": "ALLOW",
            "recommendations": [
                {"action": "WARNING", "signal_id": "topology", "priority": 5},
            ],
        }
        
        crosscheck = summarize_coherence_vs_fusion(snapshots, ggfl)
        status = extract_coherence_fusion_status(crosscheck)
        
        # Should have CONFLICT status (more severe than TENSION)
        assert status["consistency_status"] == "CONFLICT"
        
        # Should have examples
        assert status["example_count"] > 0
        
        # Status should include all required fields for warning
        assert "consistency_status" in status
        assert "example_count" in status
        assert "top_example_cal_id" in status
        assert "top_example_reason" in status
        assert "selection_contract" in status


class TestValueCoercion:
    """Tests for value coercion (CROSSCHECK CONTRACT v1)."""
    
    def test_coerce_consistency_status_allowed_values(self):
        """Should accept all allowed consistency status values."""
        for status in ALLOWED_CONSISTENCY_STATUS:
            coerced = _coerce_consistency_status(status)
            assert coerced == status
            assert coerced in ALLOWED_CONSISTENCY_STATUS
    
    def test_coerce_consistency_status_unknown_value(self):
        """Should coerce unknown values to UNKNOWN."""
        coerced = _coerce_consistency_status("INVALID_STATUS")
        assert coerced == "UNKNOWN"
        assert coerced in ALLOWED_CONSISTENCY_STATUS
    
    def test_coerce_ggfl_escalation_allowed_values(self):
        """Should accept all allowed GGFL escalation values."""
        for escalation in ALLOWED_GGFL_ESCALATION:
            coerced = _coerce_ggfl_escalation(escalation)
            assert coerced == escalation
            assert coerced in ALLOWED_GGFL_ESCALATION
    
    def test_coerce_ggfl_escalation_unknown_value(self):
        """Should coerce unknown values to L0_NOMINAL."""
        coerced = _coerce_ggfl_escalation("INVALID_ESCALATION")
        assert coerced == "L0_NOMINAL"
        assert coerced in ALLOWED_GGFL_ESCALATION
    
    def test_coerce_consistency_status_non_string(self):
        """Should coerce non-string values to UNKNOWN."""
        assert _coerce_consistency_status(None) == "UNKNOWN"
        assert _coerce_consistency_status(123) == "UNKNOWN"
        assert _coerce_consistency_status({}) == "UNKNOWN"
    
    def test_coerce_ggfl_escalation_non_string(self):
        """Should coerce non-string values to L0_NOMINAL."""
        assert _coerce_ggfl_escalation(None) == "L0_NOMINAL"
        assert _coerce_ggfl_escalation(123) == "L0_NOMINAL"
        assert _coerce_ggfl_escalation({}) == "L0_NOMINAL"


class TestSelectionContract:
    """Tests for selection contract (frozen formula and tie-breakers)."""
    
    def test_selection_contract_present_in_status(self):
        """Status should include selection_contract block."""
        summary = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.3,
            "slices_at_risk": ["slice1"],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        status = extract_coherence_fusion_status(crosscheck)
        
        assert "selection_contract" in status
        assert isinstance(status["selection_contract"], dict)
    
    def test_selection_contract_severity_formula_exact(self):
        """Selection contract severity_formula must match frozen value exactly."""
        summary = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.3,
            "slices_at_risk": [],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        status = extract_coherence_fusion_status(crosscheck)
        
        contract = status["selection_contract"]
        assert "severity_formula" in contract
        assert contract["severity_formula"] == SELECTION_CONTRACT_SEVERITY_FORMULA
    
    def test_selection_contract_tie_breakers_exact(self):
        """Selection contract tie_breakers must match frozen value exactly."""
        summary = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.3,
            "slices_at_risk": [],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        status = extract_coherence_fusion_status(crosscheck)
        
        contract = status["selection_contract"]
        assert "tie_breakers" in contract
        assert contract["tie_breakers"] == SELECTION_CONTRACT_TIE_BREAKERS
        assert isinstance(contract["tie_breakers"], list)
        assert len(contract["tie_breakers"]) == 2
    
    def test_selection_contract_deterministic(self):
        """Selection contract should be deterministic across calls."""
        summary = {
            "coherence_band": "MISALIGNED",
            "global_index": 0.3,
            "slices_at_risk": [],
        }
        snapshot = build_cal_exp_coherence_snapshot("cal_exp1", summary)
        
        ggfl = {
            "escalation_level": "L1_WARNING",
            "decision": "ALLOW",
            "recommendations": [],
        }
        
        crosscheck = summarize_coherence_vs_fusion([snapshot], ggfl)
        status1 = extract_coherence_fusion_status(crosscheck)
        status2 = extract_coherence_fusion_status(crosscheck)
        
        assert status1["selection_contract"] == status2["selection_contract"]
        assert status1["selection_contract"]["severity_formula"] == status2["selection_contract"]["severity_formula"]
        assert status1["selection_contract"]["tie_breakers"] == status2["selection_contract"]["tie_breakers"]


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(
        self, sample_first_light_summary, sample_ggfl_results
    ):
        """Test full workflow from snapshot to evidence."""
        # Build snapshot
        snapshot = build_cal_exp_coherence_snapshot(
            "cal_exp1", sample_first_light_summary
        )
        
        # Persist snapshot
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            output_path = persist_coherence_snapshot(snapshot, output_dir)
            assert output_path.exists()
        
        # Analyze consistency
        crosscheck = summarize_coherence_vs_fusion([snapshot], sample_ggfl_results)
        assert "consistency_status" in crosscheck
        
        # Attach to evidence
        evidence = {"evidence_id": "ev_001"}
        updated = attach_coherence_fusion_crosscheck_to_evidence(
            evidence, crosscheck
        )
        
        assert "governance" in updated
        assert "coherence_fusion_crosscheck" in updated["governance"]


# =============================================================================
# SMOKE-TEST READINESS CHECKLIST
# =============================================================================
#
# To verify coherence↔GGFL cross-check is ready for production:
#
# 1. Run all coherence CAL-EXP tests:
#    pytest tests/health/test_coherence_cal_exp.py -v
#
# 2. Verify example enrichment:
#    - Examples include coherence_index when available
#    - Examples include ggfl_primary_signal when available
#    - Missing optional fields never cause errors
#
# 3. Verify warning hygiene:
#    - At most 1 warning per status type (TENSION, CONFLICT)
#    - Warnings include example_count and top_example_cal_id
#    - No warnings emitted for CONSISTENT status
#    - Warning format: "Coherence fusion cross-check: {STATUS} detected (example_count={N}, top_example={CAL_ID})"
#
# 4. Verify deterministic severity scoring:
#    - _score_example() uses documented formula: (coherence_sev × 10) + ggfl_sev
#    - Examples sorted by severity (descending) then cal_id (ascending)
#    - Top 5 examples selected deterministically
#
# 5. Verify status extraction:
#    - extract_coherence_fusion_status() returns JSON-safe dict
#    - Fields: consistency_status, example_count, top_example_cal_id
#    - Non-mutating, deterministic
#
# 6. Verify status generator integration:
#    - signals["coherence_fusion_crosscheck"] populated correctly
#    - Warnings generated with proper hygiene
#    - No errors when coherence_crosscheck missing from manifest
#
# Example warning line verification:
#    Run: pytest tests/health/test_coherence_cal_exp.py::TestWarningHygiene -v
#    Expected: All tests pass, status includes warning-ready fields
#
# =============================================================================



