"""
Tests for RFL Evidence Fusion with TDA Integration
"""

import pytest
from pathlib import Path
import tempfile
import json

from rfl.evidence_fusion import (
    TDAOutcome,
    InconsistencyType,
    TDAFields,
    RunEntry,
    InconsistencyReport,
    FusedEvidenceSummary,
    fuse_evidence_summaries,
    save_fused_evidence,
    load_fused_evidence,
)


class TestTDAFields:
    """Tests for TDAFields dataclass."""
    
    def test_tda_fields_creation(self):
        """Test TDA fields creation with valid values."""
        tda = TDAFields(
            HSS=0.85,
            block_rate=0.15,
            tda_outcome=TDAOutcome.PASS,
        )
        
        assert tda.HSS == 0.85
        assert tda.block_rate == 0.15
        assert tda.tda_outcome == TDAOutcome.PASS
    
    def test_tda_fields_validation(self):
        """Test TDA fields validation for block_rate."""
        # Valid block rate
        tda = TDAFields(block_rate=0.5)
        assert tda.block_rate == 0.5
        
        # Invalid block rate (negative)
        with pytest.raises(ValueError):
            TDAFields(block_rate=-0.1)
        
        # Invalid block rate (> 1.0)
        with pytest.raises(ValueError):
            TDAFields(block_rate=1.5)
    
    def test_tda_fields_serialization(self):
        """Test TDA fields to_dict and from_dict."""
        tda = TDAFields(
            HSS=0.9,
            block_rate=0.1,
            tda_outcome=TDAOutcome.WARN,
        )
        
        data = tda.to_dict()
        assert data["HSS"] == 0.9
        assert data["block_rate"] == 0.1
        assert data["tda_outcome"] == "warn"
        
        # Round-trip
        tda2 = TDAFields.from_dict(data)
        assert tda2.HSS == tda.HSS
        assert tda2.block_rate == tda.block_rate
        assert tda2.tda_outcome == tda.tda_outcome


class TestRunEntry:
    """Tests for RunEntry dataclass."""
    
    def test_run_entry_creation(self):
        """Test run entry creation with TDA fields."""
        tda = TDAFields(HSS=0.8, block_rate=0.2, tda_outcome=TDAOutcome.PASS)
        
        run = RunEntry(
            run_id="run_001",
            experiment_id="EXP_001",
            slice_name="slice_a",
            mode="baseline",
            coverage_rate=0.75,
            novelty_rate=0.5,
            throughput=10.0,
            success_rate=0.8,
            abstention_fraction=0.2,
            tda=tda,
        )
        
        assert run.run_id == "run_001"
        assert run.tda.HSS == 0.8
    
    def test_run_entry_serialization(self):
        """Test run entry serialization."""
        tda = TDAFields(HSS=0.9, block_rate=0.1, tda_outcome=TDAOutcome.PASS)
        run = RunEntry(
            run_id="run_002",
            experiment_id="EXP_002",
            slice_name="slice_b",
            mode="rfl",
            coverage_rate=0.85,
            novelty_rate=0.6,
            throughput=12.0,
            success_rate=0.9,
            abstention_fraction=0.1,
            tda=tda,
        )
        
        data = run.to_dict()
        assert data["run_id"] == "run_002"
        assert data["tda"]["HSS"] == 0.9
        
        # Round-trip
        run2 = RunEntry.from_dict(data)
        assert run2.run_id == run.run_id
        assert run2.tda.HSS == run.tda.HSS


class TestFuseEvidenceSummaries:
    """Tests for fuse_evidence_summaries function."""
    
    def test_basic_fusion(self):
        """Test basic evidence fusion."""
        baseline_runs = [
            RunEntry(
                run_id=f"baseline_{i}",
                experiment_id="EXP_001",
                slice_name="slice_a",
                mode="baseline",
                coverage_rate=0.7,
                novelty_rate=0.5,
                throughput=10.0,
                success_rate=0.75,
                abstention_fraction=0.25,
                tda=TDAFields(HSS=0.8, block_rate=0.1, tda_outcome=TDAOutcome.PASS),
            )
            for i in range(3)
        ]
        
        rfl_runs = [
            RunEntry(
                run_id=f"rfl_{i}",
                experiment_id="EXP_001",
                slice_name="slice_a",
                mode="rfl",
                coverage_rate=0.85,
                novelty_rate=0.6,
                throughput=12.0,
                success_rate=0.85,
                abstention_fraction=0.15,
                tda=TDAFields(HSS=0.85, block_rate=0.08, tda_outcome=TDAOutcome.PASS),
            )
            for i in range(3)
        ]
        
        summary = fuse_evidence_summaries(
            baseline_runs=baseline_runs,
            rfl_runs=rfl_runs,
            experiment_id="EXP_001",
            slice_name="slice_a",
        )
        
        assert summary.experiment_id == "EXP_001"
        assert summary.slice_name == "slice_a"
        assert len(summary.baseline_runs) == 3
        assert len(summary.rfl_runs) == 3
        assert summary.baseline_mean_coverage == pytest.approx(0.7)
        assert summary.rfl_mean_coverage == pytest.approx(0.85)
        assert summary.fusion_hash is not None
    
    def test_missing_tda_detection(self):
        """Test detection of missing TDA data."""
        baseline_runs = [
            RunEntry(
                run_id="baseline_1",
                experiment_id="EXP_001",
                slice_name="slice_a",
                mode="baseline",
                coverage_rate=0.7,
                novelty_rate=0.5,
                throughput=10.0,
                success_rate=0.75,
                abstention_fraction=0.25,
                tda=TDAFields(),  # Missing TDA data
            )
        ]
        
        rfl_runs = []
        
        summary = fuse_evidence_summaries(
            baseline_runs=baseline_runs,
            rfl_runs=rfl_runs,
            experiment_id="EXP_001",
            slice_name="slice_a",
        )
        
        # Should detect missing TDA data
        missing_tda_inconsistencies = [
            inc for inc in summary.inconsistencies
            if inc.inconsistency_type == InconsistencyType.MISSING_TDA_DATA
        ]
        assert len(missing_tda_inconsistencies) > 0
    
    def test_high_block_rate_detection(self):
        """Test detection of high block rates."""
        rfl_runs = [
            RunEntry(
                run_id="rfl_1",
                experiment_id="EXP_001",
                slice_name="slice_a",
                mode="rfl",
                coverage_rate=0.8,
                novelty_rate=0.6,
                throughput=12.0,
                success_rate=0.8,
                abstention_fraction=0.2,
                tda=TDAFields(HSS=0.7, block_rate=0.9, tda_outcome=TDAOutcome.WARN),
            )
        ]
        
        summary = fuse_evidence_summaries(
            baseline_runs=[],
            rfl_runs=rfl_runs,
            experiment_id="EXP_001",
            slice_name="slice_a",
        )
        
        # Should detect high block rate
        high_block_inconsistencies = [
            inc for inc in summary.inconsistencies
            if inc.inconsistency_type == InconsistencyType.HIGH_BLOCK_RATE
        ]
        assert len(high_block_inconsistencies) > 0
    
    def test_tda_hard_gate_shadow_mode(self):
        """Test TDA hard gate in SHADOW mode."""
        rfl_runs = [
            RunEntry(
                run_id="rfl_1",
                experiment_id="EXP_001",
                slice_name="slice_a",
                mode="rfl",
                coverage_rate=0.8,
                novelty_rate=0.6,
                throughput=12.0,
                success_rate=0.8,
                abstention_fraction=0.2,
                tda=TDAFields(HSS=0.2, block_rate=0.95, tda_outcome=TDAOutcome.BLOCK),
            )
        ]
        
        summary = fuse_evidence_summaries(
            baseline_runs=[],
            rfl_runs=rfl_runs,
            experiment_id="EXP_001",
            slice_name="slice_a",
            tda_hard_gate_mode="SHADOW",
        )
        
        # In SHADOW mode, should not block promotion
        assert not summary.promotion_blocked
    
    def test_tda_hard_gate_enforce_mode(self):
        """Test TDA hard gate in ENFORCE mode."""
        rfl_runs = [
            RunEntry(
                run_id="rfl_1",
                experiment_id="EXP_001",
                slice_name="slice_a",
                mode="rfl",
                coverage_rate=0.8,
                novelty_rate=0.6,
                throughput=12.0,
                success_rate=0.8,
                abstention_fraction=0.2,
                tda=TDAFields(HSS=0.2, block_rate=0.95, tda_outcome=TDAOutcome.BLOCK),
            )
        ]
        
        summary = fuse_evidence_summaries(
            baseline_runs=[],
            rfl_runs=rfl_runs,
            experiment_id="EXP_001",
            slice_name="slice_a",
            tda_hard_gate_mode="ENFORCE",
        )
        
        # In ENFORCE mode, should block promotion
        assert summary.promotion_blocked
        assert "TDA Hard Gate" in summary.promotion_block_reason
    
    def test_deterministic_fusion_hash(self):
        """Test that fusion hash is deterministic."""
        runs = [
            RunEntry(
                run_id=f"run_{i}",
                experiment_id="EXP_001",
                slice_name="slice_a",
                mode="baseline",
                coverage_rate=0.7,
                novelty_rate=0.5,
                throughput=10.0,
                success_rate=0.75,
                abstention_fraction=0.25,
                tda=TDAFields(HSS=0.8, block_rate=0.1, tda_outcome=TDAOutcome.PASS),
            )
            for i in range(3)
        ]
        
        summary1 = fuse_evidence_summaries(
            baseline_runs=runs,
            rfl_runs=[],
            experiment_id="EXP_001",
            slice_name="slice_a",
        )
        
        summary2 = fuse_evidence_summaries(
            baseline_runs=runs,
            rfl_runs=[],
            experiment_id="EXP_001",
            slice_name="slice_a",
        )
        
        assert summary1.fusion_hash == summary2.fusion_hash


class TestFusedEvidencePersistence:
    """Tests for fused evidence save/load."""
    
    def test_save_and_load(self):
        """Test saving and loading fused evidence."""
        baseline_runs = [
            RunEntry(
                run_id="baseline_1",
                experiment_id="EXP_001",
                slice_name="slice_a",
                mode="baseline",
                coverage_rate=0.7,
                novelty_rate=0.5,
                throughput=10.0,
                success_rate=0.75,
                abstention_fraction=0.25,
                tda=TDAFields(HSS=0.8, block_rate=0.1, tda_outcome=TDAOutcome.PASS),
            )
        ]
        
        summary = fuse_evidence_summaries(
            baseline_runs=baseline_runs,
            rfl_runs=[],
            experiment_id="EXP_001",
            slice_name="slice_a",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fused_evidence.json"
            save_fused_evidence(summary, output_path)
            
            # Load and verify
            loaded = load_fused_evidence(output_path)
            
            assert loaded.experiment_id == summary.experiment_id
            assert loaded.fusion_hash == summary.fusion_hash
            assert len(loaded.baseline_runs) == len(summary.baseline_runs)
