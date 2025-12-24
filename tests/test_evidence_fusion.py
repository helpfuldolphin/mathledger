"""
Tests for TDA-aware evidence fusion and promotion precheck.

This test suite validates:
1. Evidence fusion with TDA conflict detection
2. Alignment status computation (OK/WARN/BLOCK)
3. Promotion precheck CLI exit codes
4. Edge cases and error handling
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from evidence_fusion import (
    AlignmentStatus,
    FusedEvidence,
    PromotionDecision,
    RunEvidence,
    TDAMetrics,
    TDAOutcome,
    UpliftMetrics,
    detect_conflicts,
    fuse_evidence_summaries,
    load_evidence_summaries,
    save_fused_evidence,
)
from promotion_precheck import load_fused_evidence, run_precheck


def make_run_evidence(
    run_id: str,
    promotion_decision: PromotionDecision,
    tda_outcome: TDAOutcome,
    hss: float = 0.8,
    delta_p: float = 0.1,
    abstention_rate: float = 0.2,
    block_rate: float = 0.0,
) -> RunEvidence:
    """Helper to create RunEvidence objects."""
    return RunEvidence(
        run_id=run_id,
        uplift=UpliftMetrics(
            delta_p=delta_p,
            abstention_rate=abstention_rate,
            promotion_decision=promotion_decision,
        ),
        tda=TDAMetrics(
            HSS=hss,
            block_rate=block_rate,
            tda_outcome=tda_outcome,
        ),
    )


class TestEvidenceFusion:
    """Tests for evidence fusion functionality."""

    def test_pass_uplift_ok_tda_alignment_ok(self):
        """Test: PASS uplift + OK TDA → alignment OK."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.9),
            make_run_evidence("run2", PromotionDecision.PASS, TDAOutcome.OK, hss=0.85),
        ]

        fused = fuse_evidence_summaries(runs)

        assert fused.tda_alignment.alignment_status == AlignmentStatus.OK
        assert len(fused.tda_alignment.conflicted_runs) == 0
        assert len(fused.tda_alignment.hidden_instability_runs) == 0

    def test_pass_uplift_block_tda_alignment_block(self):
        """Test: PASS uplift + BLOCK TDA → alignment BLOCK."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.BLOCK, hss=0.9),
        ]

        fused = fuse_evidence_summaries(runs)

        assert fused.tda_alignment.alignment_status == AlignmentStatus.BLOCK
        assert "run1" in fused.tda_alignment.conflicted_runs
        assert len(fused.tda_alignment.hidden_instability_runs) == 0

    def test_pass_uplift_low_hss_alignment_warn(self):
        """Test: PASS uplift + low HSS → alignment WARN."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.5),
        ]

        fused = fuse_evidence_summaries(runs, hss_threshold=0.7)

        assert fused.tda_alignment.alignment_status == AlignmentStatus.WARN
        assert len(fused.tda_alignment.conflicted_runs) == 0
        assert "run1" in fused.tda_alignment.hidden_instability_runs

    def test_multiple_conflicts(self):
        """Test: Multiple runs with different conflict types."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.BLOCK, hss=0.9),
            make_run_evidence("run2", PromotionDecision.PASS, TDAOutcome.OK, hss=0.5),
            make_run_evidence("run3", PromotionDecision.PASS, TDAOutcome.OK, hss=0.9),
        ]

        fused = fuse_evidence_summaries(runs, hss_threshold=0.7)

        # BLOCK takes precedence over WARN
        assert fused.tda_alignment.alignment_status == AlignmentStatus.BLOCK
        assert "run1" in fused.tda_alignment.conflicted_runs
        assert "run2" in fused.tda_alignment.hidden_instability_runs
        assert len(fused.tda_alignment.conflicted_runs) == 1
        assert len(fused.tda_alignment.hidden_instability_runs) == 1

    def test_warn_or_block_decisions_do_not_conflict(self):
        """Test: WARN/BLOCK uplift decisions don't trigger conflicts."""
        runs = [
            make_run_evidence("run1", PromotionDecision.WARN, TDAOutcome.BLOCK, hss=0.5),
            make_run_evidence("run2", PromotionDecision.BLOCK, TDAOutcome.BLOCK, hss=0.5),
        ]

        fused = fuse_evidence_summaries(runs, hss_threshold=0.7)

        # No conflicts because promotion_decision is not PASS
        assert fused.tda_alignment.alignment_status == AlignmentStatus.OK
        assert len(fused.tda_alignment.conflicted_runs) == 0
        assert len(fused.tda_alignment.hidden_instability_runs) == 0

    def test_empty_runs_list(self):
        """Test: Empty runs list returns OK alignment."""
        fused = fuse_evidence_summaries([])

        assert fused.tda_alignment.alignment_status == AlignmentStatus.OK
        assert len(fused.runs) == 0
        assert len(fused.tda_alignment.conflicted_runs) == 0

    def test_custom_hss_threshold(self):
        """Test: Custom HSS threshold affects hidden instability detection."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.75),
        ]

        # With default threshold (0.7), no warning
        fused1 = fuse_evidence_summaries(runs, hss_threshold=0.7)
        assert fused1.tda_alignment.alignment_status == AlignmentStatus.OK

        # With higher threshold (0.8), warning
        fused2 = fuse_evidence_summaries(runs, hss_threshold=0.8)
        assert fused2.tda_alignment.alignment_status == AlignmentStatus.WARN
        assert "run1" in fused2.tda_alignment.hidden_instability_runs


class TestDetectConflicts:
    """Tests for conflict detection logic."""

    def test_detect_conflicts_pass_and_block(self):
        """Test conflict detection for PASS + BLOCK."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.BLOCK),
        ]

        alignment = detect_conflicts(runs, hss_threshold=0.7)

        assert alignment.alignment_status == AlignmentStatus.BLOCK
        assert "run1" in alignment.conflicted_runs

    def test_detect_conflicts_low_hss(self):
        """Test conflict detection for low HSS."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.5),
        ]

        alignment = detect_conflicts(runs, hss_threshold=0.7)

        assert alignment.alignment_status == AlignmentStatus.WARN
        assert "run1" in alignment.hidden_instability_runs

    def test_detect_conflicts_block_precedence(self):
        """Test that BLOCK takes precedence over WARN."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.BLOCK, hss=0.5),
        ]

        alignment = detect_conflicts(runs, hss_threshold=0.7)

        assert alignment.alignment_status == AlignmentStatus.BLOCK
        assert "run1" in alignment.conflicted_runs
        assert "run1" in alignment.hidden_instability_runs  # Also flagged for low HSS


class TestSerialization:
    """Tests for JSON serialization/deserialization."""

    def test_run_evidence_roundtrip(self):
        """Test RunEvidence to_dict/from_dict roundtrip."""
        run = make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.9)

        d = run.to_dict()
        run2 = RunEvidence.from_dict(d)

        assert run2.run_id == run.run_id
        assert run2.uplift.delta_p == run.uplift.delta_p
        assert run2.uplift.promotion_decision == run.uplift.promotion_decision
        assert run2.tda.HSS == run.tda.HSS
        assert run2.tda.tda_outcome == run.tda.tda_outcome

    def test_fused_evidence_roundtrip(self):
        """Test FusedEvidence to_dict/from_dict roundtrip."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.9),
        ]
        fused = fuse_evidence_summaries(runs)

        d = fused.to_dict()
        fused2 = FusedEvidence.from_dict(d)

        assert len(fused2.runs) == len(fused.runs)
        assert fused2.tda_alignment.alignment_status == fused.tda_alignment.alignment_status

    def test_save_and_load_fused_evidence(self, tmp_path):
        """Test saving and loading fused evidence to/from file."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.9),
        ]
        fused = fuse_evidence_summaries(runs)

        output_path = tmp_path / "fused.json"
        save_fused_evidence(fused, output_path)

        assert output_path.exists()

        loaded = load_fused_evidence(output_path)
        assert len(loaded.runs) == len(fused.runs)
        assert loaded.tda_alignment.alignment_status == fused.tda_alignment.alignment_status

    def test_load_evidence_summaries(self, tmp_path):
        """Test loading evidence summaries from file."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.9),
            make_run_evidence("run2", PromotionDecision.PASS, TDAOutcome.OK, hss=0.85),
        ]

        input_path = tmp_path / "runs.json"
        with open(input_path, "w") as f:
            json.dump({"runs": [r.to_dict() for r in runs]}, f)

        loaded = load_evidence_summaries(input_path)

        assert len(loaded) == 2
        assert loaded[0].run_id == "run1"
        assert loaded[1].run_id == "run2"

    def test_load_evidence_summaries_invalid_format(self, tmp_path):
        """Test loading evidence summaries with invalid format."""
        input_path = tmp_path / "invalid.json"
        with open(input_path, "w") as f:
            json.dump({"invalid": "format"}, f)

        with pytest.raises(ValueError, match="missing 'runs' key"):
            load_evidence_summaries(input_path)


class TestPromotionPrecheck:
    """Tests for promotion precheck CLI."""

    def test_precheck_ok_exit_code(self, tmp_path):
        """Test precheck with OK alignment returns exit code 0."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.9),
        ]
        fused = fuse_evidence_summaries(runs)

        fused_path = tmp_path / "fused.json"
        save_fused_evidence(fused, fused_path)

        exit_code = run_precheck(fused_path)
        assert exit_code == 0

    def test_precheck_warn_exit_code(self, tmp_path):
        """Test precheck with WARN alignment returns exit code 0."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.5),
        ]
        fused = fuse_evidence_summaries(runs, hss_threshold=0.7)

        fused_path = tmp_path / "fused.json"
        save_fused_evidence(fused, fused_path)

        exit_code = run_precheck(fused_path)
        assert exit_code == 0

    def test_precheck_block_exit_code(self, tmp_path):
        """Test precheck with BLOCK alignment returns exit code 1."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.BLOCK, hss=0.9),
        ]
        fused = fuse_evidence_summaries(runs)

        fused_path = tmp_path / "fused.json"
        save_fused_evidence(fused, fused_path)

        exit_code = run_precheck(fused_path)
        assert exit_code == 1

    def test_precheck_missing_file_exit_code(self, tmp_path):
        """Test precheck with missing file returns exit code 2."""
        fused_path = tmp_path / "nonexistent.json"

        exit_code = run_precheck(fused_path)
        assert exit_code == 2

    def test_precheck_invalid_file_exit_code(self, tmp_path):
        """Test precheck with invalid file returns exit code 2."""
        fused_path = tmp_path / "invalid.json"
        with open(fused_path, "w") as f:
            f.write("not valid json")

        exit_code = run_precheck(fused_path)
        assert exit_code == 2

    def test_precheck_cli_ok(self, tmp_path):
        """Test precheck CLI with OK alignment."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.9),
        ]
        fused = fuse_evidence_summaries(runs)

        fused_path = tmp_path / "fused.json"
        save_fused_evidence(fused, fused_path)

        # Run CLI
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "experiments" / "promotion_precheck.py"),
                str(fused_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "No TDA alignment issues detected" in result.stdout

    def test_precheck_cli_warn(self, tmp_path):
        """Test precheck CLI with WARN alignment."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.OK, hss=0.5),
        ]
        fused = fuse_evidence_summaries(runs, hss_threshold=0.7)

        fused_path = tmp_path / "fused.json"
        save_fused_evidence(fused, fused_path)

        # Run CLI
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "experiments" / "promotion_precheck.py"),
                str(fused_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "WARNING: Hidden instability detected" in result.stdout
        assert "WARNING: Hidden instability detected" in result.stderr

    def test_precheck_cli_block(self, tmp_path):
        """Test precheck CLI with BLOCK alignment."""
        runs = [
            make_run_evidence("run1", PromotionDecision.PASS, TDAOutcome.BLOCK, hss=0.9),
        ]
        fused = fuse_evidence_summaries(runs)

        fused_path = tmp_path / "fused.json"
        save_fused_evidence(fused, fused_path)

        # Run CLI
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "experiments" / "promotion_precheck.py"),
                str(fused_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "BLOCK: Uplift/TDA conflict detected" in result.stdout
        assert "advisory BLOCK: TDA conflict" in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
