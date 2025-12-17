"""
Tests for TDA Evidence Pack Attachment

Tests the attach_tda_to_evidence() function and related utilities.
See: docs/system_law/TDA_PhaseX_Binding.md Section 8

SHADOW MODE: All tests verify observational metrics only.
"""

import pytest

from backend.tda.evidence import (
    attach_tda_to_evidence,
    format_tda_evidence_summary,
    compute_topology_match_score,
    TDAEvidenceBlock,
)
from backend.tda.monitor import TDASummary


class TestAttachTDAToEvidence:
    """Tests for attach_tda_to_evidence function."""

    def test_attach_creates_governance_section(self):
        """Should create governance section if missing."""
        evidence = {}
        attach_tda_to_evidence(evidence)
        assert "governance" in evidence
        assert "tda" in evidence["governance"]

    def test_attach_preserves_existing_governance(self):
        """Should preserve existing governance data."""
        evidence = {"governance": {"other": "data"}}
        attach_tda_to_evidence(evidence)
        assert evidence["governance"]["other"] == "data"
        assert "tda" in evidence["governance"]

    def test_attach_with_p3_summary(self):
        """Should include P3 metrics."""
        summary = TDASummary(
            total_cycles=100,
            sns_mean=0.2,
            sns_max=0.35,
            pcs_mean=0.85,
            pcs_min=0.65,
            hss_mean=0.9,
            hss_min=0.75,
            envelope_occupancy=0.94,
        )
        evidence = {}
        attach_tda_to_evidence(evidence, tda_summary_p3=summary)

        tda = evidence["governance"]["tda"]
        assert tda["p3_synthetic"]["available"] is True
        assert tda["p3_synthetic"]["metrics"]["sns"]["mean"] == pytest.approx(0.2, abs=1e-6)
        assert tda["p3_synthetic"]["envelope"]["occupancy"] == pytest.approx(0.94, abs=1e-4)

    def test_attach_with_p4_summary(self):
        """Should include P4 metrics."""
        summary = TDASummary(
            total_cycles=100,
            drs_mean=0.05,
            drs_max=0.12,
            sns_mean=0.18,
            pcs_mean=0.82,
            hss_mean=0.88,
            envelope_occupancy=0.91,
        )
        evidence = {}
        attach_tda_to_evidence(evidence, tda_summary_p4=summary)

        tda = evidence["governance"]["tda"]
        assert tda["p4_shadow"]["available"] is True
        assert tda["p4_shadow"]["metrics"]["drs"]["mean"] == pytest.approx(0.05, abs=1e-6)

    def test_attach_with_both_summaries(self):
        """Should include topology matching when both summaries provided."""
        p3_summary = TDASummary(
            total_cycles=100,
            sns_mean=0.2,
            pcs_mean=0.85,
            hss_mean=0.9,
            envelope_occupancy=0.94,
        )
        p4_summary = TDASummary(
            total_cycles=100,
            sns_mean=0.22,
            pcs_mean=0.82,
            hss_mean=0.88,
            envelope_occupancy=0.91,
        )
        evidence = {}
        attach_tda_to_evidence(evidence, tda_summary_p3=p3_summary, tda_summary_p4=p4_summary)

        tda = evidence["governance"]["tda"]
        assert tda["topology_matching"]["available"] is True
        assert "match_score" in tda["topology_matching"]
        assert "match_quality" in tda["topology_matching"]
        assert tda["topology_matching"]["match_score"] > 0.5

    def test_attach_includes_verifier_guidance(self):
        """Should include verifier guidance section."""
        evidence = {}
        attach_tda_to_evidence(evidence)

        tda = evidence["governance"]["tda"]
        assert "verifier_guidance" in tda
        assert "metric_definitions" in tda["verifier_guidance"]
        assert "shadow_mode_contract" in tda["verifier_guidance"]

    def test_attach_sets_shadow_mode(self):
        """Should include SHADOW mode marker."""
        evidence = {}
        attach_tda_to_evidence(evidence)

        tda = evidence["governance"]["tda"]
        assert tda["mode"] == "SHADOW"


class TestComputeTopologyMatchScore:
    """Tests for topology matching score computation."""

    def test_identical_summaries_perfect_match(self):
        """Identical summaries should yield match score = 1.0."""
        summary = TDASummary(
            sns_mean=0.2,
            pcs_mean=0.85,
            hss_mean=0.9,
            envelope_occupancy=0.94,
        )
        score = compute_topology_match_score(summary, summary)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_similar_summaries_high_match(self):
        """Similar summaries should yield high match score."""
        p3 = TDASummary(
            sns_mean=0.2,
            pcs_mean=0.85,
            hss_mean=0.9,
            envelope_occupancy=0.94,
        )
        p4 = TDASummary(
            sns_mean=0.22,
            pcs_mean=0.83,
            hss_mean=0.88,
            envelope_occupancy=0.92,
        )
        score = compute_topology_match_score(p3, p4)
        assert score >= 0.75  # Should be GOOD or better

    def test_divergent_summaries_low_match(self):
        """Highly divergent summaries should yield low match score."""
        p3 = TDASummary(
            sns_mean=0.1,
            pcs_mean=0.95,
            hss_mean=0.95,
            envelope_occupancy=0.99,
        )
        p4 = TDASummary(
            sns_mean=0.5,
            pcs_mean=0.5,
            hss_mean=0.5,
            envelope_occupancy=0.5,
        )
        score = compute_topology_match_score(p3, p4)
        assert score < 0.5  # Should be POOR

    def test_match_score_range(self):
        """Match score should always be in [0.0, 1.0]."""
        for sns in [0.0, 0.2, 0.5, 0.8, 1.0]:
            for pcs in [0.0, 0.3, 0.6, 0.9, 1.0]:
                p3 = TDASummary(sns_mean=sns, pcs_mean=pcs, hss_mean=0.8, envelope_occupancy=0.9)
                p4 = TDASummary(sns_mean=0.3, pcs_mean=0.7, hss_mean=0.7, envelope_occupancy=0.8)
                score = compute_topology_match_score(p3, p4)
                assert 0.0 <= score <= 1.0


class TestFormatTDAEvidenceSummary:
    """Tests for human-readable summary formatting."""

    def test_format_with_p3_only(self):
        """Should format P3 section."""
        summary = TDASummary(
            total_cycles=100,
            sns_mean=0.2,
            sns_max=0.35,
            pcs_mean=0.85,
            pcs_min=0.65,
            hss_mean=0.9,
            hss_min=0.75,
            envelope_occupancy=0.94,
            total_red_flags=2,
        )
        output = format_tda_evidence_summary(tda_summary_p3=summary)

        assert "P3 First-Light" in output
        assert "SNS" in output
        assert "PCS" in output
        assert "HSS" in output
        assert "Envelope" in output

    def test_format_with_p4_only(self):
        """Should format P4 section."""
        summary = TDASummary(
            total_cycles=100,
            drs_mean=0.05,
            drs_max=0.12,
            sns_mean=0.18,
            pcs_mean=0.82,
            hss_mean=0.88,
            envelope_occupancy=0.91,
        )
        output = format_tda_evidence_summary(tda_summary_p4=summary)

        assert "P4 Shadow" in output
        assert "DRS" in output

    def test_format_with_both(self):
        """Should format both sections with matching."""
        p3 = TDASummary(sns_mean=0.2, pcs_mean=0.85, hss_mean=0.9, envelope_occupancy=0.94)
        p4 = TDASummary(sns_mean=0.22, pcs_mean=0.82, hss_mean=0.88, envelope_occupancy=0.91)
        output = format_tda_evidence_summary(tda_summary_p3=p3, tda_summary_p4=p4)

        assert "Topology Matching" in output
        assert "Match score" in output


class TestTDAEvidenceBlock:
    """Tests for TDAEvidenceBlock class."""

    def test_block_creation(self):
        """Should create valid evidence block."""
        block = TDAEvidenceBlock()
        d = block.to_dict()
        assert d["schema_version"] == "1.0.0"
        assert d["mode"] == "SHADOW"

    def test_block_with_p3_summary(self):
        """Should include P3 data."""
        summary = TDASummary(
            total_cycles=100,
            sns_mean=0.2,
            pcs_mean=0.85,
            hss_mean=0.9,
            envelope_occupancy=0.94,
        )
        block = TDAEvidenceBlock(p3_summary=summary)
        d = block.to_dict()
        assert d["p3_synthetic"]["available"] is True

    def test_block_includes_interpretations(self):
        """Should include metric interpretations."""
        summary = TDASummary(
            sns_mean=0.2,
            pcs_mean=0.85,
            hss_mean=0.9,
            envelope_occupancy=0.94,
        )
        block = TDAEvidenceBlock(p3_summary=summary)
        d = block.to_dict()

        assert "interpretation" in d["p3_synthetic"]["metrics"]["sns"]
        assert "interpretation" in d["p3_synthetic"]["metrics"]["pcs"]
        assert "interpretation" in d["p3_synthetic"]["metrics"]["hss"]
        assert "interpretation" in d["p3_synthetic"]["envelope"]

    def test_block_verifier_guidance(self):
        """Should include complete verifier guidance."""
        block = TDAEvidenceBlock()
        d = block.to_dict()

        guidance = d["verifier_guidance"]
        assert "metric_definitions" in guidance
        assert "SNS" in guidance["metric_definitions"]
        assert "PCS" in guidance["metric_definitions"]
        assert "DRS" in guidance["metric_definitions"]
        assert "HSS" in guidance["metric_definitions"]
        assert "envelope_definition" in guidance
        assert "topology_matching_criteria" in guidance
        assert "red_flag_semantics" in guidance
        assert "shadow_mode_contract" in guidance
