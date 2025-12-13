"""
Tests for CAL-EXP curriculum harmonic grid.

STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

Tests that:
- Per-experiment annex snapshots are emitted correctly
- Annexes are persisted to disk
- Harmonic grid aggregates across experiments correctly
- Evidence attachment includes grid
- All outputs are JSON-safe, deterministic, and non-mutating
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from backend.health.harmonic_alignment_p3p4_integration import (
    attach_curriculum_harmonic_delta_to_evidence,
    attach_curriculum_harmonic_grid_to_evidence,
    build_curriculum_harmonic_delta,
    build_curriculum_harmonic_grid,
    emit_cal_exp_curriculum_harmonic_annex,
    persist_curriculum_harmonic_annex,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_annex() -> dict[str, Any]:
    """Sample curriculum harmonic annex."""
    return {
        "schema_version": "1.0.0",
        "harmonic_band": "PARTIAL",
        "evolution_status": "EVOLVING",
        "misaligned_concepts": ["slice_b", "slice_c"],
        "priority_adjustments": ["slice_b", "slice_c"],
    }


@pytest.fixture
def sample_annexes() -> list[dict[str, Any]]:
    """Sample list of annexes from multiple experiments."""
    return [
        {
            "schema_version": "1.0.0",
            "cal_id": "CAL-EXP-1",
            "harmonic_band": "COHERENT",
            "evolution_status": "STABLE",
            "misaligned_concepts": ["slice_a"],
            "priority_adjustments": [],
        },
        {
            "schema_version": "1.0.0",
            "cal_id": "CAL-EXP-2",
            "harmonic_band": "PARTIAL",
            "evolution_status": "EVOLVING",
            "misaligned_concepts": ["slice_b", "slice_c"],
            "priority_adjustments": ["slice_b"],
        },
        {
            "schema_version": "1.0.0",
            "cal_id": "CAL-EXP-3",
            "harmonic_band": "MISMATCHED",
            "evolution_status": "DIVERGING",
            "misaligned_concepts": ["slice_b", "slice_d", "slice_e"],
            "priority_adjustments": ["slice_b", "slice_d"],
        },
    ]


@pytest.fixture
def sample_evidence() -> dict[str, Any]:
    """Sample evidence pack."""
    return {
        "timestamp": "2025-12-09T12:00:00.000000+00:00",
        "run_id": "test_run_123",
        "data": {"metrics": {"success_rate": 0.85}},
    }


# =============================================================================
# TEST GROUP 1: PER-EXPERIMENT ANNEX EMISSION
# =============================================================================


class TestEmitCalExpCurriculumHarmonicAnnex:
    """Tests for emit_cal_exp_curriculum_harmonic_annex."""

    def test_01_snapshot_has_required_keys(
        self, sample_annex: dict[str, Any]
    ) -> None:
        """Test that snapshot has all required keys."""
        snapshot = emit_cal_exp_curriculum_harmonic_annex("CAL-EXP-1", sample_annex)

        required_keys = {
            "schema_version",
            "cal_id",
            "harmonic_band",
            "evolution_status",
            "misaligned_concepts",
            "priority_adjustments",
        }
        assert required_keys.issubset(set(snapshot.keys()))

    def test_02_snapshot_extracts_correct_values(
        self, sample_annex: dict[str, Any]
    ) -> None:
        """Test that snapshot extracts correct values from annex."""
        snapshot = emit_cal_exp_curriculum_harmonic_annex("CAL-EXP-1", sample_annex)

        assert snapshot["cal_id"] == "CAL-EXP-1"
        assert snapshot["harmonic_band"] == "PARTIAL"
        assert snapshot["evolution_status"] == "EVOLVING"
        assert snapshot["misaligned_concepts"] == ["slice_b", "slice_c"]
        assert snapshot["priority_adjustments"] == ["slice_b", "slice_c"]

    def test_03_snapshot_serializes_to_json(
        self, sample_annex: dict[str, Any]
    ) -> None:
        """Test that snapshot can be serialized to JSON."""
        snapshot = emit_cal_exp_curriculum_harmonic_annex("CAL-EXP-1", sample_annex)

        json_str = json.dumps(snapshot)
        assert json_str

        deserialized = json.loads(json_str)
        assert deserialized == snapshot

    def test_04_snapshot_is_deterministic(self, sample_annex: dict[str, Any]) -> None:
        """Test that snapshot is deterministic."""
        snapshots = [
            emit_cal_exp_curriculum_harmonic_annex("CAL-EXP-1", sample_annex)
            for _ in range(5)
        ]

        for i in range(1, len(snapshots)):
            assert snapshots[0] == snapshots[i]


# =============================================================================
# TEST GROUP 2: PERSISTENCE
# =============================================================================


class TestPersistCurriculumHarmonicAnnex:
    """Tests for persist_curriculum_harmonic_annex."""

    def test_05_persists_to_correct_path(self, sample_annex: dict[str, Any]) -> None:
        """Test that snapshot is persisted to correct path."""
        snapshot = emit_cal_exp_curriculum_harmonic_annex("CAL-EXP-1", sample_annex)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            output_path = persist_curriculum_harmonic_annex(snapshot, output_dir)

            assert output_path.name == "curriculum_harmonic_annex_CAL-EXP-1.json"
            assert output_path.exists()

    def test_06_persisted_file_is_valid_json(
        self, sample_annex: dict[str, Any]
    ) -> None:
        """Test that persisted file contains valid JSON."""
        snapshot = emit_cal_exp_curriculum_harmonic_annex("CAL-EXP-1", sample_annex)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            output_path = persist_curriculum_harmonic_annex(snapshot, output_dir)

            with open(output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded == snapshot

    def test_07_round_trip_preserves_data(self, sample_annex: dict[str, Any]) -> None:
        """Test that round-trip (emit -> persist -> load) preserves data."""
        snapshot = emit_cal_exp_curriculum_harmonic_annex("CAL-EXP-1", sample_annex)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            output_path = persist_curriculum_harmonic_annex(snapshot, output_dir)

            with open(output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded["cal_id"] == snapshot["cal_id"]
            assert loaded["harmonic_band"] == snapshot["harmonic_band"]
            assert loaded["evolution_status"] == snapshot["evolution_status"]
            assert loaded["misaligned_concepts"] == snapshot["misaligned_concepts"]


# =============================================================================
# TEST GROUP 3: HARMONIC GRID
# =============================================================================


class TestBuildCurriculumHarmonicGrid:
    """Tests for build_curriculum_harmonic_grid."""

    def test_08_grid_has_required_keys(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that grid has all required keys."""
        grid = build_curriculum_harmonic_grid(sample_annexes)

        required_keys = {
            "schema_version",
            "num_experiments",
            "harmonic_band_counts",
            "top_misaligned_concepts",
        }
        assert required_keys.issubset(set(grid.keys()))

    def test_09_grid_counts_harmonic_bands(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that grid counts harmonic bands correctly."""
        grid = build_curriculum_harmonic_grid(sample_annexes)

        counts = grid["harmonic_band_counts"]
        assert counts["COHERENT"] == 1
        assert counts["PARTIAL"] == 1
        assert counts["MISMATCHED"] == 1

    def test_10_grid_finds_top_misaligned_concepts(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that grid finds top misaligned concepts by frequency."""
        grid = build_curriculum_harmonic_grid(sample_annexes)

        top_concepts = grid["top_misaligned_concepts"]
        # slice_b appears in 2 experiments (CAL-EXP-2 and CAL-EXP-3)
        assert len(top_concepts) > 0
        assert top_concepts[0]["concept"] == "slice_b"
        assert top_concepts[0]["frequency"] == 2

    def test_11_grid_sorts_concepts_by_frequency(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that grid sorts concepts by frequency (descending)."""
        grid = build_curriculum_harmonic_grid(sample_annexes)

        top_concepts = grid["top_misaligned_concepts"]
        if len(top_concepts) > 1:
            frequencies = [c["frequency"] for c in top_concepts]
            assert frequencies == sorted(frequencies, reverse=True)

    def test_12_grid_includes_experiment_list(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that grid includes list of experiments for each concept."""
        grid = build_curriculum_harmonic_grid(sample_annexes)

        top_concepts = grid["top_misaligned_concepts"]
        slice_b_entry = next((c for c in top_concepts if c["concept"] == "slice_b"), None)
        assert slice_b_entry is not None
        assert "experiments" in slice_b_entry
        assert "CAL-EXP-2" in slice_b_entry["experiments"]
        assert "CAL-EXP-3" in slice_b_entry["experiments"]

    def test_13_grid_handles_empty_list(self) -> None:
        """Test that grid handles empty annex list."""
        grid = build_curriculum_harmonic_grid([])

        assert grid["num_experiments"] == 0
        assert grid["harmonic_band_counts"]["COHERENT"] == 0
        assert grid["top_misaligned_concepts"] == []

    def test_14_grid_is_deterministic(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that grid is deterministic."""
        grids = [build_curriculum_harmonic_grid(sample_annexes) for _ in range(5)]

        for i in range(1, len(grids)):
            assert grids[0] == grids[i]

    def test_15_grid_serializes_to_json(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that grid can be serialized to JSON."""
        grid = build_curriculum_harmonic_grid(sample_annexes)

        json_str = json.dumps(grid)
        assert json_str

        deserialized = json.loads(json_str)
        assert deserialized == grid

    def test_19_grid_includes_top_driver_concepts(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that grid includes top_driver_concepts (top 5)."""
        grid = build_curriculum_harmonic_grid(sample_annexes)

        assert "top_driver_concepts" in grid
        assert isinstance(grid["top_driver_concepts"], list)
        assert len(grid["top_driver_concepts"]) <= 5
        # slice_b appears in 2 experiments, should be first
        assert grid["top_driver_concepts"][0] == "slice_b"

    def test_20_top_driver_concepts_sorted_by_frequency(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that top_driver_concepts are sorted by frequency (desc), then name (asc)."""
        grid = build_curriculum_harmonic_grid(sample_annexes)

        top_drivers = grid["top_driver_concepts"]
        # Get frequencies for verification
        concept_freqs = {
            entry["concept"]: entry["frequency"]
            for entry in grid["top_misaligned_concepts"]
        }

        # Verify sorting: frequency desc, then name asc
        for i in range(len(top_drivers) - 1):
            curr = top_drivers[i]
            next_concept = top_drivers[i + 1]
            curr_freq = concept_freqs.get(curr, 0)
            next_freq = concept_freqs.get(next_concept, 0)

            if curr_freq == next_freq:
                assert curr < next_concept  # Name ascending
            else:
                assert curr_freq >= next_freq  # Frequency descending

    def test_21_grid_includes_top_driver_cal_ids(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that grid includes top_driver_cal_ids mapping."""
        grid = build_curriculum_harmonic_grid(sample_annexes)

        assert "top_driver_cal_ids" in grid
        assert isinstance(grid["top_driver_cal_ids"], dict)

        # slice_b should have CAL-EXP-2 and CAL-EXP-3
        if "slice_b" in grid["top_driver_cal_ids"]:
            cal_ids = grid["top_driver_cal_ids"]["slice_b"]
            assert "CAL-EXP-2" in cal_ids
            assert "CAL-EXP-3" in cal_ids
            assert cal_ids == sorted(cal_ids)  # Sorted

    def test_22_top_driver_concepts_deterministic(
        self, sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that top_driver_concepts are deterministic."""
        grids = [build_curriculum_harmonic_grid(sample_annexes) for _ in range(5)]

        for i in range(1, len(grids)):
            assert grids[0]["top_driver_concepts"] == grids[i]["top_driver_concepts"]
            assert grids[0]["top_driver_cal_ids"] == grids[i]["top_driver_cal_ids"]


# =============================================================================
# TEST GROUP 4: EVIDENCE ATTACHMENT
# =============================================================================


class TestAttachCurriculumHarmonicGridToEvidence:
    """Tests for attach_curriculum_harmonic_grid_to_evidence."""

    def test_16_attaches_grid_to_evidence(
        self, sample_evidence: dict[str, Any], sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that grid is attached to evidence."""
        grid = build_curriculum_harmonic_grid(sample_annexes)
        enriched = attach_curriculum_harmonic_grid_to_evidence(sample_evidence, grid)

        assert "governance" in enriched
        assert "harmonic_curriculum_panel" in enriched["governance"]
        assert enriched["governance"]["harmonic_curriculum_panel"] == grid

    def test_17_evidence_non_mutating(
        self, sample_evidence: dict[str, Any], sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that evidence attachment does not mutate input."""
        original = dict(sample_evidence)
        grid = build_curriculum_harmonic_grid(sample_annexes)

        attach_curriculum_harmonic_grid_to_evidence(sample_evidence, grid)

        assert sample_evidence == original
        assert "governance" not in sample_evidence

    def test_18_evidence_serializes_to_json(
        self, sample_evidence: dict[str, Any], sample_annexes: list[dict[str, Any]]
    ) -> None:
        """Test that enriched evidence can be serialized to JSON."""
        grid = build_curriculum_harmonic_grid(sample_annexes)
        enriched = attach_curriculum_harmonic_grid_to_evidence(sample_evidence, grid)

        json_str = json.dumps(enriched)
        assert json_str

        deserialized = json.loads(json_str)
        assert "governance" in deserialized
        assert "harmonic_curriculum_panel" in deserialized["governance"]


# =============================================================================
# TEST GROUP 5: DELTA TRACKING
# =============================================================================


@pytest.fixture
def mock_grid() -> dict[str, Any]:
    """Sample mock grid."""
    return {
        "schema_version": "1.0.0",
        "num_experiments": 3,
        "harmonic_band_counts": {"COHERENT": 2, "PARTIAL": 1, "MISMATCHED": 0},
        "top_misaligned_concepts": [
            {"concept": "slice_a", "frequency": 2, "experiments": ["CAL-EXP-1", "CAL-EXP-2"]},
            {"concept": "slice_b", "frequency": 1, "experiments": ["CAL-EXP-1"]},
            {"concept": "slice_c", "frequency": 1, "experiments": ["CAL-EXP-3"]},
        ],
        "top_driver_concepts": ["slice_a", "slice_b", "slice_c"],
        "top_driver_cal_ids": {
            "slice_a": ["CAL-EXP-1", "CAL-EXP-2"],
            "slice_b": ["CAL-EXP-1"],
            "slice_c": ["CAL-EXP-3"],
        },
    }


@pytest.fixture
def real_grid() -> dict[str, Any]:
    """Sample real grid."""
    return {
        "schema_version": "1.0.0",
        "num_experiments": 3,
        "harmonic_band_counts": {"COHERENT": 1, "PARTIAL": 1, "MISMATCHED": 1},
        "top_misaligned_concepts": [
            {"concept": "slice_a", "frequency": 3, "experiments": ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"]},
            {"concept": "slice_d", "frequency": 2, "experiments": ["CAL-EXP-1", "CAL-EXP-2"]},
            {"concept": "slice_b", "frequency": 1, "experiments": ["CAL-EXP-3"]},
        ],
        "top_driver_concepts": ["slice_a", "slice_d", "slice_b"],
        "top_driver_cal_ids": {
            "slice_a": ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"],
            "slice_d": ["CAL-EXP-1", "CAL-EXP-2"],
            "slice_b": ["CAL-EXP-3"],
        },
    }


class TestBuildCurriculumHarmonicDelta:
    """Tests for build_curriculum_harmonic_delta."""

    def test_23_delta_has_required_keys(
        self, mock_grid: dict[str, Any], real_grid: dict[str, Any]
    ) -> None:
        """Test that delta has all required keys."""
        delta = build_curriculum_harmonic_delta(mock_grid, real_grid)

        required_keys = {
            "schema_version",
            "top_driver_overlap",
            "top_driver_only_mock",
            "top_driver_only_real",
            "frequency_shifts",
        }
        assert required_keys.issubset(set(delta.keys()))

    def test_24_delta_finds_overlap(
        self, mock_grid: dict[str, Any], real_grid: dict[str, Any]
    ) -> None:
        """Test that delta finds overlapping top driver concepts."""
        delta = build_curriculum_harmonic_delta(mock_grid, real_grid)

        # slice_a and slice_b appear in both
        assert "slice_a" in delta["top_driver_overlap"]
        assert "slice_b" in delta["top_driver_overlap"]
        assert delta["top_driver_overlap"] == sorted(delta["top_driver_overlap"])

    def test_25_delta_finds_mock_only(
        self, mock_grid: dict[str, Any], real_grid: dict[str, Any]
    ) -> None:
        """Test that delta finds concepts only in mock."""
        delta = build_curriculum_harmonic_delta(mock_grid, real_grid)

        # slice_c appears only in mock
        assert "slice_c" in delta["top_driver_only_mock"]
        assert delta["top_driver_only_mock"] == sorted(delta["top_driver_only_mock"])

    def test_26_delta_finds_real_only(
        self, mock_grid: dict[str, Any], real_grid: dict[str, Any]
    ) -> None:
        """Test that delta finds concepts only in real."""
        delta = build_curriculum_harmonic_delta(mock_grid, real_grid)

        # slice_d appears only in real
        assert "slice_d" in delta["top_driver_only_real"]
        assert delta["top_driver_only_real"] == sorted(delta["top_driver_only_real"])

    def test_27_delta_computes_frequency_shifts(
        self, mock_grid: dict[str, Any], real_grid: dict[str, Any]
    ) -> None:
        """Test that delta computes frequency shifts correctly."""
        delta = build_curriculum_harmonic_delta(mock_grid, real_grid)

        shifts = delta["frequency_shifts"]
        # slice_a: mock=2, real=3, delta=+1
        slice_a_shift = next((s for s in shifts if s["concept"] == "slice_a"), None)
        assert slice_a_shift is not None
        assert slice_a_shift["mock_frequency"] == 2
        assert slice_a_shift["real_frequency"] == 3
        assert slice_a_shift["delta"] == 1

    def test_28_delta_sorts_frequency_shifts(
        self, mock_grid: dict[str, Any], real_grid: dict[str, Any]
    ) -> None:
        """Test that frequency shifts are sorted by absolute delta (desc), then name (asc)."""
        delta = build_curriculum_harmonic_delta(mock_grid, real_grid)

        shifts = delta["frequency_shifts"]
        if len(shifts) > 1:
            for i in range(len(shifts) - 1):
                curr = shifts[i]
                next_shift = shifts[i + 1]
                curr_abs_delta = abs(curr["delta"])
                next_abs_delta = abs(next_shift["delta"])

                if curr_abs_delta == next_abs_delta:
                    assert curr["concept"] < next_shift["concept"]  # Name ascending
                else:
                    assert curr_abs_delta >= next_abs_delta  # Absolute delta descending

    def test_29_delta_is_deterministic(
        self, mock_grid: dict[str, Any], real_grid: dict[str, Any]
    ) -> None:
        """Test that delta is deterministic."""
        deltas = [
            build_curriculum_harmonic_delta(mock_grid, real_grid) for _ in range(5)
        ]

        for i in range(1, len(deltas)):
            assert deltas[0] == deltas[i]

    def test_30_delta_serializes_to_json(
        self, mock_grid: dict[str, Any], real_grid: dict[str, Any]
    ) -> None:
        """Test that delta can be serialized to JSON."""
        delta = build_curriculum_harmonic_delta(mock_grid, real_grid)

        json_str = json.dumps(delta)
        assert json_str

        deserialized = json.loads(json_str)
        assert deserialized == delta


class TestAttachCurriculumHarmonicDeltaToEvidence:
    """Tests for attach_curriculum_harmonic_delta_to_evidence."""

    def test_31_attaches_delta_to_evidence(
        self,
        sample_evidence: dict[str, Any],
        mock_grid: dict[str, Any],
        real_grid: dict[str, Any],
    ) -> None:
        """Test that delta is attached to evidence."""
        # First attach grid
        enriched = attach_curriculum_harmonic_grid_to_evidence(sample_evidence, mock_grid)

        # Then attach delta
        delta = build_curriculum_harmonic_delta(mock_grid, real_grid)
        enriched = attach_curriculum_harmonic_delta_to_evidence(enriched, delta)

        assert "governance" in enriched
        assert "harmonic_curriculum_panel" in enriched["governance"]
        assert "delta" in enriched["governance"]["harmonic_curriculum_panel"]
        assert enriched["governance"]["harmonic_curriculum_panel"]["delta"] == delta

    def test_32_delta_evidence_non_mutating(
        self,
        sample_evidence: dict[str, Any],
        mock_grid: dict[str, Any],
        real_grid: dict[str, Any],
    ) -> None:
        """Test that delta evidence attachment does not mutate input."""
        enriched = attach_curriculum_harmonic_grid_to_evidence(sample_evidence, mock_grid)
        original = dict(enriched)

        delta = build_curriculum_harmonic_delta(mock_grid, real_grid)
        attach_curriculum_harmonic_delta_to_evidence(enriched, delta)

        # Original should not have delta
        assert "delta" not in original.get("governance", {}).get("harmonic_curriculum_panel", {})

    def test_33_delta_evidence_serializes_to_json(
        self,
        sample_evidence: dict[str, Any],
        mock_grid: dict[str, Any],
        real_grid: dict[str, Any],
    ) -> None:
        """Test that enriched evidence with delta can be serialized to JSON."""
        enriched = attach_curriculum_harmonic_grid_to_evidence(sample_evidence, mock_grid)
        delta = build_curriculum_harmonic_delta(mock_grid, real_grid)
        enriched = attach_curriculum_harmonic_delta_to_evidence(enriched, delta)

        json_str = json.dumps(enriched)
        assert json_str

        deserialized = json.loads(json_str)
        assert "governance" in deserialized
        assert "harmonic_curriculum_panel" in deserialized["governance"]
        assert "delta" in deserialized["governance"]["harmonic_curriculum_panel"]



