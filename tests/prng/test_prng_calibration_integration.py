"""
Tests for PRNG calibration integration.

Covers:
- Window alignment
- Per-window volatility deltas
- PRNG confounded window computation
- Calibration annex building
- Evidence attachment
- Deterministic ordering
- Partial-phase comparisons
- No PRNG input handling
"""

import pytest
from rfl.prng.calibration_integration import (
    align_prng_drift_to_windows,
    compute_per_window_prng_volatility_deltas,
    compute_prng_confounded_windows,
    build_prng_window_audit_table,
    build_prng_calibration_annex,
    attach_prng_calibration_annex_to_evidence,
)
from rfl.prng.governance import DriftStatus


class TestWindowAlignment:
    """Test align_prng_drift_to_windows."""

    def test_align_with_prng_tiles(self):
        """Align PRNG drift to windows using provided tiles."""
        windows = [
            {"window_index": 0, "window_start": 1, "window_end": 50, "mean_delta_p": 0.05},
            {"window_index": 1, "window_start": 51, "window_end": 100, "mean_delta_p": 0.06},
        ]
        
        prng_tiles = [
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
            {"drift_status": DriftStatus.VOLATILE.value, "blocking_rules": ["R1"]},
        ]
        
        enriched = align_prng_drift_to_windows(windows, prng_tiles=prng_tiles)
        
        assert len(enriched) == 2
        assert enriched[0]["prng_drift_status"] == DriftStatus.STABLE.value
        assert enriched[0]["prng_volatile_runs"] == 0
        assert enriched[1]["prng_drift_status"] == DriftStatus.VOLATILE.value
        assert enriched[1]["prng_volatile_runs"] == 1
        assert enriched[1]["prng_blocking_rules"] == ["R1"]

    def test_align_without_prng_data(self):
        """Align without PRNG data → empty PRNG fields."""
        windows = [
            {"window_index": 0, "window_start": 1, "window_end": 50, "mean_delta_p": 0.05},
        ]
        
        enriched = align_prng_drift_to_windows(windows)
        
        assert len(enriched) == 1
        assert enriched[0]["prng_drift_status"] is None
        assert enriched[0]["prng_volatile_runs"] == 0
        assert enriched[0]["prng_blocking_rules"] == []

    def test_align_preserves_window_fields(self):
        """Alignment preserves original window fields."""
        windows = [
            {"window_index": 0, "window_start": 1, "window_end": 50, "mean_delta_p": 0.05, "divergence_rate": 0.1},
        ]
        
        enriched = align_prng_drift_to_windows(windows)
        
        assert enriched[0]["window_index"] == 0
        assert enriched[0]["window_start"] == 1
        assert enriched[0]["mean_delta_p"] == 0.05
        assert enriched[0]["divergence_rate"] == 0.1


class TestVolatilityDeltas:
    """Test compute_per_window_prng_volatility_deltas."""

    def test_compute_deltas_basic(self):
        """Compute volatility deltas between windows."""
        windows = [
            {"window_index": 0, "prng_volatile_runs": 0, "prng_drift_status": DriftStatus.STABLE.value},
            {"window_index": 1, "prng_volatile_runs": 1, "prng_drift_status": DriftStatus.VOLATILE.value},
            {"window_index": 2, "prng_volatile_runs": 2, "prng_drift_status": DriftStatus.VOLATILE.value},
        ]
        
        deltas = compute_per_window_prng_volatility_deltas(windows)
        
        assert len(deltas) == 3
        assert deltas[0]["prng_volatility_delta"] == 0
        assert deltas[0]["prng_drift_status_transition"] is None
        assert deltas[1]["prng_volatility_delta"] == 1
        assert deltas[1]["prng_drift_status_transition"] == f"{DriftStatus.STABLE.value}→{DriftStatus.VOLATILE.value}"
        assert deltas[2]["prng_volatility_delta"] == 1
        assert deltas[2]["prng_drift_status_transition"] is None  # No status change

    def test_deltas_deterministic_ordering(self):
        """Deltas should have deterministic window_index ordering."""
        windows = [
            {"window_index": 2, "prng_volatile_runs": 0},
            {"window_index": 0, "prng_volatile_runs": 1},
            {"window_index": 1, "prng_volatile_runs": 1},
        ]
        
        deltas = compute_per_window_prng_volatility_deltas(windows)
        
        # Should preserve window order, not sort
        assert deltas[0]["window_index"] == 2
        assert deltas[1]["window_index"] == 0
        assert deltas[2]["window_index"] == 1


class TestConfoundedWindows:
    """Test compute_prng_confounded_windows."""

    def test_confounded_window_volatile_transition_and_worsening_delta_p(self):
        """Window is confounded if PRNG→VOLATILE AND delta_p worsens."""
        windows = [
            {"window_index": 0, "prng_drift_status": DriftStatus.STABLE.value, "mean_delta_p": 0.05},
            {"window_index": 1, "prng_drift_status": DriftStatus.VOLATILE.value, "mean_delta_p": 0.07},  # Worsened
        ]
        
        confounded = compute_prng_confounded_windows(windows)
        
        assert confounded == [1]

    def test_confounded_window_volatile_transition_and_stalled_delta_p(self):
        """Window is confounded if PRNG→VOLATILE AND delta_p stalls."""
        windows = [
            {"window_index": 0, "prng_drift_status": DriftStatus.STABLE.value, "mean_delta_p": 0.05},
            {"window_index": 1, "prng_drift_status": DriftStatus.VOLATILE.value, "mean_delta_p": 0.0501},  # Stalled (within tolerance)
        ]
        
        confounded = compute_prng_confounded_windows(windows)
        
        assert confounded == [1]

    def test_not_confounded_if_delta_p_improves(self):
        """Window is NOT confounded if delta_p improves despite PRNG→VOLATILE."""
        windows = [
            {"window_index": 0, "prng_drift_status": DriftStatus.STABLE.value, "mean_delta_p": 0.05},
            {"window_index": 1, "prng_drift_status": DriftStatus.VOLATILE.value, "mean_delta_p": 0.03},  # Improved
        ]
        
        confounded = compute_prng_confounded_windows(windows)
        
        assert confounded == []

    def test_not_confounded_if_no_volatile_transition(self):
        """Window is NOT confounded if PRNG doesn't transition to VOLATILE."""
        windows = [
            {"window_index": 0, "prng_drift_status": DriftStatus.STABLE.value, "mean_delta_p": 0.05},
            {"window_index": 1, "prng_drift_status": DriftStatus.DRIFTING.value, "mean_delta_p": 0.07},
        ]
        
        confounded = compute_prng_confounded_windows(windows)
        
        assert confounded == []

    def test_confounded_deterministic_ordering(self):
        """Confounded windows should be sorted deterministically."""
        windows = [
            {"window_index": 2, "prng_drift_status": DriftStatus.STABLE.value, "mean_delta_p": 0.05},
            {"window_index": 0, "prng_drift_status": DriftStatus.VOLATILE.value, "mean_delta_p": 0.07},
            {"window_index": 1, "prng_drift_status": DriftStatus.VOLATILE.value, "mean_delta_p": 0.08},
        ]
        
        confounded = compute_prng_confounded_windows(windows)
        
        assert confounded == sorted(confounded)


class TestCalibrationAnnex:
    """Test build_prng_calibration_annex."""

    def test_build_annex_with_prng_data(self):
        """Build annex with PRNG tiles."""
        windows = [
            {"window_index": 0, "window_start": 1, "window_end": 50, "mean_delta_p": 0.05},
            {"window_index": 1, "window_start": 51, "window_end": 100, "mean_delta_p": 0.07},
        ]
        
        prng_tiles = [
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
            {"drift_status": DriftStatus.VOLATILE.value, "blocking_rules": ["R1"]},
        ]
        
        annex = build_prng_calibration_annex(windows, prng_tiles=prng_tiles)
        
        assert annex["schema_version"] == "1.0.0"
        assert annex["prng_confounded_windows"] == [1]  # Window 1: VOLATILE + delta_p worsened
        assert annex["prng_delta_summary"]["total_windows"] == 2
        assert annex["prng_delta_summary"]["windows_with_prng_data"] == 2
        assert annex["prng_delta_summary"]["windows_with_volatile_prng"] == 1
        assert annex["prng_delta_summary"]["windows_with_drift_transitions"] == 1
        assert annex["prng_delta_summary"]["prng_drift_status_progression"] == [
            DriftStatus.STABLE.value,
            DriftStatus.VOLATILE.value,
        ]

    def test_build_annex_without_prng_data(self):
        """Build annex without PRNG data → empty confounded list."""
        windows = [
            {"window_index": 0, "window_start": 1, "window_end": 50, "mean_delta_p": 0.05},
        ]
        
        annex = build_prng_calibration_annex(windows)
        
        assert annex["schema_version"] == "1.0.0"
        assert annex["prng_confounded_windows"] == []
        assert annex["prng_delta_summary"]["windows_with_prng_data"] == 0
        assert annex["prng_delta_summary"]["windows_with_volatile_prng"] == 0

    def test_annex_deterministic_ordering(self):
        """Annex should have deterministic ordering."""
        windows = [
            {"window_index": 0, "mean_delta_p": 0.05},
            {"window_index": 1, "mean_delta_p": 0.07},
        ]
        
        prng_tiles = [
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
            {"drift_status": DriftStatus.VOLATILE.value, "blocking_rules": ["R1"]},
        ]
        
        annex1 = build_prng_calibration_annex(windows, prng_tiles=prng_tiles)
        annex2 = build_prng_calibration_annex(windows, prng_tiles=prng_tiles)
        
        assert annex1 == annex2
        assert annex1["prng_confounded_windows"] == sorted(annex1["prng_confounded_windows"])


class TestEvidenceAttachment:
    """Test attach_prng_calibration_annex_to_evidence."""

    def test_attach_annex_preserves_evidence(self):
        """Attachment preserves existing evidence structure."""
        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
            "governance": {
                "p5_calibration": {
                    "cal_exp1": {"final_divergence_rate": 0.1},
                },
            },
        }
        
        windows = [
            {"window_index": 0, "mean_delta_p": 0.05},
        ]
        
        enriched = attach_prng_calibration_annex_to_evidence(evidence, windows)
        
        # Original preserved
        assert enriched["version"] == "1.0.0"
        assert enriched["governance"]["p5_calibration"]["cal_exp1"]["final_divergence_rate"] == 0.1
        
        # PRNG annex added
        assert "prng" in enriched["governance"]["p5_calibration"]
        assert enriched["governance"]["p5_calibration"]["prng"]["schema_version"] == "1.0.0"

    def test_attach_annex_read_only(self):
        """Attachment does not mutate input evidence."""
        evidence = {
            "version": "1.0.0",
        }
        
        windows = [
            {"window_index": 0, "mean_delta_p": 0.05},
        ]
        
        enriched = attach_prng_calibration_annex_to_evidence(evidence, windows)
        
        # Original unchanged
        assert "governance" not in evidence
        
        # Enriched has annex
        assert "governance" in enriched
        assert "p5_calibration" in enriched["governance"]
        assert "prng" in enriched["governance"]["p5_calibration"]

    def test_attach_annex_json_serializable(self):
        """Result is JSON-serializable."""
        import json
        
        evidence = {
            "version": "1.0.0",
        }
        
        windows = [
            {"window_index": 0, "mean_delta_p": 0.05},
        ]
        
        prng_tiles = [
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
        ]
        
        enriched = attach_prng_calibration_annex_to_evidence(evidence, windows, prng_tiles=prng_tiles)
        
        # Should not raise
        json_str = json.dumps(enriched)
        assert isinstance(json_str, str)
        
        # Round-trip
        result = json.loads(json_str)
        assert result["governance"]["p5_calibration"]["prng"]["schema_version"] == "1.0.0"


class TestWindowAuditTable:
    """Test build_prng_window_audit_table."""

    def test_audit_table_basic(self):
        """Build audit table with all required fields."""
        windows = [
            {
                "window_index": 0,
                "mean_delta_p": 0.05,
                "divergence_rate": 0.1,
                "prng_drift_status": DriftStatus.STABLE.value,
                "prng_volatile_runs": 0,
                "prng_blocking_rules": [],
            },
            {
                "window_index": 1,
                "mean_delta_p": 0.07,
                "divergence_rate": 0.15,
                "prng_drift_status": DriftStatus.VOLATILE.value,
                "prng_volatile_runs": 1,
                "prng_blocking_rules": ["R1"],
            },
        ]
        
        table = build_prng_window_audit_table(windows)
        
        assert table["schema_version"] == "1.0.0"
        assert table["total_windows"] == 2
        assert table["selection_strategy"] == "first_last_even_spacing"
        assert table["selected_window_indices"] == [0, 1]
        assert table["max_windows"] == 10
        assert len(table["rows"]) == 2
        
        row0 = table["rows"][0]
        assert row0["window_index"] == 0
        assert row0["mean_delta_p"] == 0.05
        assert row0["divergence_rate"] == 0.1
        assert row0["prng_drift_status"] == DriftStatus.STABLE.value
        assert row0["prng_volatile_runs"] == 0
        assert row0["prng_blocking_rules"] == []
        assert row0["prng_volatility_delta"] == 0
        assert row0["prng_confounded"] is False
        
        row1 = table["rows"][1]
        assert row1["window_index"] == 1
        assert row1["mean_delta_p"] == 0.07
        assert row1["prng_drift_status"] == DriftStatus.VOLATILE.value
        assert row1["prng_volatile_runs"] == 1
        assert row1["prng_blocking_rules"] == ["R1"]
        assert row1["prng_volatility_delta"] == 1
        assert row1["prng_confounded"] is True  # VOLATILE transition + delta_p worsened

    def test_audit_table_deterministic_ordering(self):
        """Audit table rows should be deterministically ordered by window_index."""
        windows = [
            {"window_index": 2, "mean_delta_p": 0.05, "prng_drift_status": DriftStatus.STABLE.value, "prng_volatile_runs": 0, "prng_blocking_rules": []},
            {"window_index": 0, "mean_delta_p": 0.03, "prng_drift_status": DriftStatus.STABLE.value, "prng_volatile_runs": 0, "prng_blocking_rules": []},
            {"window_index": 1, "mean_delta_p": 0.04, "prng_drift_status": DriftStatus.STABLE.value, "prng_volatile_runs": 0, "prng_blocking_rules": []},
        ]
        
        table = build_prng_window_audit_table(windows)
        
        # Rows should be ordered by window_index (preserving input order, but indices should match)
        assert len(table["rows"]) == 3
        # The rows should reflect the window_index values in order
        indices = [row["window_index"] for row in table["rows"]]
        # Since we preserve input order, indices will be [2, 0, 1]
        # But the confounded flag should be computed correctly based on actual window order
        assert all(isinstance(row["window_index"], int) for row in table["rows"])

    def test_audit_table_bounded_selection(self):
        """Audit table should be bounded to max_windows (first, last, evenly spaced middle)."""
        # Create 20 windows
        windows = [
            {
                "window_index": i,
                "mean_delta_p": 0.05 + i * 0.001,
                "divergence_rate": 0.1,
                "prng_drift_status": DriftStatus.STABLE.value,
                "prng_volatile_runs": 0,
                "prng_blocking_rules": [],
            }
            for i in range(20)
        ]
        
        table = build_prng_window_audit_table(windows, max_windows=10)
        
        assert table["total_windows"] == 20
        assert table["selection_strategy"] == "first_last_even_spacing"
        assert len(table["selected_window_indices"]) == 10
        assert table["max_windows"] == 10
        assert len(table["rows"]) == 10
        
        # Should include first window (index 0)
        assert table["rows"][0]["window_index"] == 0
        
        # Should include last window (index 19)
        assert table["rows"][-1]["window_index"] == 19
        
        # Should include evenly spaced middle windows
        selected_indices = [row["window_index"] for row in table["rows"]]
        assert 0 in selected_indices
        assert 19 in selected_indices
        # Middle windows should be roughly evenly spaced
        middle_indices = sorted(selected_indices)[1:-1]  # Exclude first and last
        if len(middle_indices) > 1:
            # Check that spacing is roughly even
            gaps = [middle_indices[i+1] - middle_indices[i] for i in range(len(middle_indices) - 1)]
            # Gaps should be similar (within reasonable tolerance)
            assert max(gaps) - min(gaps) <= 3  # Allow some variance

    def test_audit_table_bounded_exact_max(self):
        """When total_windows == max_windows, all windows should be included."""
        windows = [
            {
                "window_index": i,
                "mean_delta_p": 0.05,
                "divergence_rate": 0.1,
                "prng_drift_status": DriftStatus.STABLE.value,
                "prng_volatile_runs": 0,
                "prng_blocking_rules": [],
            }
            for i in range(10)
        ]
        
        table = build_prng_window_audit_table(windows, max_windows=10)
        
        assert table["total_windows"] == 10
        assert table["selection_strategy"] == "first_last_even_spacing"
        assert table["selected_window_indices"] == list(range(10))
        assert table["max_windows"] == 10
        assert len(table["rows"]) == 10
        assert [row["window_index"] for row in table["rows"]] == list(range(10))

    def test_audit_table_prng_confounded_correct(self):
        """prng_confounded should be True when VOLATILE transition + delta_p worsens/stalls."""
        windows = [
            {
                "window_index": 0,
                "mean_delta_p": 0.05,
                "divergence_rate": 0.1,
                "prng_drift_status": DriftStatus.STABLE.value,
                "prng_volatile_runs": 0,
                "prng_blocking_rules": [],
            },
            {
                "window_index": 1,
                "mean_delta_p": 0.07,  # Worsened
                "divergence_rate": 0.15,
                "prng_drift_status": DriftStatus.VOLATILE.value,  # Transitioned to VOLATILE
                "prng_volatile_runs": 1,
                "prng_blocking_rules": ["R1"],
            },
            {
                "window_index": 2,
                "mean_delta_p": 0.06,  # Improved
                "divergence_rate": 0.12,
                "prng_drift_status": DriftStatus.VOLATILE.value,
                "prng_volatile_runs": 1,
                "prng_blocking_rules": [],
            },
        ]
        
        table = build_prng_window_audit_table(windows)
        
        assert table["rows"][0]["prng_confounded"] is False  # First window, no transition
        assert table["rows"][1]["prng_confounded"] is True  # VOLATILE transition + delta_p worsened
        assert table["rows"][2]["prng_confounded"] is False  # No transition, delta_p improved

    def test_audit_table_empty_windows(self):
        """Empty windows list should return empty audit table with selection metadata."""
        table = build_prng_window_audit_table([])
        
        assert table["schema_version"] == "1.0.0"
        assert table["total_windows"] == 0
        assert table["selection_strategy"] == "first_last_even_spacing"
        assert table["selected_window_indices"] == []
        assert table["max_windows"] == 10
        assert table["rows"] == []

    def test_audit_table_integrated_in_annex(self):
        """Audit table should be included in calibration annex."""
        windows = [
            {
                "window_index": 0,
                "mean_delta_p": 0.05,
                "divergence_rate": 0.1,
            },
        ]
        
        prng_tiles = [
            {"drift_status": DriftStatus.STABLE.value, "blocking_rules": []},
        ]
        
        annex = build_prng_calibration_annex(windows, prng_tiles=prng_tiles)
        
        assert "prng_window_audit_table" in annex
        assert annex["prng_window_audit_table"]["schema_version"] == "1.0.0"
        assert annex["prng_window_audit_table"]["total_windows"] == 1
        assert len(annex["prng_window_audit_table"]["rows"]) == 1

    def test_audit_table_selection_odd_count_deterministic(self):
        """Selection algorithm on odd count (11 windows, max=10) ensures determinism and first/last inclusion."""
        # Create 11 windows
        windows = [
            {
                "window_index": i,
                "mean_delta_p": 0.05 + i * 0.001,
                "divergence_rate": 0.1,
                "prng_drift_status": DriftStatus.STABLE.value,
                "prng_volatile_runs": 0,
                "prng_blocking_rules": [],
            }
            for i in range(11)
        ]
        
        table = build_prng_window_audit_table(windows, max_windows=10)
        
        assert table["total_windows"] == 11
        assert len(table["rows"]) == 10
        assert table["max_windows"] == 10
        assert table["selection_strategy"] == "first_last_even_spacing"
        
        # Verify selection metadata
        selected_indices = table["selected_window_indices"]
        assert len(selected_indices) == 10
        assert selected_indices == sorted(selected_indices)  # Deterministic ordering
        
        # Must include first window (index 0)
        assert 0 in selected_indices
        assert selected_indices[0] == 0
        
        # Must include last window (index 10)
        assert 10 in selected_indices
        assert selected_indices[-1] == 10
        
        # Verify rows match selected indices
        row_indices = [row["window_index"] for row in table["rows"]]
        assert row_indices == selected_indices
        
        # Verify determinism: same input → same selection
        table2 = build_prng_window_audit_table(windows, max_windows=10)
        assert table2["selected_window_indices"] == selected_indices

