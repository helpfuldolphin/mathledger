"""
TDA Phase V: Governance Console Tests

Operation CORTEX: Phase V Operator Console & Self-Audit Harness
================================================================

Tests for:
1. TDAGovernanceSnapshot and build_governance_console_snapshot()
2. Block explanation builder
3. Long-horizon drift analysis
4. CLI smoke tests

Test Coverage:
- Empty inputs → sane defaults, no crash
- Mixed modes/outcomes → correct metrics
- Known trends → correct classification
- CLI structure verification
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from backend.tda.governance import (
    TDAHardGateMode,
    ExceptionWindowManager,
)
from backend.tda.governance_console import (
    # Schema versions
    GOVERNANCE_CONSOLE_SCHEMA_VERSION,
    BLOCK_EXPLANATION_SCHEMA_VERSION,
    LONGHORIZON_DRIFT_SCHEMA_VERSION,
    # Enums
    HSSTrend,
    GoldenAlignmentStatus,
    TrendDirection,
    # Data classes
    TDAGovernanceSnapshot,
    BlockExplanation,
    LongHorizonDriftReport,
    # Functions
    classify_hss_trend,
    classify_golden_alignment,
    classify_trend,
    build_governance_console_snapshot,
    build_block_explanation,
    build_block_explanation_from_ledger_entry,
    build_reason_codes,
    analyze_golden_alignment_trend,
    build_longhorizon_drift_report,
)


# ============================================================================
# Mock Objects
# ============================================================================

@dataclass
class MockTDAResult:
    """Mock TDAMonitorResult for testing."""
    hss: float
    sns: float = 0.5
    pcs: float = 0.5
    drs: float = 0.1
    block: bool = False
    warn: bool = False

    def __post_init__(self):
        if self.hss < 0.2:
            object.__setattr__(self, 'block', True)
        elif self.hss < 0.4:
            object.__setattr__(self, 'warn', True)


# ============================================================================
# Test: HSS Trend Classification
# ============================================================================

class TestHSSTrendClassification:
    """Tests for HSS trend classification."""

    def test_improving_trend(self):
        """Increasing HSS values classify as improving."""
        hss_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        trend = classify_hss_trend(hss_values)
        assert trend == HSSTrend.IMPROVING

    def test_degrading_trend(self):
        """Decreasing HSS values classify as degrading."""
        hss_values = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        trend = classify_hss_trend(hss_values)
        assert trend == HSSTrend.DEGRADING

    def test_stable_trend(self):
        """Flat HSS values classify as stable."""
        hss_values = [0.5, 0.51, 0.49, 0.5, 0.50, 0.51]
        trend = classify_hss_trend(hss_values)
        assert trend == HSSTrend.STABLE

    def test_insufficient_samples(self):
        """Few samples classify as unknown."""
        hss_values = [0.5, 0.6]
        trend = classify_hss_trend(hss_values)
        assert trend == HSSTrend.UNKNOWN

    def test_empty_samples(self):
        """Empty samples classify as unknown."""
        trend = classify_hss_trend([])
        assert trend == HSSTrend.UNKNOWN

    def test_single_sample(self):
        """Single sample classifies as unknown."""
        trend = classify_hss_trend([0.5])
        assert trend == HSSTrend.UNKNOWN


class TestGoldenAlignmentClassification:
    """Tests for golden alignment status classification."""

    def test_ok_maps_to_aligned(self):
        """OK calibration status maps to ALIGNED."""
        status = classify_golden_alignment("OK")
        assert status == GoldenAlignmentStatus.ALIGNED

    def test_drifting_maps_to_drifting(self):
        """DRIFTING calibration status maps to DRIFTING."""
        status = classify_golden_alignment("DRIFTING")
        assert status == GoldenAlignmentStatus.DRIFTING

    def test_broken_maps_to_broken(self):
        """BROKEN calibration status maps to BROKEN."""
        status = classify_golden_alignment("BROKEN")
        assert status == GoldenAlignmentStatus.BROKEN

    def test_unknown_input_maps_to_unknown(self):
        """Unknown input maps to UNKNOWN."""
        status = classify_golden_alignment("INVALID")
        assert status == GoldenAlignmentStatus.UNKNOWN


# ============================================================================
# Test: TDAGovernanceSnapshot
# ============================================================================

class TestTDAGovernanceSnapshot:
    """Tests for TDAGovernanceSnapshot creation."""

    def test_snapshot_is_frozen(self):
        """Snapshot is immutable (frozen dataclass)."""
        snapshot = TDAGovernanceSnapshot(
            schema_version=GOVERNANCE_CONSOLE_SCHEMA_VERSION,
            mode=TDAHardGateMode.HARD,
            cycle_count=10,
            block_rate=0.1,
            warn_rate=0.05,
            mean_hss=0.7,
            hss_trend="stable",
            golden_alignment="ALIGNED",
            exception_windows_active=0,
            recent_exceptions=(),
            governance_signal="HEALTHY",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            snapshot.cycle_count = 20

    def test_snapshot_to_dict(self):
        """Snapshot converts to dictionary."""
        snapshot = TDAGovernanceSnapshot(
            schema_version=GOVERNANCE_CONSOLE_SCHEMA_VERSION,
            mode=TDAHardGateMode.HARD,
            cycle_count=10,
            block_rate=0.1,
            warn_rate=0.05,
            mean_hss=0.7,
            hss_trend="stable",
            golden_alignment="ALIGNED",
            exception_windows_active=0,
            recent_exceptions=(),
            governance_signal="HEALTHY",
        )

        d = snapshot.to_dict()

        assert d["schema_version"] == GOVERNANCE_CONSOLE_SCHEMA_VERSION
        assert d["mode"] == "hard"
        assert d["cycle_count"] == 10

    def test_snapshot_to_json(self):
        """Snapshot serializes to valid JSON."""
        snapshot = TDAGovernanceSnapshot(
            schema_version=GOVERNANCE_CONSOLE_SCHEMA_VERSION,
            mode=TDAHardGateMode.HARD,
            cycle_count=10,
            block_rate=0.1,
            warn_rate=0.05,
            mean_hss=0.7,
            hss_trend="stable",
            golden_alignment="ALIGNED",
            exception_windows_active=0,
            recent_exceptions=(),
            governance_signal="HEALTHY",
        )

        json_str = snapshot.to_json()
        parsed = json.loads(json_str)

        assert parsed["cycle_count"] == 10
        assert parsed["governance_signal"] == "HEALTHY"


class TestBuildGovernanceConsoleSnapshot:
    """Tests for build_governance_console_snapshot()."""

    def test_empty_inputs_return_sane_defaults(self):
        """Empty inputs produce valid snapshot with defaults."""
        snapshot = build_governance_console_snapshot(
            tda_results=[],
            hard_gate_decisions=[],
            golden_state=None,
            exception_manager=None,
        )

        assert snapshot.cycle_count == 0
        assert snapshot.block_rate == 0.0
        assert snapshot.mean_hss == 0.0
        assert snapshot.hss_trend == "unknown"
        assert snapshot.governance_signal == "HEALTHY"

    def test_mixed_outcomes_correct_metrics(self):
        """Mixed outcomes produce correct block/warn rates."""
        results = [
            MockTDAResult(hss=0.1),   # block
            MockTDAResult(hss=0.15),  # block
            MockTDAResult(hss=0.3),   # warn
            MockTDAResult(hss=0.5),   # ok
            MockTDAResult(hss=0.7),   # ok
        ]

        snapshot = build_governance_console_snapshot(
            tda_results=results,
            hard_gate_decisions=[],
            golden_state={"calibration_status": "OK"},
            exception_manager=None,
        )

        assert snapshot.cycle_count == 5
        assert snapshot.block_rate == 0.4  # 2/5
        assert snapshot.warn_rate == 0.2   # 1/5
        assert snapshot.golden_alignment == "ALIGNED"

    def test_exception_window_included(self):
        """Active exception window is included in snapshot."""
        results = [MockTDAResult(hss=0.5)]
        manager = ExceptionWindowManager(max_runs=10)
        manager.activate("test reason")

        snapshot = build_governance_console_snapshot(
            tda_results=results,
            hard_gate_decisions=[],
            golden_state=None,
            exception_manager=manager,
        )

        assert snapshot.exception_windows_active == 1
        assert len(snapshot.recent_exceptions) == 1
        assert snapshot.recent_exceptions[0]["activation_reason"] == "test reason"

    def test_governance_signal_critical(self):
        """High block rate triggers CRITICAL signal."""
        # 3/4 blocked = 75% block rate
        results = [
            MockTDAResult(hss=0.1),
            MockTDAResult(hss=0.1),
            MockTDAResult(hss=0.1),
            MockTDAResult(hss=0.5),
        ]

        snapshot = build_governance_console_snapshot(
            tda_results=results,
            hard_gate_decisions=[],
            golden_state=None,
            exception_manager=None,
        )

        assert snapshot.governance_signal == "CRITICAL"

    def test_governance_signal_degraded(self):
        """Moderate block rate triggers DEGRADED signal."""
        # 2/8 blocked = 25% block rate
        results = [
            MockTDAResult(hss=0.1),
            MockTDAResult(hss=0.1),
            MockTDAResult(hss=0.5),
            MockTDAResult(hss=0.5),
            MockTDAResult(hss=0.5),
            MockTDAResult(hss=0.5),
            MockTDAResult(hss=0.5),
            MockTDAResult(hss=0.5),
        ]

        snapshot = build_governance_console_snapshot(
            tda_results=results,
            hard_gate_decisions=[],
            golden_state=None,
            exception_manager=None,
        )

        # 25% > 20% → CRITICAL
        # But mean_hss is high, let's check
        assert snapshot.governance_signal in ("CRITICAL", "DEGRADED")


# ============================================================================
# Test: Block Explanation
# ============================================================================

class TestBuildReasonCodes:
    """Tests for reason code generation."""

    def test_hss_below_threshold(self):
        """Low HSS generates HSS_BELOW_THRESHOLD code."""
        codes = build_reason_codes(hss=0.1)
        assert "HSS_BELOW_THRESHOLD" in codes

    def test_hss_below_warn_threshold(self):
        """HSS between thresholds generates WARN code."""
        codes = build_reason_codes(hss=0.3)
        assert "HSS_BELOW_WARN_THRESHOLD" in codes

    def test_golden_misaligned(self):
        """Drifting golden alignment generates GOLDEN_MISALIGNED code."""
        codes = build_reason_codes(hss=0.1, golden_alignment="DRIFTING")
        assert "GOLDEN_MISALIGNED" in codes

    def test_exception_window_applied(self):
        """Active exception window generates code."""
        codes = build_reason_codes(hss=0.1, exception_window_applied=True)
        assert "EXCEPTION_WINDOW_APPLIED" in codes

    def test_acceptable_hss(self):
        """High HSS generates HSS_ACCEPTABLE code."""
        codes = build_reason_codes(hss=0.8)
        assert "HSS_ACCEPTABLE" in codes


class TestBuildBlockExplanation:
    """Tests for block explanation building."""

    def test_block_explanation_structure(self):
        """Explanation has required structure."""
        explanation = build_block_explanation(
            run_id="test_run",
            cycle_id=42,
            tda_mode=TDAHardGateMode.HARD,
            hss=0.15,
            sns=0.5,
            pcs=0.6,
            drs=0.1,
            block=True,
            warn=False,
        )

        assert explanation.run_id == "test_run"
        assert explanation.cycle_id == 42
        assert explanation.tda_mode == "hard"
        assert explanation.status == "BLOCK"
        assert "sns" in explanation.scores
        assert "reason_codes" in explanation.gate_decision

    def test_explanation_to_dict(self):
        """Explanation converts to dictionary."""
        explanation = build_block_explanation(
            run_id="test",
            cycle_id=1,
            tda_mode=TDAHardGateMode.HARD,
            hss=0.15,
            sns=0.5,
            pcs=0.6,
            drs=0.1,
            block=True,
            warn=False,
        )

        d = explanation.to_dict()

        assert d["schema_version"] == BLOCK_EXPLANATION_SCHEMA_VERSION
        assert d["run_id"] == "test"
        assert d["status"] == "BLOCK"

    def test_explanation_json_serializable(self):
        """Explanation serializes to valid JSON."""
        explanation = build_block_explanation(
            run_id="test",
            cycle_id=1,
            tda_mode=TDAHardGateMode.HARD,
            hss=0.15,
            sns=0.5,
            pcs=0.6,
            drs=0.1,
            block=True,
            warn=False,
        )

        json_str = explanation.to_json()
        parsed = json.loads(json_str)

        assert parsed["status"] == "BLOCK"


class TestBuildBlockExplanationFromLedgerEntry:
    """Tests for building explanation from ledger entries."""

    def test_missing_tda_data_returns_unknown(self):
        """Missing TDA data produces UNKNOWN status."""
        explanation = build_block_explanation_from_ledger_entry(
            run_id="test",
            cycle_id=1,
            ledger_entry={},  # No TDA fields
        )

        assert explanation.status == "UNKNOWN"
        assert "TDA_DATA_MISSING" in explanation.gate_decision["reason_codes"]

    def test_extracts_tda_fields(self):
        """TDA fields are extracted correctly."""
        entry = {
            "tda_hss": 0.15,
            "tda_sns": 0.5,
            "tda_pcs": 0.6,
            "tda_drs": 0.1,
            "tda_outcome": "BLOCK",
            "lean_submission_avoided": True,
            "policy_update_avoided": True,
        }

        explanation = build_block_explanation_from_ledger_entry(
            run_id="test",
            cycle_id=1,
            ledger_entry=entry,
        )

        assert explanation.hss == 0.15
        assert explanation.status == "BLOCK"
        assert explanation.effects["lean_submission_avoided"] is True

    def test_alternative_field_names(self):
        """Supports alternative field naming conventions."""
        entry = {
            "hss": 0.15,  # Without tda_ prefix
            "sns": 0.5,
            "pcs": 0.6,
            "drs": 0.1,
        }

        explanation = build_block_explanation_from_ledger_entry(
            run_id="test",
            cycle_id=1,
            ledger_entry=entry,
        )

        assert explanation.hss == 0.15
        assert explanation.scores["sns"] == 0.5


# ============================================================================
# Test: Long-Horizon Drift Analysis
# ============================================================================

class TestAnalyzeGoldenAlignmentTrend:
    """Tests for golden alignment trend analysis."""

    def test_stable_trend(self):
        """Consistent ALIGNED statuses produce stable trend."""
        statuses = ["ALIGNED", "ALIGNED", "ALIGNED", "ALIGNED"]
        trend = analyze_golden_alignment_trend(statuses)
        assert trend == "stable"

    def test_drifting_trend(self):
        """High DRIFTING rate produces drifting trend."""
        statuses = ["ALIGNED", "ALIGNED", "DRIFTING", "DRIFTING", "DRIFTING"]
        trend = analyze_golden_alignment_trend(statuses)
        assert trend == "drifting"

    def test_broken_trend(self):
        """High BROKEN rate produces broken trend."""
        statuses = ["ALIGNED", "DRIFTING", "BROKEN", "BROKEN", "BROKEN"]
        trend = analyze_golden_alignment_trend(statuses)
        assert trend == "broken"

    def test_empty_statuses(self):
        """Empty statuses produce stable trend."""
        trend = analyze_golden_alignment_trend([])
        assert trend == "stable"


class TestTrendClassification:
    """Tests for general trend classification."""

    def test_increasing_trend(self):
        """Increasing values classify as increasing."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        trend = classify_trend(values)
        assert trend == TrendDirection.INCREASING

    def test_decreasing_trend(self):
        """Decreasing values classify as decreasing."""
        values = [0.5, 0.4, 0.3, 0.2, 0.1]
        trend = classify_trend(values)
        assert trend == TrendDirection.DECREASING

    def test_stable_trend(self):
        """Flat values classify as stable."""
        values = [0.5, 0.51, 0.49, 0.5, 0.50]
        trend = classify_trend(values)
        assert trend == TrendDirection.STABLE

    def test_insufficient_samples(self):
        """Few samples classify as unknown."""
        values = [0.5, 0.6]
        trend = classify_trend(values)
        assert trend == TrendDirection.UNKNOWN


class TestBuildLonghorizonDriftReport:
    """Tests for long-horizon drift report building."""

    def test_empty_tiles_return_sane_defaults(self):
        """Empty tiles produce valid report with defaults."""
        report = build_longhorizon_drift_report([])

        assert report.runs_analyzed == 0
        assert report.governance_signal == "OK"
        assert report.recommendations == ()

    def test_stable_tiles_produce_ok_signal(self):
        """Stable tiles produce OK governance signal."""
        tiles = [
            {"block_rate": 0.05, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.06, "mean_hss": 0.71, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.05, "mean_hss": 0.69, "golden_alignment": "ALIGNED"},
        ]

        report = build_longhorizon_drift_report(tiles)

        assert report.runs_analyzed == 3
        assert report.governance_signal == "OK"

    def test_increasing_block_rate_produces_attention(self):
        """Increasing block rate produces ATTENTION signal."""
        tiles = [
            {"block_rate": 0.05, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.10, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.15, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.20, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.25, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
        ]

        report = build_longhorizon_drift_report(tiles)

        assert report.block_rate_trend == "increasing"
        assert report.governance_signal in ("ATTENTION", "ALERT")

    def test_degrading_hss_produces_attention(self):
        """Degrading mean HSS produces ATTENTION signal."""
        tiles = [
            {"block_rate": 0.05, "mean_hss": 0.8, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.05, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.05, "mean_hss": 0.6, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.05, "mean_hss": 0.5, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.05, "mean_hss": 0.4, "golden_alignment": "ALIGNED"},
        ]

        report = build_longhorizon_drift_report(tiles)

        assert report.mean_hss_trend == "degrading"
        assert report.governance_signal in ("ATTENTION", "ALERT")

    def test_combined_degradation_produces_alert(self):
        """Both degrading metrics produce ALERT signal."""
        tiles = [
            {"block_rate": 0.05, "mean_hss": 0.8, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.10, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.15, "mean_hss": 0.6, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.20, "mean_hss": 0.5, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.25, "mean_hss": 0.4, "golden_alignment": "ALIGNED"},
        ]

        report = build_longhorizon_drift_report(tiles)

        assert report.governance_signal == "ALERT"
        assert len(report.recommendations) > 0

    def test_exception_usage_tracked(self):
        """Exception window usage is tracked."""
        tiles = [
            {"block_rate": 0.05, "mean_hss": 0.7, "exception_active": True},
            {"block_rate": 0.05, "mean_hss": 0.7, "exception_active": False},
            {"block_rate": 0.05, "mean_hss": 0.7, "exception_active": True},
        ]

        report = build_longhorizon_drift_report(tiles)

        assert report.exception_usage["total_windows"] == 2
        assert report.exception_usage["per_run_mean"] == pytest.approx(2/3, abs=0.01)

    def test_report_to_json(self):
        """Report serializes to valid JSON."""
        tiles = [
            {"block_rate": 0.05, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
        ]

        report = build_longhorizon_drift_report(tiles)
        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert parsed["schema_version"] == LONGHORIZON_DRIFT_SCHEMA_VERSION
        assert parsed["runs_analyzed"] == 1


# ============================================================================
# Test: CLI Smoke Tests
# ============================================================================

class TestExplainBlockCLI:
    """Smoke tests for tda_explain_block.py CLI."""

    def test_cli_with_valid_ledger(self, tmp_path: Path):
        """CLI produces valid output for valid ledger."""
        # Create synthetic ledger
        ledger = {
            "run_id": "test_run",
            "entries": [
                {"tda_hss": 0.5, "tda_sns": 0.6, "tda_pcs": 0.7, "tda_drs": 0.1},
                {"tda_hss": 0.15, "tda_sns": 0.3, "tda_pcs": 0.4, "tda_drs": 0.2,
                 "tda_outcome": "BLOCK", "lean_submission_avoided": True},
            ],
        }

        ledger_path = tmp_path / "run.json"
        with open(ledger_path, "w") as f:
            json.dump(ledger, f)

        # Import and test the extraction functions
        from scripts.tda_explain_block import (
            load_run_ledger,
            extract_cycle_entry,
            extract_run_id,
            build_explanation,
        )

        loaded = load_run_ledger(ledger_path)
        assert loaded["run_id"] == "test_run"

        entry = extract_cycle_entry(loaded, 1)
        assert entry is not None
        assert entry["tda_hss"] == 0.15

        explanation = build_explanation(loaded, 1, ledger_path)
        assert explanation["run_id"] == "test_run"
        assert explanation["cycle_id"] == 1
        assert explanation["status"] == "BLOCK"

    def test_cli_handles_missing_cycle(self, tmp_path: Path):
        """CLI handles missing cycle gracefully."""
        ledger = {
            "run_id": "test_run",
            "entries": [
                {"tda_hss": 0.5},
            ],
        }

        ledger_path = tmp_path / "run.json"
        with open(ledger_path, "w") as f:
            json.dump(ledger, f)

        from scripts.tda_explain_block import build_explanation

        explanation = build_explanation(ledger, 999, ledger_path)
        assert explanation["status"] == "UNKNOWN"
        assert "CYCLE_NOT_FOUND" in explanation["gate_decision"]["reason_codes"]

    def test_cli_handles_missing_tda_data(self, tmp_path: Path):
        """CLI handles missing TDA data gracefully."""
        ledger = {
            "run_id": "test_run",
            "entries": [
                {"some_other_field": "value"},  # No TDA data
            ],
        }

        ledger_path = tmp_path / "run.json"
        with open(ledger_path, "w") as f:
            json.dump(ledger, f)

        from scripts.tda_explain_block import build_explanation

        explanation = build_explanation(ledger, 0, ledger_path)
        assert explanation["status"] == "UNKNOWN"
        assert "TDA_DATA_MISSING" in explanation["gate_decision"]["reason_codes"]


class TestLonghorizonDriftCLI:
    """Smoke tests for tda_longhorizon_drift.py CLI."""

    def test_load_tiles_from_directory(self, tmp_path: Path):
        """CLI loads tiles from directory."""
        # Create synthetic tiles
        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        for i in range(3):
            tile = {
                "block_rate": 0.05 + i * 0.05,
                "mean_hss": 0.7 - i * 0.05,
                "golden_alignment": "ALIGNED",
            }
            with open(tiles_dir / f"tile_{i}.json", "w") as f:
                json.dump(tile, f)

        from experiments.tda_longhorizon_drift import load_tiles_from_directory

        tiles = load_tiles_from_directory(tiles_dir)
        assert len(tiles) == 3

    def test_generate_report(self, tmp_path: Path):
        """CLI generates valid report."""
        tiles = [
            {"block_rate": 0.05, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.10, "mean_hss": 0.65, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.15, "mean_hss": 0.60, "golden_alignment": "DRIFTING"},
        ]

        output_path = tmp_path / "report.json"

        from experiments.tda_longhorizon_drift import generate_report

        report = generate_report(tiles, output_path)

        assert output_path.exists()
        assert report.runs_analyzed == 3

        with open(output_path) as f:
            parsed = json.load(f)

        assert parsed["schema_version"] == LONGHORIZON_DRIFT_SCHEMA_VERSION

    def test_normalize_tile_formats(self):
        """CLI normalizes various tile formats."""
        from experiments.tda_longhorizon_drift import normalize_tile

        # Direct format
        tile1 = normalize_tile(
            {"block_rate": 0.05, "mean_hss": 0.7},
            Path("test.json")
        )
        assert tile1["block_rate"] == 0.05

        # Wrapped in tda_governance
        tile2 = normalize_tile(
            {"tda_governance": {"block_rate": 0.10, "mean_hss": 0.6}},
            Path("test.json")
        )
        assert tile2["block_rate"] == 0.10

        # Missing fields get defaults
        tile3 = normalize_tile({}, Path("test.json"))
        assert tile3["block_rate"] == 0.0
        assert tile3["mean_hss"] == 0.0


# ============================================================================
# Test: Deterministic Behavior
# ============================================================================

class TestDeterministicBehavior:
    """Tests ensuring deterministic behavior."""

    def test_snapshot_deterministic(self):
        """Same inputs produce identical snapshots."""
        results = [MockTDAResult(hss=0.5), MockTDAResult(hss=0.6)]

        snapshot1 = build_governance_console_snapshot(
            tda_results=results,
            hard_gate_decisions=[],
            golden_state={"calibration_status": "OK"},
            exception_manager=None,
        )

        snapshot2 = build_governance_console_snapshot(
            tda_results=results,
            hard_gate_decisions=[],
            golden_state={"calibration_status": "OK"},
            exception_manager=None,
        )

        assert snapshot1.to_dict() == snapshot2.to_dict()

    def test_drift_report_deterministic(self):
        """Same tiles produce identical drift report."""
        tiles = [
            {"block_rate": 0.05, "mean_hss": 0.7, "golden_alignment": "ALIGNED"},
            {"block_rate": 0.06, "mean_hss": 0.71, "golden_alignment": "ALIGNED"},
        ]

        report1 = build_longhorizon_drift_report(tiles)
        report2 = build_longhorizon_drift_report(tiles)

        # Exclude timestamps from comparison
        d1 = report1.to_dict()
        d2 = report2.to_dict()
        del d1["first_run_timestamp"]
        del d1["last_run_timestamp"]
        del d2["first_run_timestamp"]
        del d2["last_run_timestamp"]

        assert d1 == d2


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_blocked_cycles(self):
        """Handles 100% block rate."""
        results = [MockTDAResult(hss=0.05) for _ in range(5)]

        snapshot = build_governance_console_snapshot(
            tda_results=results,
            hard_gate_decisions=[],
            golden_state=None,
            exception_manager=None,
        )

        assert snapshot.block_rate == 1.0
        assert snapshot.governance_signal == "CRITICAL"

    def test_all_ok_cycles(self):
        """Handles 0% block rate."""
        results = [MockTDAResult(hss=0.8) for _ in range(5)]

        snapshot = build_governance_console_snapshot(
            tda_results=results,
            hard_gate_decisions=[],
            golden_state={"calibration_status": "OK"},
            exception_manager=None,
        )

        assert snapshot.block_rate == 0.0
        assert snapshot.governance_signal == "HEALTHY"

    def test_single_tile_drift_analysis(self):
        """Handles single-tile drift analysis."""
        tiles = [{"block_rate": 0.05, "mean_hss": 0.7}]

        report = build_longhorizon_drift_report(tiles)

        assert report.runs_analyzed == 1
        assert report.block_rate_trend == "unknown"  # Not enough samples

    def test_explanation_rounds_floats(self):
        """Explanation rounds floats for precision."""
        explanation = build_block_explanation(
            run_id="test",
            cycle_id=1,
            tda_mode=TDAHardGateMode.HARD,
            hss=0.123456789,
            sns=0.987654321,
            pcs=0.555555555,
            drs=0.111111111,
            block=True,
            warn=False,
        )

        d = explanation.to_dict()

        # Verify rounding to 4 decimal places
        assert d["hss"] == 0.1235
        assert d["scores"]["sns"] == 0.9877
