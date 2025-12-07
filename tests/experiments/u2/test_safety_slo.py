"""
Tests for U2 Safety SLO Engine

Verifies:
- Timeline building determinism
- Scenario matrix construction
- SLO evaluation logic (OK/WARN/BLOCK)
- Type safety contracts
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from experiments.u2.safety_slo import (
    SafetyEnvelope,
    SafetyStatus,
    SafetySLOPoint,
    SafetySLOTimeline,
    ScenarioSafetyCell,
    ScenarioSafetyMatrix,
    SafetySLOEvaluation,
    build_safety_slo_timeline,
    build_scenario_safety_matrix,
    evaluate_safety_slo,
    MAX_BLOCK_RATE,
    MAX_WARN_RATE,
    MAX_PERF_FAILURE_RATE,
)


class TestSafetyEnvelopeTypes:
    """Test SafetyEnvelope TypedDict structure."""
    
    def test_create_valid_envelope(self):
        """Can create a valid SafetyEnvelope."""
        envelope: SafetyEnvelope = {
            "schema_version": "1.0",
            "config": {"slice": "test"},
            "perf_ok": True,
            "safety_status": "OK",
            "lint_issues": [],
            "warnings": [],
            "run_id": "run_001",
            "slice_name": "slice_easy",
            "mode": "baseline",
            "timestamp": "2025-01-01T00:00:00",
        }
        
        assert envelope["run_id"] == "run_001"
        assert envelope["safety_status"] == "OK"
        assert envelope["mode"] in ["baseline", "rfl"]


class TestBuildSafetyTimeline:
    """Test build_safety_slo_timeline function."""
    
    def test_empty_envelopes_produces_empty_timeline(self):
        """Empty envelope list produces empty timeline."""
        timeline = build_safety_slo_timeline([])
        
        assert timeline.schema_version == "1.0"
        assert len(timeline.points) == 0
        assert timeline.status_counts == {"OK": 0, "WARN": 0, "BLOCK": 0}
        assert timeline.perf_ok_rate == 0.0
        assert timeline.lint_issue_rate == 0.0
    
    def test_single_envelope_produces_single_point(self):
        """Single envelope produces one SLO point."""
        envelope: SafetyEnvelope = {
            "schema_version": "1.0",
            "config": {},
            "perf_ok": True,
            "safety_status": "OK",
            "lint_issues": [],
            "warnings": [],
            "run_id": "run_001",
            "slice_name": "slice_easy",
            "mode": "baseline",
            "timestamp": "2025-01-01T00:00:00",
        }
        
        timeline = build_safety_slo_timeline([envelope])
        
        assert len(timeline.points) == 1
        assert timeline.points[0].run_id == "run_001"
        assert timeline.points[0].safety_status == "OK"
        assert timeline.status_counts["OK"] == 1
        assert timeline.perf_ok_rate == 1.0
        assert timeline.lint_issue_rate == 0.0
    
    def test_timeline_determinism_different_order(self):
        """Same envelopes in different order produce identical timeline."""
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        envelopes_1: List[SafetyEnvelope] = [
            {
                "schema_version": "1.0",
                "config": {},
                "perf_ok": True,
                "safety_status": "OK",
                "lint_issues": [],
                "warnings": [],
                "run_id": f"run_{i:03d}",
                "slice_name": "slice_easy",
                "mode": "baseline",
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            }
            for i in range(10)
        ]
        
        # Reverse order
        envelopes_2 = list(reversed(envelopes_1))
        
        timeline_1 = build_safety_slo_timeline(envelopes_1)
        timeline_2 = build_safety_slo_timeline(envelopes_2)
        
        # Should have same points in same order
        assert len(timeline_1.points) == len(timeline_2.points)
        for p1, p2 in zip(timeline_1.points, timeline_2.points):
            assert p1.run_id == p2.run_id
            assert p1.timestamp == p2.timestamp
            assert p1.safety_status == p2.safety_status
        
        assert timeline_1.status_counts == timeline_2.status_counts
        assert timeline_1.perf_ok_rate == timeline_2.perf_ok_rate
    
    def test_timeline_sorts_by_timestamp_then_run_id(self):
        """Timeline points are sorted by timestamp, then run_id."""
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        # Create envelopes with same timestamp but different run_ids
        envelopes: List[SafetyEnvelope] = [
            {
                "schema_version": "1.0",
                "config": {},
                "perf_ok": True,
                "safety_status": "OK",
                "lint_issues": [],
                "warnings": [],
                "run_id": run_id,
                "slice_name": "slice_easy",
                "mode": "baseline",
                "timestamp": base_time.isoformat(),
            }
            for run_id in ["run_003", "run_001", "run_002"]
        ]
        
        timeline = build_safety_slo_timeline(envelopes)
        
        # Should be sorted by run_id since timestamps are equal
        assert timeline.points[0].run_id == "run_001"
        assert timeline.points[1].run_id == "run_002"
        assert timeline.points[2].run_id == "run_003"
    
    def test_timeline_computes_correct_rates(self):
        """Timeline computes correct performance and lint rates."""
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        envelopes: List[SafetyEnvelope] = [
            {
                "schema_version": "1.0",
                "config": {},
                "perf_ok": i % 2 == 0,  # 50% perf_ok
                "safety_status": "OK",
                "lint_issues": ["issue"] if i % 3 == 0 else [],  # 33% with lint issues
                "warnings": [],
                "run_id": f"run_{i:03d}",
                "slice_name": "slice_easy",
                "mode": "baseline",
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            }
            for i in range(10)
        ]
        
        timeline = build_safety_slo_timeline(envelopes)
        
        assert timeline.perf_ok_rate == 0.5  # 5/10
        assert abs(timeline.lint_issue_rate - 0.4) < 0.01  # 4/10 (0, 3, 6, 9)
    
    def test_timeline_counts_all_statuses(self):
        """Timeline counts OK/WARN/BLOCK statuses correctly."""
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        statuses: List[SafetyStatus] = ["OK", "OK", "WARN", "WARN", "WARN", "BLOCK"]
        envelopes: List[SafetyEnvelope] = [
            {
                "schema_version": "1.0",
                "config": {},
                "perf_ok": True,
                "safety_status": status,
                "lint_issues": [],
                "warnings": [],
                "run_id": f"run_{i:03d}",
                "slice_name": "slice_easy",
                "mode": "baseline",
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            }
            for i, status in enumerate(statuses)
        ]
        
        timeline = build_safety_slo_timeline(envelopes)
        
        assert timeline.status_counts["OK"] == 2
        assert timeline.status_counts["WARN"] == 3
        assert timeline.status_counts["BLOCK"] == 1


class TestBuildScenarioMatrix:
    """Test build_scenario_safety_matrix function."""
    
    def test_empty_timeline_produces_empty_matrix(self):
        """Empty timeline produces empty matrix."""
        timeline = SafetySLOTimeline(
            schema_version="1.0",
            points=[],
            status_counts={"OK": 0, "WARN": 0, "BLOCK": 0},
            perf_ok_rate=0.0,
            lint_issue_rate=0.0,
        )
        
        matrix = build_scenario_safety_matrix(timeline)
        
        assert matrix.schema_version == "1.0"
        assert len(matrix.cells) == 0
        assert matrix.total_runs == 0
        assert matrix.total_slices == 0
    
    def test_single_scenario_produces_single_cell(self):
        """Single scenario produces one cell."""
        points = [
            SafetySLOPoint(
                run_id="run_001",
                slice_name="slice_easy",
                mode="baseline",
                safety_status="OK",
                perf_ok=True,
                lint_issue_count=0,
                warnings_count=0,
                timestamp=datetime(2025, 1, 1, 0, 0, 0),
            )
        ]
        
        timeline = SafetySLOTimeline(
            schema_version="1.0",
            points=points,
            status_counts={"OK": 1, "WARN": 0, "BLOCK": 0},
            perf_ok_rate=1.0,
            lint_issue_rate=0.0,
        )
        
        matrix = build_scenario_safety_matrix(timeline)
        
        assert len(matrix.cells) == 1
        assert matrix.cells[0].slice_name == "slice_easy"
        assert matrix.cells[0].mode == "baseline"
        assert matrix.cells[0].runs == 1
        assert matrix.cells[0].ok_runs == 1
        assert matrix.cells[0].worst_status == "OK"
        assert matrix.total_slices == 1
    
    def test_multiple_scenarios_sorted_correctly(self):
        """Multiple scenarios are sorted by (slice_name, mode)."""
        points = [
            SafetySLOPoint(
                run_id=f"run_{i:03d}",
                slice_name=slice_name,
                mode=mode,
                safety_status="OK",
                perf_ok=True,
                lint_issue_count=0,
                warnings_count=0,
                timestamp=datetime(2025, 1, 1, 0, i, 0),
            )
            for i, (slice_name, mode) in enumerate([
                ("slice_hard", "rfl"),
                ("slice_easy", "baseline"),
                ("slice_hard", "baseline"),
                ("slice_easy", "rfl"),
            ])
        ]
        
        timeline = SafetySLOTimeline(
            schema_version="1.0",
            points=points,
            status_counts={"OK": 4, "WARN": 0, "BLOCK": 0},
            perf_ok_rate=1.0,
            lint_issue_rate=0.0,
        )
        
        matrix = build_scenario_safety_matrix(timeline)
        
        # Should be sorted: (easy, baseline), (easy, rfl), (hard, baseline), (hard, rfl)
        assert len(matrix.cells) == 4
        assert matrix.cells[0].slice_name == "slice_easy"
        assert matrix.cells[0].mode == "baseline"
        assert matrix.cells[1].slice_name == "slice_easy"
        assert matrix.cells[1].mode == "rfl"
        assert matrix.cells[2].slice_name == "slice_hard"
        assert matrix.cells[2].mode == "baseline"
        assert matrix.cells[3].slice_name == "slice_hard"
        assert matrix.cells[3].mode == "rfl"
        assert matrix.total_slices == 2
    
    def test_worst_status_computed_correctly(self):
        """Worst status is correctly determined (BLOCK > WARN > OK)."""
        # Scenario with mixed statuses
        points = [
            SafetySLOPoint(
                run_id=f"run_{i:03d}",
                slice_name="slice_test",
                mode="baseline",
                safety_status=status,
                perf_ok=True,
                lint_issue_count=0,
                warnings_count=0,
                timestamp=datetime(2025, 1, 1, 0, i, 0),
            )
            for i, status in enumerate(["OK", "WARN", "OK"])
        ]
        
        timeline = SafetySLOTimeline(
            schema_version="1.0",
            points=points,
            status_counts={"OK": 2, "WARN": 1, "BLOCK": 0},
            perf_ok_rate=1.0,
            lint_issue_rate=0.0,
        )
        
        matrix = build_scenario_safety_matrix(timeline)
        
        assert len(matrix.cells) == 1
        assert matrix.cells[0].worst_status == "WARN"
        
        # Add a BLOCK status
        points.append(
            SafetySLOPoint(
                run_id="run_003",
                slice_name="slice_test",
                mode="baseline",
                safety_status="BLOCK",
                perf_ok=False,
                lint_issue_count=5,
                warnings_count=2,
                timestamp=datetime(2025, 1, 1, 0, 3, 0),
            )
        )
        
        timeline = SafetySLOTimeline(
            schema_version="1.0",
            points=points,
            status_counts={"OK": 2, "WARN": 1, "BLOCK": 1},
            perf_ok_rate=0.75,
            lint_issue_rate=0.25,
        )
        
        matrix = build_scenario_safety_matrix(timeline)
        
        assert matrix.cells[0].worst_status == "BLOCK"
    
    def test_counts_aggregated_correctly(self):
        """Scenario counts are aggregated correctly."""
        points = [
            SafetySLOPoint(
                run_id=f"run_{i:03d}",
                slice_name="slice_test",
                mode="baseline",
                safety_status=["OK", "WARN", "BLOCK", "OK", "WARN"][i],
                perf_ok=i % 2 == 0,  # Runs 0, 2, 4 have perf_ok
                lint_issue_count=0,
                warnings_count=0,
                timestamp=datetime(2025, 1, 1, 0, i, 0),
            )
            for i in range(5)
        ]
        
        timeline = SafetySLOTimeline(
            schema_version="1.0",
            points=points,
            status_counts={"OK": 2, "WARN": 2, "BLOCK": 1},
            perf_ok_rate=0.6,
            lint_issue_rate=0.0,
        )
        
        matrix = build_scenario_safety_matrix(timeline)
        
        assert len(matrix.cells) == 1
        cell = matrix.cells[0]
        assert cell.runs == 5
        assert cell.ok_runs == 2
        assert cell.warn_runs == 2
        assert cell.blocked_runs == 1
        assert cell.perf_failure_runs == 2  # Runs 1, 3


class TestEvaluateSafetySLO:
    """Test evaluate_safety_slo function."""
    
    def test_empty_matrix_is_ok(self):
        """Empty matrix evaluates as OK."""
        matrix = ScenarioSafetyMatrix(
            schema_version="1.0",
            cells=[],
            total_runs=0,
            total_slices=0,
        )
        
        evaluation = evaluate_safety_slo(matrix)
        
        assert evaluation.overall_status == "OK"
        assert len(evaluation.failing_scenarios) == 0
        assert "No runs to evaluate" in evaluation.reasons[0]
    
    def test_all_ok_produces_ok_status(self):
        """All OK scenarios produce overall OK status."""
        cells = [
            ScenarioSafetyCell(
                slice_name=f"slice_{i}",
                mode="baseline",
                runs=10,
                blocked_runs=0,
                warn_runs=0,
                ok_runs=10,
                perf_failure_runs=0,
                worst_status="OK",
            )
            for i in range(3)
        ]
        
        matrix = ScenarioSafetyMatrix(
            schema_version="1.0",
            cells=cells,
            total_runs=30,
            total_slices=3,
        )
        
        evaluation = evaluate_safety_slo(matrix)
        
        assert evaluation.overall_status == "OK"
        assert len(evaluation.failing_scenarios) == 0
        assert "All safety SLO thresholds met" in evaluation.reasons[0]
    
    def test_high_warn_rate_produces_warn_status(self):
        """High global warn rate produces WARN status."""
        # 25% warn rate (exceeds 20% threshold)
        cells = [
            ScenarioSafetyCell(
                slice_name="slice_test",
                mode="baseline",
                runs=100,
                blocked_runs=0,
                warn_runs=25,
                ok_runs=75,
                perf_failure_runs=0,
                worst_status="WARN",
            )
        ]
        
        matrix = ScenarioSafetyMatrix(
            schema_version="1.0",
            cells=cells,
            total_runs=100,
            total_slices=1,
        )
        
        evaluation = evaluate_safety_slo(matrix)
        
        assert evaluation.overall_status == "WARN"
        assert "warn_rate=0.25 exceeds 0.20" in evaluation.reasons[0]
    
    def test_high_perf_failure_rate_produces_warn_status(self):
        """High global perf failure rate produces WARN status."""
        # 15% perf failure rate (exceeds 10% threshold)
        cells = [
            ScenarioSafetyCell(
                slice_name="slice_test",
                mode="baseline",
                runs=100,
                blocked_runs=0,
                warn_runs=0,
                ok_runs=100,
                perf_failure_runs=15,
                worst_status="OK",
            )
        ]
        
        matrix = ScenarioSafetyMatrix(
            schema_version="1.0",
            cells=cells,
            total_runs=100,
            total_slices=1,
        )
        
        evaluation = evaluate_safety_slo(matrix)
        
        assert evaluation.overall_status == "WARN"
        assert "perf_failure_rate=0.15 exceeds 0.10" in evaluation.reasons[0]
    
    def test_high_scenario_block_rate_produces_block_status(self):
        """High per-scenario block rate produces BLOCK status."""
        # Scenario with 10% block rate (exceeds 5% threshold)
        cells = [
            ScenarioSafetyCell(
                slice_name="slice_hard",
                mode="baseline",
                runs=100,
                blocked_runs=10,
                warn_runs=20,
                ok_runs=70,
                perf_failure_runs=5,
                worst_status="BLOCK",
            )
        ]
        
        matrix = ScenarioSafetyMatrix(
            schema_version="1.0",
            cells=cells,
            total_runs=100,
            total_slices=1,
        )
        
        evaluation = evaluate_safety_slo(matrix)
        
        assert evaluation.overall_status == "BLOCK"
        assert "slice_hard:baseline" in evaluation.failing_scenarios
        assert "block_rate=0.10 exceeds 0.05" in evaluation.reasons[0]
    
    def test_high_global_block_rate_produces_block_status(self):
        """High global block rate produces BLOCK status."""
        # Two scenarios, each with 4% block rate, but global is 8%
        cells = [
            ScenarioSafetyCell(
                slice_name=f"slice_{i}",
                mode="baseline",
                runs=50,
                blocked_runs=4,
                warn_runs=10,
                ok_runs=36,
                perf_failure_runs=2,
                worst_status="BLOCK",
            )
            for i in range(2)
        ]
        
        matrix = ScenarioSafetyMatrix(
            schema_version="1.0",
            cells=cells,
            total_runs=100,
            total_slices=2,
        )
        
        evaluation = evaluate_safety_slo(matrix)
        
        assert evaluation.overall_status == "BLOCK"
        # Should have reasons for both per-scenario and global block rates
        reason_text = " ".join(evaluation.reasons)
        assert "block_rate" in reason_text
        assert "0.08" in reason_text or "0.04" in reason_text
    
    def test_custom_thresholds(self):
        """Custom thresholds can be provided."""
        cells = [
            ScenarioSafetyCell(
                slice_name="slice_test",
                mode="baseline",
                runs=100,
                blocked_runs=7,  # 7% - would fail default but passes custom
                warn_runs=0,
                ok_runs=93,
                perf_failure_runs=0,
                worst_status="BLOCK",
            )
        ]
        
        matrix = ScenarioSafetyMatrix(
            schema_version="1.0",
            cells=cells,
            total_runs=100,
            total_slices=1,
        )
        
        # With default threshold (5%), should BLOCK
        evaluation_default = evaluate_safety_slo(matrix)
        assert evaluation_default.overall_status == "BLOCK"
        
        # With custom threshold (10%), should pass
        evaluation_custom = evaluate_safety_slo(matrix, max_block_rate=0.10)
        assert evaluation_custom.overall_status == "OK"
    
    def test_multiple_failure_reasons(self):
        """Multiple failing conditions produce multiple reasons."""
        cells = [
            ScenarioSafetyCell(
                slice_name="slice_bad",
                mode="baseline",
                runs=100,
                blocked_runs=10,  # Exceeds threshold
                warn_runs=30,  # Global warn rate will be high
                ok_runs=60,
                perf_failure_runs=20,  # Exceeds threshold
                worst_status="BLOCK",
            )
        ]
        
        matrix = ScenarioSafetyMatrix(
            schema_version="1.0",
            cells=cells,
            total_runs=100,
            total_slices=1,
        )
        
        evaluation = evaluate_safety_slo(matrix)
        
        assert evaluation.overall_status == "BLOCK"
        assert len(evaluation.reasons) >= 2
        # Should have both scenario and global block rate reasons
        reason_text = " ".join(evaluation.reasons)
        assert "slice_bad:baseline" in reason_text
        assert "block_rate" in reason_text


class TestIntegration:
    """Integration tests for end-to-end SLO pipeline."""
    
    def test_full_pipeline_ok_status(self):
        """Full pipeline: envelopes → timeline → matrix → evaluation (OK)."""
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        # Create 20 successful envelopes across 2 slices × 2 modes
        envelopes: List[SafetyEnvelope] = []
        counter = 0
        for slice_idx in range(2):
            for mode in ["baseline", "rfl"]:
                for _ in range(5):
                    envelope: SafetyEnvelope = {
                        "schema_version": "1.0",
                        "config": {},
                        "perf_ok": True,
                        "safety_status": "OK",
                        "lint_issues": [],
                        "warnings": [],
                        "run_id": f"run_{counter:03d}",
                        "slice_name": f"slice_{slice_idx}",
                        "mode": mode,
                        "timestamp": (base_time + timedelta(minutes=counter)).isoformat(),
                    }
                    envelopes.append(envelope)
                    counter += 1
        
        # Build timeline
        timeline = build_safety_slo_timeline(envelopes)
        assert len(timeline.points) == 20
        assert timeline.perf_ok_rate == 1.0
        
        # Build matrix
        matrix = build_scenario_safety_matrix(timeline)
        assert len(matrix.cells) == 4  # 2 slices × 2 modes
        assert matrix.total_runs == 20
        
        # Evaluate SLO
        evaluation = evaluate_safety_slo(matrix)
        assert evaluation.overall_status == "OK"
        assert len(evaluation.failing_scenarios) == 0
    
    def test_full_pipeline_warn_status(self):
        """Full pipeline: envelopes → timeline → matrix → evaluation (WARN)."""
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        # Create envelopes with 25% warn rate
        envelopes: List[SafetyEnvelope] = []
        for i in range(100):
            status: SafetyStatus = "WARN" if i < 25 else "OK"
            
            envelope: SafetyEnvelope = {
                "schema_version": "1.0",
                "config": {},
                "perf_ok": True,
                "safety_status": status,
                "lint_issues": [],
                "warnings": [],
                "run_id": f"run_{i:03d}",
                "slice_name": "slice_test",
                "mode": "baseline",
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            }
            envelopes.append(envelope)
        
        timeline = build_safety_slo_timeline(envelopes)
        matrix = build_scenario_safety_matrix(timeline)
        evaluation = evaluate_safety_slo(matrix)
        
        assert evaluation.overall_status == "WARN"
        assert "warn_rate=0.25" in evaluation.reasons[0]
    
    def test_full_pipeline_block_status(self):
        """Full pipeline: envelopes → timeline → matrix → evaluation (BLOCK)."""
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        # Create envelopes with 10% block rate in one scenario
        envelopes: List[SafetyEnvelope] = []
        for i in range(100):
            status: SafetyStatus = "BLOCK" if i < 10 else "OK"
            
            envelope: SafetyEnvelope = {
                "schema_version": "1.0",
                "config": {},
                "perf_ok": True,
                "safety_status": status,
                "lint_issues": [],
                "warnings": [],
                "run_id": f"run_{i:03d}",
                "slice_name": "slice_hard",
                "mode": "rfl",
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            }
            envelopes.append(envelope)
        
        timeline = build_safety_slo_timeline(envelopes)
        matrix = build_scenario_safety_matrix(timeline)
        evaluation = evaluate_safety_slo(matrix)
        
        assert evaluation.overall_status == "BLOCK"
        assert "slice_hard:rfl" in evaluation.failing_scenarios
        assert "block_rate=0.10" in evaluation.reasons[0]
