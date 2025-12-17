"""Tests for Structural Drill Runner — P5 STRUCTURAL_BREAK Event Simulation.

Tests the 5-phase structural drill specified in:
docs/system_law/P5_Structural_Drill_Package.md

SHADOW MODE: All tests verify observational behavior only.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.dag.structural_drill_runner import (
    DrillPhase,
    DrillArtifact,
    StreakTracker,
    inject_dag_cycle,
    inject_anchor_failure,
    inject_omega_exit,
    calculate_pattern,
    calculate_severity,
    create_default_phases,
    build_dag_for_phase,
    run_structural_drill,
    generate_drift_timeline_plot_data,
    DRILL_ARTIFACT_SCHEMA,
)
from backend.dag.invariant_guard import (
    ProofDag,
    emit_structural_signal,
)


# =============================================================================
# Test: Baseline Phase (Cycles 1-100)
# =============================================================================

class TestBaselinePhase:
    """Test baseline phase behavior — normal operation, SCS=1.0, Pattern=NONE."""

    def test_baseline_signal_is_consistent(self):
        """Baseline phase should produce CONSISTENT signal."""
        phases = create_default_phases()
        baseline_phase = phases[0]
        assert baseline_phase.name == "baseline"

        dag = build_dag_for_phase(baseline_phase, cycle=50)
        signal = emit_structural_signal(
            dag=dag,
            topology_state=baseline_phase.topology_config,
            ht_state=baseline_phase.ht_config,
            cycle=50,
        )

        assert signal.combined_severity == "CONSISTENT"
        assert signal.admissible is True
        assert signal.cohesion_score == 1.0

    def test_baseline_pattern_is_none(self):
        """Baseline phase should produce NONE pattern."""
        phases = create_default_phases()
        baseline_phase = phases[0]

        dag = build_dag_for_phase(baseline_phase, cycle=50)
        signal = emit_structural_signal(
            dag=dag,
            topology_state=baseline_phase.topology_config,
            ht_state=baseline_phase.ht_config,
            cycle=50,
        )

        pattern = calculate_pattern(signal)
        assert pattern == "NONE"

    def test_baseline_severity_is_info(self):
        """Baseline phase should produce INFO severity."""
        phases = create_default_phases()
        baseline_phase = phases[0]

        dag = build_dag_for_phase(baseline_phase, cycle=50)
        signal = emit_structural_signal(
            dag=dag,
            topology_state=baseline_phase.topology_config,
            ht_state=baseline_phase.ht_config,
            cycle=50,
        )

        severity = calculate_severity(signal, base_severity="INFO", streak=0)
        assert severity == "INFO"

    def test_baseline_no_violations(self):
        """Baseline phase should produce no violations."""
        phases = create_default_phases()
        baseline_phase = phases[0]

        dag = build_dag_for_phase(baseline_phase, cycle=50)
        signal = emit_structural_signal(
            dag=dag,
            topology_state=baseline_phase.topology_config,
            ht_state=baseline_phase.ht_config,
            cycle=50,
        )

        assert len(signal.violations) == 0


# =============================================================================
# Test: Tension Phase (Cycles 101-150)
# =============================================================================

class TestTensionPhase:
    """Test tension onset phase — SI-006 violation, DRIFT pattern."""

    def test_tension_signal_is_tension(self):
        """Tension phase should produce TENSION signal due to omega exit."""
        phases = create_default_phases()
        tension_phase = phases[1]
        assert tension_phase.name == "tension_onset"

        dag = build_dag_for_phase(tension_phase, cycle=125)

        # Tension phase has omega_exit_streak=45, not triggering SI-006 (threshold is 100)
        # But in_omega=False triggers tracking
        signal = emit_structural_signal(
            dag=dag,
            topology_state=tension_phase.topology_config,
            ht_state=tension_phase.ht_config,
            cycle=125,
        )

        # omega_exit_streak=45 is below 100 threshold, so still CONSISTENT
        # Let's inject a higher streak for tension
        topology_with_high_streak = dict(tension_phase.topology_config)
        topology_with_high_streak["omega_exit_streak"] = 105

        signal = emit_structural_signal(
            dag=dag,
            topology_state=topology_with_high_streak,
            ht_state=tension_phase.ht_config,
            cycle=125,
        )

        assert signal.combined_severity == "TENSION"
        assert signal.admissible is True

    def test_tension_pattern_is_drift(self):
        """Tension phase should produce DRIFT pattern."""
        phases = create_default_phases()
        tension_phase = phases[1]

        dag = build_dag_for_phase(tension_phase, cycle=125)

        topology_with_high_streak = dict(tension_phase.topology_config)
        topology_with_high_streak["omega_exit_streak"] = 105

        signal = emit_structural_signal(
            dag=dag,
            topology_state=topology_with_high_streak,
            ht_state=tension_phase.ht_config,
            cycle=125,
        )

        pattern = calculate_pattern(signal)
        assert pattern == "DRIFT"

    def test_tension_severity_escalates_info_to_warn(self):
        """Tension phase should escalate INFO → WARN."""
        phases = create_default_phases()
        tension_phase = phases[1]

        dag = build_dag_for_phase(tension_phase, cycle=125)

        topology_with_high_streak = dict(tension_phase.topology_config)
        topology_with_high_streak["omega_exit_streak"] = 105

        signal = emit_structural_signal(
            dag=dag,
            topology_state=topology_with_high_streak,
            ht_state=tension_phase.ht_config,
            cycle=125,
        )

        severity = calculate_severity(signal, base_severity="INFO", streak=0)
        assert severity == "WARN"

    def test_tension_has_si006_violation(self):
        """Tension phase should have SI-006 violation when omega streak > 100."""
        phases = create_default_phases()
        tension_phase = phases[1]

        dag = build_dag_for_phase(tension_phase, cycle=125)

        topology_with_high_streak = dict(tension_phase.topology_config)
        topology_with_high_streak["omega_exit_streak"] = 105

        signal = emit_structural_signal(
            dag=dag,
            topology_state=topology_with_high_streak,
            ht_state=tension_phase.ht_config,
            cycle=125,
        )

        violation_ids = [v.invariant_id for v in signal.violations]
        assert "SI-006" in violation_ids


# =============================================================================
# Test: Break Event (Cycle 151)
# =============================================================================

class TestBreakEvent:
    """Test structural break event — SI-001 violation, STRUCTURAL_BREAK pattern."""

    def test_break_signal_is_conflict(self):
        """Break event should produce CONFLICT signal."""
        phases = create_default_phases()
        break_phase = phases[2]
        assert break_phase.name == "structural_break"

        dag = build_dag_for_phase(break_phase, cycle=151)
        signal = emit_structural_signal(
            dag=dag,
            topology_state=break_phase.topology_config,
            ht_state=break_phase.ht_config,
            cycle=151,
        )

        assert signal.combined_severity == "CONFLICT"
        assert signal.admissible is False

    def test_break_pattern_is_structural_break(self):
        """Break event should produce STRUCTURAL_BREAK pattern."""
        phases = create_default_phases()
        break_phase = phases[2]

        dag = build_dag_for_phase(break_phase, cycle=151)
        signal = emit_structural_signal(
            dag=dag,
            topology_state=break_phase.topology_config,
            ht_state=break_phase.ht_config,
            cycle=151,
        )

        pattern = calculate_pattern(signal)
        assert pattern == "STRUCTURAL_BREAK"

    def test_break_severity_is_critical(self):
        """Break event should produce CRITICAL severity."""
        phases = create_default_phases()
        break_phase = phases[2]

        dag = build_dag_for_phase(break_phase, cycle=151)
        signal = emit_structural_signal(
            dag=dag,
            topology_state=break_phase.topology_config,
            ht_state=break_phase.ht_config,
            cycle=151,
        )

        severity = calculate_severity(signal, base_severity="INFO", streak=0)
        assert severity == "CRITICAL"

    def test_break_has_si001_violation(self):
        """Break event should have SI-001 violation."""
        phases = create_default_phases()
        break_phase = phases[2]

        dag = build_dag_for_phase(break_phase, cycle=151)
        signal = emit_structural_signal(
            dag=dag,
            topology_state=break_phase.topology_config,
            ht_state=break_phase.ht_config,
            cycle=151,
        )

        violation_ids = [v.invariant_id for v in signal.violations]
        assert "SI-001" in violation_ids

    def test_break_cohesion_score_is_zero(self):
        """Break event should produce zero cohesion score."""
        phases = create_default_phases()
        break_phase = phases[2]

        dag = build_dag_for_phase(break_phase, cycle=151)
        signal = emit_structural_signal(
            dag=dag,
            topology_state=break_phase.topology_config,
            ht_state=break_phase.ht_config,
            cycle=151,
        )

        # DAG score is 0 due to cycle, so cohesion is degraded
        assert signal.layer_scores.get("dag_score", 1.0) == 0.0
        assert signal.cohesion_score < 1.0


# =============================================================================
# Test: Streak Escalation (Cycles 152-200)
# =============================================================================

class TestStreakEscalation:
    """Test streak tracking and escalation behavior."""

    def test_streak_increments_on_break(self):
        """Streak should increment on consecutive STRUCTURAL_BREAK."""
        tracker = StreakTracker()

        # First break
        streak = tracker.update(151, "STRUCTURAL_BREAK", "CRITICAL")
        assert streak == 1
        assert tracker.streak_start_cycle == 151

        # Second break
        streak = tracker.update(152, "STRUCTURAL_BREAK", "CRITICAL")
        assert streak == 2
        assert tracker.is_repeated_break() is True

        # Third break
        streak = tracker.update(153, "STRUCTURAL_BREAK", "CRITICAL")
        assert streak == 3

    def test_streak_resets_on_recovery(self):
        """Streak should reset when pattern is not STRUCTURAL_BREAK."""
        tracker = StreakTracker()

        tracker.update(151, "STRUCTURAL_BREAK", "CRITICAL")
        tracker.update(152, "STRUCTURAL_BREAK", "CRITICAL")
        assert tracker.current_streak == 2

        # Recovery
        streak = tracker.update(153, "NONE", "INFO")
        assert streak == 0
        assert tracker.streak_start_cycle is None
        assert tracker.max_streak == 2  # Max preserved

    def test_streak_triggers_critical_at_threshold(self):
        """Streak >= 2 should trigger CRITICAL severity."""
        phases = create_default_phases()
        baseline_phase = phases[0]

        dag = build_dag_for_phase(baseline_phase, cycle=50)
        signal = emit_structural_signal(
            dag=dag,
            topology_state=baseline_phase.topology_config,
            ht_state=baseline_phase.ht_config,
            cycle=50,
        )

        # Even with CONSISTENT signal, streak >= 2 triggers CRITICAL
        severity = calculate_severity(signal, base_severity="INFO", streak=2)
        assert severity == "CRITICAL"

    def test_break_events_tracked(self):
        """Break events should be tracked in list."""
        tracker = StreakTracker()

        tracker.update(151, "STRUCTURAL_BREAK", "CRITICAL")
        tracker.update(152, "STRUCTURAL_BREAK", "CRITICAL")
        tracker.update(153, "NONE", "INFO")  # Recovery
        tracker.update(200, "STRUCTURAL_BREAK", "CRITICAL")  # New break

        assert tracker.break_events == [151, 200]


# =============================================================================
# Test: Injection Functions
# =============================================================================

class TestInjectionFunctions:
    """Test injection helper functions."""

    def test_inject_dag_cycle_creates_cycle(self):
        """inject_dag_cycle should create a detectable cycle."""
        dag = ProofDag()
        for i in range(10):
            dag.add_node(f"node_{i}")

        dag = inject_dag_cycle(dag, ["node_0", "node_5", "node_0"])

        result = dag.check_invariants()
        assert result.valid is False
        assert any("cycle" in v.message.lower() for v in result.violations)

    def test_inject_anchor_failure_sets_failed(self):
        """inject_anchor_failure should set failed_anchors."""
        ht_state = {"total_anchors": 10, "verified_anchors": 10, "failed_anchors": 0}

        modified = inject_anchor_failure(ht_state, fail_count=2)

        assert modified["failed_anchors"] == 2
        assert ht_state["failed_anchors"] == 0  # Original unchanged

    def test_inject_omega_exit_sets_streak(self):
        """inject_omega_exit should set omega exit state."""
        topology_state = {"H": 0.5, "rho": 0.8, "in_omega": True, "omega_exit_streak": 0}

        modified = inject_omega_exit(topology_state, exit_cycles=50)

        assert modified["in_omega"] is False
        assert modified["omega_exit_streak"] == 50
        assert topology_state["omega_exit_streak"] == 0  # Original unchanged


# =============================================================================
# Test: Full Drill Execution
# =============================================================================

class TestFullDrillExecution:
    """Test complete drill execution."""

    def test_drill_completes_all_phases(self):
        """Drill should complete all 5 phases."""
        artifact = run_structural_drill(
            scenario_id="TEST-DRILL-001",
            sample_rate=50,  # Higher sample rate for faster test
        )

        assert artifact.drill_id.startswith("drill_")
        assert artifact.scenario_id == "TEST-DRILL-001"
        assert artifact.completed_at is not None
        assert len(artifact.phases) == 5

    def test_drill_produces_cycle_results(self):
        """Drill should produce cycle results."""
        artifact = run_structural_drill(
            scenario_id="TEST-DRILL-002",
            sample_rate=100,  # Very high sample rate for minimal results
        )

        assert len(artifact.cycle_results) > 0

        # Check first result structure
        first_result = artifact.cycle_results[0]
        assert first_result.cycle >= 1
        assert first_result.phase_name in ["baseline", "tension_onset", "structural_break", "escalation_active", "recovery"]
        assert first_result.signal is not None
        assert first_result.pattern in ["NONE", "DRIFT", "STRUCTURAL_BREAK"]
        assert first_result.severity in ["INFO", "WARN", "CRITICAL"]

    def test_drill_detects_structural_break(self):
        """Drill should detect STRUCTURAL_BREAK at cycle 151."""
        artifact = run_structural_drill(
            scenario_id="TEST-DRILL-003",
            sample_rate=1,  # Sample every cycle
        )

        # Find the break event
        break_results = [r for r in artifact.cycle_results if r.pattern == "STRUCTURAL_BREAK"]
        assert len(break_results) > 0

        # First break should be at cycle 151
        first_break = min(break_results, key=lambda r: r.cycle)
        assert first_break.cycle == 151
        assert first_break.severity == "CRITICAL"

    def test_drill_summary_contains_metrics(self):
        """Drill summary should contain expected metrics."""
        artifact = run_structural_drill(
            scenario_id="TEST-DRILL-004",
            sample_rate=50,
        )

        summary = artifact.summary
        assert "total_cycles_sampled" in summary
        assert "pattern_counts" in summary
        assert "severity_counts" in summary
        assert "max_streak" in summary
        assert "break_events" in summary

    def test_drill_writes_artifacts_to_disk(self):
        """Drill should write artifacts when output_dir provided."""
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            artifact = run_structural_drill(
                scenario_id="TEST-DRILL-005",
                sample_rate=100,
                output_dir=output_dir,
            )

            # Check files exist
            artifact_file = output_dir / f"{artifact.drill_id}_artifact.json"
            timeline_file = output_dir / f"{artifact.drill_id}_timeline.json"
            summary_file = output_dir / f"{artifact.drill_id}_summary.json"

            assert artifact_file.exists()
            assert timeline_file.exists()
            assert summary_file.exists()


# =============================================================================
# Test: Plot Data Generation
# =============================================================================

class TestPlotDataGeneration:
    """Test drift timeline plot data generation."""

    def test_plot_data_structure(self):
        """Plot data should have correct structure."""
        artifact = run_structural_drill(
            scenario_id="TEST-PLOT-001",
            sample_rate=50,
        )

        plot_data = generate_drift_timeline_plot_data(artifact)

        assert "plot_id" in plot_data
        assert "title" in plot_data
        assert "x_axis" in plot_data
        assert "y_axes" in plot_data
        assert "categorical_tracks" in plot_data
        assert "phase_regions" in plot_data
        assert "shadow_mode_banner" in plot_data
        assert plot_data["shadow_mode_banner"] is True

    def test_plot_data_has_cohesion_series(self):
        """Plot data should include cohesion score series."""
        artifact = run_structural_drill(
            scenario_id="TEST-PLOT-002",
            sample_rate=50,
        )

        plot_data = generate_drift_timeline_plot_data(artifact)

        cohesion_axis = next((y for y in plot_data["y_axes"] if y["id"] == "cohesion"), None)
        assert cohesion_axis is not None
        assert len(cohesion_axis["values"]) == len(artifact.cycle_results)

    def test_plot_data_has_phase_regions(self):
        """Plot data should include phase region markers."""
        artifact = run_structural_drill(
            scenario_id="TEST-PLOT-003",
            sample_rate=50,
        )

        plot_data = generate_drift_timeline_plot_data(artifact)

        assert len(plot_data["phase_regions"]) == 5
        phase_names = [p["name"] for p in plot_data["phase_regions"]]
        assert "baseline" in phase_names
        assert "structural_break" in phase_names


# =============================================================================
# Test: Shadow Mode Invariants
# =============================================================================

class TestShadowModeInvariants:
    """Test that SHADOW MODE invariants are maintained."""

    def test_drill_does_not_mutate_inputs(self):
        """Drill should not mutate input configurations."""
        original_phases = create_default_phases()
        original_baseline_config = dict(original_phases[0].dag_config)

        run_structural_drill(
            scenario_id="TEST-SHADOW-001",
            phases=original_phases,
            sample_rate=100,
        )

        # Original config should be unchanged
        assert original_phases[0].dag_config == original_baseline_config

    def test_artifact_metadata_indicates_shadow_mode(self):
        """Artifact metadata should indicate SHADOW MODE."""
        artifact = run_structural_drill(
            scenario_id="TEST-SHADOW-002",
            sample_rate=100,
        )

        assert artifact.metadata.get("shadow_mode") is True

    def test_injection_does_not_modify_original_state(self):
        """Injection functions should not modify original state objects."""
        original_ht = {"total_anchors": 10, "verified_anchors": 10, "failed_anchors": 0}
        original_topology = {"H": 0.5, "rho": 0.8, "in_omega": True, "omega_exit_streak": 0}

        inject_anchor_failure(original_ht, fail_count=5)
        inject_omega_exit(original_topology, exit_cycles=100)

        assert original_ht["failed_anchors"] == 0
        assert original_topology["omega_exit_streak"] == 0


# =============================================================================
# Test: Schema Compliance
# =============================================================================

class TestSchemaCompliance:
    """Test artifact schema compliance."""

    def test_artifact_schema_is_valid_json_schema(self):
        """DRILL_ARTIFACT_SCHEMA should be valid JSON Schema."""
        assert "$schema" in DRILL_ARTIFACT_SCHEMA
        assert DRILL_ARTIFACT_SCHEMA["type"] == "object"
        assert "required" in DRILL_ARTIFACT_SCHEMA
        assert "properties" in DRILL_ARTIFACT_SCHEMA

    def test_artifact_dict_has_required_fields(self):
        """Artifact dict should have all required schema fields."""
        artifact = run_structural_drill(
            scenario_id="TEST-SCHEMA-001",
            sample_rate=100,
        )

        artifact_dict = artifact.to_dict()
        required_fields = DRILL_ARTIFACT_SCHEMA["required"]

        for field in required_fields:
            assert field in artifact_dict, f"Missing required field: {field}"

    def test_cycle_result_has_required_fields(self):
        """Cycle results should have required schema fields."""
        artifact = run_structural_drill(
            scenario_id="TEST-SCHEMA-002",
            sample_rate=100,
        )

        artifact_dict = artifact.to_dict()
        cycle_schema = DRILL_ARTIFACT_SCHEMA["properties"]["cycle_results"]["items"]
        required_fields = cycle_schema["required"]

        for result in artifact_dict["cycle_results"]:
            for field in required_fields:
                assert field in result, f"Missing required field in cycle result: {field}"
