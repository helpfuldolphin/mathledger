"""
Integration tests for P5 divergence diagnostic harness emission.

Tests that:
1. Running harness with --emit-p5-diagnostic produces p5_divergence_diagnostic.json
2. The diagnostic file is JSON-safe
3. Evidence pack builder auto-detects and includes the diagnostic

SHADOW MODE CONTRACT:
- These tests validate observational behavior only
- No actual governance decisions are influenced
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_divergence_snapshot() -> Dict[str, Any]:
    """Mock divergence snapshot for testing."""
    return {
        "cycle": 50,
        "severity": "NONE",
        "type": "NONE",
        "divergence_pct": 0.02,
        "H_diff": 0.01,
        "rho_diff": 0.005,
    }


@pytest.fixture
def mock_result():
    """Mock P4 runner result object."""
    class MockResult:
        divergence_rate = 0.05
        total_divergences = 2
        cycles_completed = 50
        u2_success_rate_final = 0.92

    return MockResult()


@pytest.fixture
def temp_output_dir():
    """Temporary output directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Test: P5 Diagnostic File Generation
# =============================================================================


class TestP5DiagnosticFileGeneration:
    """Test that P5 diagnostic file is correctly generated."""

    def test_interpret_p5_divergence_produces_valid_json(self, mock_divergence_snapshot):
        """interpret_p5_divergence produces JSON-serializable output."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        replay_signal = {
            "status": "OK",
            "governance_alignment": "aligned",
            "conflict": False,
            "reasons": ["[Safety] All checks passed"],
        }
        topology_signal = {"mode": "STABLE", "persistence_drift": 0.02}
        budget_signal = {"stability_class": "STABLE", "health_score": 90}

        diagnostic = interpret_p5_divergence(
            divergence_snapshot=mock_divergence_snapshot,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
            cycle=50,
            run_id="test_run_001",
        )

        # Should be JSON serializable
        json_str = json.dumps(diagnostic)
        assert isinstance(json_str, str)

        # Round trip should work
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0.0"
        assert parsed["run_id"] == "test_run_001"

    def test_diagnostic_written_to_file(self, mock_divergence_snapshot, temp_output_dir):
        """Diagnostic can be written to file."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        replay_signal = {"status": "OK", "governance_alignment": "aligned", "conflict": False}
        topology_signal = {"mode": "STABLE"}
        budget_signal = {"stability_class": "STABLE"}

        diagnostic = interpret_p5_divergence(
            divergence_snapshot=mock_divergence_snapshot,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
            cycle=50,
        )

        # Write to file
        diag_path = temp_output_dir / "p5_divergence_diagnostic.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostic, f, indent=2)

        # Verify file exists and is valid JSON
        assert diag_path.exists()

        with open(diag_path, "r") as f:
            loaded = json.load(f)

        assert loaded["root_cause_hypothesis"] == "NOMINAL"


# =============================================================================
# Test: Evidence Pack Auto-Detection
# =============================================================================


class TestEvidencePackAutoDetection:
    """Test that evidence pack builder auto-detects p5_divergence_diagnostic.json."""

    def test_detect_p5_diagnostic_file_when_present(self, mock_divergence_snapshot, temp_output_dir):
        """detect_p5_diagnostic_file detects file when present."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence
        from backend.topology.first_light.evidence_pack import detect_p5_diagnostic_file

        # Create diagnostic
        replay_signal = {"status": "WARN", "governance_alignment": "tension", "conflict": False}
        topology_signal = {"mode": "STABLE"}
        budget_signal = {"stability_class": "STABLE"}

        diagnostic = interpret_p5_divergence(
            divergence_snapshot=mock_divergence_snapshot,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
            cycle=50,
            run_id="test_detect_001",
        )

        # Write diagnostic file
        diag_path = temp_output_dir / "p5_divergence_diagnostic.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostic, f, indent=2)

        # Detect
        ref = detect_p5_diagnostic_file(temp_output_dir)

        assert ref is not None
        assert ref.schema_version == "1.0.0"
        assert ref.root_cause_hypothesis == "REPLAY_FAILURE"
        assert ref.run_id == "test_detect_001"

    def test_detect_p5_diagnostic_file_when_absent(self, temp_output_dir):
        """detect_p5_diagnostic_file returns None when file absent."""
        from backend.topology.first_light.evidence_pack import detect_p5_diagnostic_file

        ref = detect_p5_diagnostic_file(temp_output_dir)

        assert ref is None

    def test_detect_p5_diagnostic_file_in_p4_shadow_subdir(
        self, mock_divergence_snapshot, temp_output_dir
    ):
        """detect_p5_diagnostic_file finds file in p4_shadow subdirectory."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence
        from backend.topology.first_light.evidence_pack import detect_p5_diagnostic_file

        # Create p4_shadow subdirectory
        p4_shadow = temp_output_dir / "p4_shadow"
        p4_shadow.mkdir()

        # Create diagnostic
        replay_signal = {"status": "OK", "governance_alignment": "aligned", "conflict": False}
        topology_signal = {"mode": "STABLE"}
        budget_signal = {"stability_class": "STABLE"}

        diagnostic = interpret_p5_divergence(
            divergence_snapshot=mock_divergence_snapshot,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
            cycle=100,
        )

        # Write to p4_shadow subdirectory
        diag_path = p4_shadow / "p5_divergence_diagnostic.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostic, f, indent=2)

        # Detect from parent
        ref = detect_p5_diagnostic_file(temp_output_dir)

        assert ref is not None
        assert ref.path == "p4_shadow/p5_divergence_diagnostic.json"


# =============================================================================
# Test: GGFL Adapter
# =============================================================================


class TestGGFLAdapter:
    """Test GGFL adapter function."""

    def test_p5_diagnostic_for_alignment_view_healthy(self, mock_divergence_snapshot):
        """p5_diagnostic_for_alignment_view produces GGFL-compatible output."""
        from backend.health.p5_divergence_interpreter import (
            interpret_p5_divergence,
            p5_diagnostic_for_alignment_view,
        )

        # Create healthy diagnostic
        replay_signal = {"status": "OK", "governance_alignment": "aligned", "conflict": False}
        topology_signal = {"mode": "STABLE"}
        budget_signal = {"stability_class": "STABLE"}

        diagnostic = interpret_p5_divergence(
            divergence_snapshot=mock_divergence_snapshot,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
        )

        ggfl_signal = p5_diagnostic_for_alignment_view(diagnostic)

        assert ggfl_signal["signal_type"] == "p5_diagnostic"
        assert ggfl_signal["status"] == "healthy"
        assert ggfl_signal["hypothesis"] == "NOMINAL"
        assert ggfl_signal["severity"] == "INFO"
        assert ggfl_signal["advisory_only"] is True
        assert ggfl_signal["shadow_mode"] is True
        assert ggfl_signal["weight"] == 0.1

    def test_p5_diagnostic_for_alignment_view_unhealthy(self):
        """Unhealthy diagnostic produces degraded status."""
        from backend.health.p5_divergence_interpreter import (
            interpret_p5_divergence,
            p5_diagnostic_for_alignment_view,
        )

        # Create unhealthy diagnostic scenario
        divergence_snapshot = {"severity": "WARN", "type": "STATE", "divergence_pct": 0.15}
        replay_signal = {"status": "OK", "governance_alignment": "aligned", "conflict": False}
        topology_signal = {"mode": "TURBULENT"}  # Triggers STRUCTURAL_BREAK
        budget_signal = {"stability_class": "STABLE"}

        diagnostic = interpret_p5_divergence(
            divergence_snapshot=divergence_snapshot,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
        )

        ggfl_signal = p5_diagnostic_for_alignment_view(diagnostic)

        assert ggfl_signal["status"] == "degraded"
        assert ggfl_signal["hypothesis"] == "STRUCTURAL_BREAK"
        assert ggfl_signal["severity"] == "WARN"

    def test_p5_diagnostic_for_alignment_view_critical(self):
        """Critical diagnostic produces unhealthy status."""
        from backend.health.p5_divergence_interpreter import (
            interpret_p5_divergence,
            p5_diagnostic_for_alignment_view,
        )

        # Create critical diagnostic scenario - identity violation
        divergence_snapshot = {"severity": "NONE", "type": "NONE"}
        replay_signal = {"status": "OK", "governance_alignment": "aligned", "conflict": False}
        topology_signal = {"mode": "STABLE"}
        budget_signal = {"stability_class": "STABLE"}
        identity_signal = {"block_hash_valid": False}  # Triggers IDENTITY_VIOLATION

        diagnostic = interpret_p5_divergence(
            divergence_snapshot=divergence_snapshot,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
            identity_signal=identity_signal,
        )

        ggfl_signal = p5_diagnostic_for_alignment_view(diagnostic)

        assert ggfl_signal["status"] == "unhealthy"
        assert ggfl_signal["hypothesis"] == "IDENTITY_VIOLATION"
        assert ggfl_signal["severity"] == "CRITICAL"


# =============================================================================
# Test: 50-Cycle Mock Run Simulation
# =============================================================================


class TestMockHarnessRun:
    """Simulate a 50-cycle harness run with mock signals."""

    def test_50_cycle_mock_run_produces_diagnostic(self, temp_output_dir):
        """Simulated 50-cycle run produces valid diagnostic file."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        # Simulate 50 cycles of divergence snapshots
        divergence_snapshots = []
        for cycle in range(50):
            # Vary divergence slightly
            pct = 0.02 + (cycle % 10) * 0.001
            snapshot = {
                "cycle": cycle,
                "severity": "NONE" if pct < 0.05 else "INFO",
                "type": "NONE",
                "divergence_pct": pct,
            }
            divergence_snapshots.append(snapshot)

        # Use last snapshot for diagnostic
        final_snapshot = divergence_snapshots[-1]

        # Build signals based on final state
        replay_signal = {
            "status": "OK",
            "governance_alignment": "aligned",
            "conflict": False,
            "reasons": ["[Safety] 50 cycles completed successfully"],
        }
        topology_signal = {"mode": "STABLE", "persistence_drift": 0.015}
        budget_signal = {"stability_class": "STABLE", "health_score": 92}

        # Generate diagnostic
        diagnostic = interpret_p5_divergence(
            divergence_snapshot=final_snapshot,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
            cycle=50,
            run_id="test_50_cycle_run",
        )

        # Write to file
        diag_path = temp_output_dir / "p5_divergence_diagnostic.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostic, f, indent=2)

        # Verify
        assert diag_path.exists()

        with open(diag_path, "r") as f:
            loaded = json.load(f)

        assert loaded["cycle"] == 50
        assert loaded["run_id"] == "test_50_cycle_run"
        assert loaded["root_cause_hypothesis"] == "NOMINAL"

    def test_50_cycle_mock_run_with_detection(self, temp_output_dir):
        """Diagnostic from 50-cycle run is detected by evidence pack."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence
        from backend.topology.first_light.evidence_pack import detect_p5_diagnostic_file

        # Create divergence log
        divergence_log_path = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log_path, "w") as f:
            for cycle in range(50):
                entry = {"cycle": cycle, "severity": "NONE", "type": "NONE", "divergence_pct": 0.02}
                f.write(json.dumps(entry) + "\n")

        # Read last entry
        with open(divergence_log_path, "r") as f:
            lines = f.readlines()
            final_snapshot = json.loads(lines[-1])

        # Generate diagnostic
        replay_signal = {"status": "OK", "governance_alignment": "aligned", "conflict": False}
        topology_signal = {"mode": "STABLE"}
        budget_signal = {"stability_class": "STABLE"}

        diagnostic = interpret_p5_divergence(
            divergence_snapshot=final_snapshot,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
            cycle=50,
            run_id="test_50_cycle_detect",
        )

        # Write diagnostic
        diag_path = temp_output_dir / "p5_divergence_diagnostic.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostic, f, indent=2)

        # Detect via evidence pack
        ref = detect_p5_diagnostic_file(temp_output_dir)

        assert ref is not None
        assert ref.cycle == 50
        assert ref.run_id == "test_50_cycle_detect"
        assert ref.root_cause_hypothesis == "NOMINAL"


# =============================================================================
# Test: JSON Safety
# =============================================================================


class TestJSONSafety:
    """Test that all outputs are JSON-safe."""

    def test_all_hypothesis_types_json_safe(self):
        """All hypothesis types produce JSON-safe output."""
        from backend.health.p5_divergence_interpreter import (
            interpret_p5_divergence,
            p5_diagnostic_for_alignment_view,
        )

        test_cases = [
            # NOMINAL
            {
                "divergence": {"severity": "NONE", "type": "NONE"},
                "replay": {"status": "OK", "governance_alignment": "aligned", "conflict": False},
                "topology": {"mode": "STABLE"},
                "budget": {"stability_class": "STABLE"},
                "identity": None,
            },
            # REPLAY_FAILURE
            {
                "divergence": {"severity": "NONE", "type": "NONE"},
                "replay": {"status": "WARN", "governance_alignment": "tension", "conflict": False},
                "topology": {"mode": "STABLE"},
                "budget": {"stability_class": "STABLE"},
                "identity": None,
            },
            # STRUCTURAL_BREAK
            {
                "divergence": {"severity": "WARN", "type": "STATE"},
                "replay": {"status": "OK", "governance_alignment": "aligned", "conflict": False},
                "topology": {"mode": "TURBULENT"},
                "budget": {"stability_class": "STABLE"},
                "identity": None,
            },
            # IDENTITY_VIOLATION
            {
                "divergence": {"severity": "NONE", "type": "NONE"},
                "replay": {"status": "OK", "governance_alignment": "aligned", "conflict": False},
                "topology": {"mode": "STABLE"},
                "budget": {"stability_class": "STABLE"},
                "identity": {"block_hash_valid": False},
            },
        ]

        for i, tc in enumerate(test_cases):
            kwargs = {
                "divergence_snapshot": tc["divergence"],
                "replay_signal": tc["replay"],
                "topology_signal": tc["topology"],
                "budget_signal": tc["budget"],
            }
            if tc["identity"]:
                kwargs["identity_signal"] = tc["identity"]

            diagnostic = interpret_p5_divergence(**kwargs)

            # Must be JSON serializable
            json_str = json.dumps(diagnostic)
            assert isinstance(json_str, str), f"Test case {i} failed to serialize"

            # GGFL adapter must also be JSON serializable
            ggfl = p5_diagnostic_for_alignment_view(diagnostic)
            ggfl_str = json.dumps(ggfl)
            assert isinstance(ggfl_str, str), f"GGFL for test case {i} failed to serialize"


# =============================================================================
# Test: Real Signal Consumption
# =============================================================================


class TestRealSignalConsumption:
    """Test that save_p5_diagnostic loads real signals from artifacts."""

    def test_all_real_signals_no_stub_markers(self, temp_output_dir, mock_result):
        """With all real signals present, no stub markers appear."""
        # Create all signal artifact files
        replay_signal = {
            "status": "OK",
            "governance_alignment": "aligned",
            "conflict": False,
            "reasons": ["[Safety] All checks passed"],
        }
        with open(temp_output_dir / "replay_safety_governance_signal.json", "w") as f:
            json.dump(replay_signal, f)

        topology_signal = {
            "p5_summary": {"joint_status": "ALIGNED"},
            "persistence_drift": 0.01,
        }
        with open(temp_output_dir / "p5_topology_auditor_report.json", "w") as f:
            json.dump(topology_signal, f)

        budget_signal = {"stability_class": "STABLE", "health_score": 95}
        with open(temp_output_dir / "budget_calibration_summary.json", "w") as f:
            json.dump(budget_signal, f)

        identity_signal = {
            "block_hash_valid": True,
            "merkle_root_valid": True,
            "chain_continuous": True,
        }
        with open(temp_output_dir / "identity_check.json", "w") as f:
            json.dump(identity_signal, f)

        structure_signal = {"dag_coherent": True, "proof_count": 100}
        with open(temp_output_dir / "dag_coherence_check.json", "w") as f:
            json.dump(structure_signal, f)

        # Create divergence log
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            for i in range(10):
                f.write(json.dumps({"cycle": i, "severity": "NONE", "type": "NONE"}) + "\n")

        # Import and call save_p5_diagnostic
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_all_real",
            cycle_count=10,
        )

        # Load and verify diagnostic
        diag_path = temp_output_dir / "p5_divergence_diagnostic.json"
        assert diag_path.exists()

        with open(diag_path, "r") as f:
            diagnostic = json.load(f)

        # Check signal_inputs
        signal_inputs = diagnostic["signal_inputs"]
        source_summary = signal_inputs["source_summary"]

        # All should be "real" except none (identity and structure may be none if not found)
        assert source_summary["divergence"] == "real"
        assert source_summary["replay"] == "real"
        assert source_summary["topology"] == "real"
        assert source_summary["budget"] == "real"
        assert source_summary["identity"] == "real"
        assert source_summary["structure"] == "real"

        # No stub markers
        assert signal_inputs["stub_count"] == 0

    def test_partial_real_signals_stub_markers_for_missing(self, temp_output_dir, mock_result):
        """With some signals missing, only missing get stub markers."""
        # Create only replay and topology signals
        replay_signal = {
            "status": "WARN",
            "governance_alignment": "tension",
            "conflict": False,
        }
        with open(temp_output_dir / "replay_safety_governance_signal.json", "w") as f:
            json.dump(replay_signal, f)

        topology_signal = {"mode": "DRIFT", "persistence_drift": 0.05}
        with open(temp_output_dir / "topology_bundle.json", "w") as f:
            json.dump(topology_signal, f)

        # Create divergence log
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "INFO", "type": "STATE"}) + "\n")

        # No budget, identity, or structure signals

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_partial_real",
            cycle_count=1,
        )

        # Load and verify
        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        source_summary = diagnostic["signal_inputs"]["source_summary"]

        # Real signals
        assert source_summary["divergence"] == "real"
        assert source_summary["replay"] == "real"
        assert source_summary["topology"] == "real"

        # Stub signals
        assert source_summary["budget"] == "stub"

        # None signals (optional)
        assert source_summary["identity"] == "none"
        assert source_summary["structure"] == "none"

        # Counts
        assert diagnostic["signal_inputs"]["real_count"] == 3
        assert diagnostic["signal_inputs"]["stub_count"] == 1
        assert diagnostic["signal_inputs"]["none_count"] == 2

    def test_no_real_signals_all_stubs(self, temp_output_dir, mock_result):
        """With no real signals, all get stub markers."""
        # Create only divergence log (minimal required)
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_no_real",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        source_summary = diagnostic["signal_inputs"]["source_summary"]

        # Divergence is real (from log)
        assert source_summary["divergence"] == "real"

        # All others are stub or none
        assert source_summary["replay"] == "stub"
        assert source_summary["topology"] == "stub"
        assert source_summary["budget"] == "stub"
        assert source_summary["identity"] == "none"
        assert source_summary["structure"] == "none"

    def test_determinism_same_artifacts_identical_output(self, temp_output_dir, mock_result):
        """Same artifacts produce identical diagnostic JSON."""
        # Create artifacts
        replay_signal = {"status": "OK", "governance_alignment": "aligned", "conflict": False}
        with open(temp_output_dir / "replay_safety_governance_signal.json", "w") as f:
            json.dump(replay_signal, f)

        topology_signal = {"mode": "STABLE", "persistence_drift": 0.02}
        with open(temp_output_dir / "topology_bundle.json", "w") as f:
            json.dump(topology_signal, f)

        budget_signal = {"stability_class": "STABLE", "health_score": 90}
        with open(temp_output_dir / "budget_calibration_summary.json", "w") as f:
            json.dump(budget_signal, f)

        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE", "divergence_pct": 0.01}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        # First run
        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="determinism_test",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic1 = json.load(f)

        # Second run with same inputs
        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="determinism_test",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic2 = json.load(f)

        # Remove timestamp for comparison
        del diagnostic1["timestamp"]
        del diagnostic2["timestamp"]

        assert diagnostic1 == diagnostic2

    def test_p4_shadow_subdirectory_signals(self, temp_output_dir, mock_result):
        """Signals in p4_shadow subdirectory are detected."""
        # Create p4_shadow subdirectory
        p4_shadow = temp_output_dir / "p4_shadow"
        p4_shadow.mkdir()

        # Create signals in p4_shadow
        replay_signal = {"status": "BLOCK", "governance_alignment": "conflict", "conflict": True}
        with open(p4_shadow / "replay_safety_governance_signal.json", "w") as f:
            json.dump(replay_signal, f)

        budget_signal = {"stability_class": "VOLATILE", "health_score": 50}
        with open(p4_shadow / "budget_calibration_summary.json", "w") as f:
            json.dump(budget_signal, f)

        # Divergence log in main dir
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "WARN", "type": "STATE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_p4_shadow",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        source_summary = diagnostic["signal_inputs"]["source_summary"]

        # p4_shadow signals should be detected as real
        assert source_summary["replay"] == "real"
        assert source_summary["budget"] == "real"

    def test_real_signal_source_marker_in_loaded_data(self, temp_output_dir, mock_result):
        """Loaded real signals contain source='real' marker."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _load_replay_signal, _load_topology_signal, _load_budget_signal

        # Create real signal file
        replay_signal = {"status": "OK", "governance_alignment": "aligned", "conflict": False}
        with open(temp_output_dir / "replay_safety_governance_signal.json", "w") as f:
            json.dump(replay_signal, f)

        # Load and check source marker (now returns 5-tuple with paths)
        loaded_signal, source, reason, paths_tried, selected_path = _load_replay_signal(temp_output_dir, mock_result)

        assert source == "real"
        assert loaded_signal["source"] == "real"
        assert loaded_signal["status"] == "OK"
        assert reason is None  # No reason for real signal
        assert paths_tried is not None
        assert selected_path == "replay_safety_governance_signal.json"

    def test_stub_signal_source_marker(self, temp_output_dir, mock_result):
        """Stub signals contain source='stub' marker."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _load_replay_signal, _load_topology_signal, _load_budget_signal

        # No signal files created - should return stub

        loaded_signal, source, reason, paths_tried, selected_path = _load_replay_signal(temp_output_dir, mock_result)

        assert source == "stub"
        assert loaded_signal["source"] == "stub"
        assert reason == "file not found"
        assert paths_tried is not None
        assert selected_path is None

        # Topology stub
        topo_signal, topo_source, topo_reason, topo_paths, topo_selected = _load_topology_signal(temp_output_dir)
        assert topo_source == "stub"
        assert topo_signal["source"] == "stub"
        assert topo_reason == "file not found"
        assert topo_paths is not None
        assert topo_selected is None

        # Budget stub
        budget_signal, budget_source, budget_reason, budget_paths, budget_selected = _load_budget_signal(temp_output_dir)
        assert budget_source == "stub"
        assert budget_signal["source"] == "stub"
        assert budget_reason == "file not found"
        assert budget_paths is not None
        assert budget_selected is None

    def test_identity_signal_from_run_config(self, temp_output_dir, mock_result):
        """Identity signal extracted from run_config.json identity_preflight."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _load_identity_signal

        # Create run_config with identity_preflight
        run_config = {
            "run_id": "test_run",
            "identity_preflight": {
                "status": "PASSED",
                "checks": ["hash", "merkle", "signature"],
            },
        }
        with open(temp_output_dir / "run_config.json", "w") as f:
            json.dump(run_config, f)

        loaded_signal, source, reason, paths_tried, selected_path = _load_identity_signal(temp_output_dir)

        assert source == "real"
        assert loaded_signal is not None
        assert loaded_signal["source"] == "real"
        assert loaded_signal["block_hash_valid"] is True
        assert reason is None  # No reason for real signal
        assert paths_tried is not None
        assert selected_path == "run_config.json"

    def test_signal_inputs_counts_correct(self, temp_output_dir, mock_result):
        """signal_inputs counts are correct for mixed sources."""
        # Create some signals
        with open(temp_output_dir / "replay_safety_governance_signal.json", "w") as f:
            json.dump({"status": "OK", "governance_alignment": "aligned", "conflict": False}, f)

        with open(temp_output_dir / "identity_check.json", "w") as f:
            json.dump({"block_hash_valid": True}, f)

        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_counts",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        signal_inputs = diagnostic["signal_inputs"]

        # Expected: divergence=real, replay=real, identity=real, topology=stub, budget=stub, structure=none
        assert signal_inputs["real_count"] == 3
        assert signal_inputs["stub_count"] == 2
        assert signal_inputs["none_count"] == 1

    def test_malformed_signal_file_falls_back_to_stub(self, temp_output_dir, mock_result):
        """Malformed JSON signal file falls back to stub."""
        # Create malformed signal file
        with open(temp_output_dir / "replay_safety_governance_signal.json", "w") as f:
            f.write("{ invalid json }")

        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        # Should not raise - falls back to stub
        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_malformed",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        # Replay should be stub due to malformed file
        assert diagnostic["signal_inputs"]["source_summary"]["replay"] == "stub"


# =============================================================================
# Test: Canonical Source Taxonomy (Prompt 1)
# =============================================================================


class TestCanonicalSourceTaxonomy:
    """Test that source labels are canonicalized to {real, stub, none}."""

    def test_validate_source_label_canonical_values(self):
        """Canonical values pass through unchanged."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _validate_source_label

        # Real
        source, advisory = _validate_source_label("real", "test_signal")
        assert source == "real"
        assert advisory is None

        # Stub
        source, advisory = _validate_source_label("stub", "test_signal")
        assert source == "stub"
        assert advisory is None

        # None
        source, advisory = _validate_source_label("none", "test_signal")
        assert source == "none"
        assert advisory is None

    def test_validate_source_label_unknown_coerced_to_stub(self):
        """Unknown source labels are coerced to stub with advisory."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _validate_source_label

        # Unknown value
        source, advisory = _validate_source_label("unknown", "replay")
        assert source == "stub"
        assert advisory is not None
        assert "[Advisory]" in advisory
        assert "unknown" in advisory
        assert "replay" in advisory
        assert "coerced to 'stub'" in advisory

        # Another unknown value
        source, advisory = _validate_source_label("REAL", "topology")  # Case matters
        assert source == "stub"
        assert advisory is not None

    def test_build_signal_inputs_coerces_unknown_sources(self):
        """_build_signal_inputs coerces unknown sources and adds advisories."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _build_signal_inputs

        source_summary = {
            "divergence": "real",
            "replay": "loaded",  # Unknown - should be coerced
            "topology": "stub",
            "budget": "NONE",  # Case wrong - should be coerced
            "identity": "none",
            "structure": "none",
        }
        missing_reasons = {
            "replay": "file not found",
            "topology": "file not found",
            "budget": "file not found",
        }

        result = _build_signal_inputs(source_summary, missing_reasons)

        # Unknown sources should be coerced to stub
        assert result["source_summary"]["replay"] == "stub"
        assert result["source_summary"]["budget"] == "stub"

        # Canonical values unchanged
        assert result["source_summary"]["divergence"] == "real"
        assert result["source_summary"]["topology"] == "stub"
        assert result["source_summary"]["identity"] == "none"

        # Advisories should be present
        assert "source_coercion_advisories" in result
        assert len(result["source_coercion_advisories"]) == 2

    def test_canonical_labels_frozen_set(self):
        """CANONICAL_SOURCE_LABELS is a frozen set with exactly 3 values."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import CANONICAL_SOURCE_LABELS

        assert isinstance(CANONICAL_SOURCE_LABELS, frozenset)
        assert CANONICAL_SOURCE_LABELS == {"real", "stub", "none"}
        assert len(CANONICAL_SOURCE_LABELS) == 3


# =============================================================================
# Test: Missing Required Artifacts Explainability (Prompt 2)
# =============================================================================


class TestMissingArtifactsExplainability:
    """Test the missing_required_artifacts explainability field."""

    def test_missing_artifacts_lists_stub_and_none_signals(self, temp_output_dir, mock_result):
        """missing_required_artifacts lists signals that are stub or none."""
        # Only create divergence log - all others missing
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_missing",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        missing_artifacts = diagnostic["signal_inputs"]["missing_required_artifacts"]

        # Should have entries for stub/none signals
        assert len(missing_artifacts) > 0

        # Check structure
        for entry in missing_artifacts:
            assert "signal" in entry
            assert "source" in entry
            assert "reason" in entry
            assert entry["source"] in ("stub", "none")

        # Should not include divergence (which is real)
        signal_names = [e["signal"] for e in missing_artifacts]
        assert "divergence" not in signal_names

    def test_missing_artifacts_reason_file_not_found(self, temp_output_dir, mock_result):
        """Reason is 'file not found' when artifact doesn't exist."""
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_reason_not_found",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        missing_artifacts = diagnostic["signal_inputs"]["missing_required_artifacts"]

        # Find replay entry
        replay_entry = next((e for e in missing_artifacts if e["signal"] == "replay"), None)
        assert replay_entry is not None
        assert replay_entry["reason"] == "file not found"

    def test_missing_artifacts_reason_malformed_json(self, temp_output_dir, mock_result):
        """Reason is 'malformed JSON' when artifact has invalid JSON."""
        # Create malformed files
        with open(temp_output_dir / "replay_safety_governance_signal.json", "w") as f:
            f.write("{ not valid json }")

        with open(temp_output_dir / "topology_bundle.json", "w") as f:
            f.write("{ also broken }")

        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_reason_malformed",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        missing_artifacts = diagnostic["signal_inputs"]["missing_required_artifacts"]

        # Check replay has malformed reason
        replay_entry = next((e for e in missing_artifacts if e["signal"] == "replay"), None)
        assert replay_entry is not None
        assert replay_entry["reason"] == "malformed JSON"

        # Check topology has malformed reason
        topology_entry = next((e for e in missing_artifacts if e["signal"] == "topology"), None)
        assert topology_entry is not None
        assert topology_entry["reason"] == "malformed JSON"

    def test_missing_artifacts_limited_to_6_entries(self, temp_output_dir, mock_result):
        """missing_required_artifacts is limited to top 6 entries."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _build_signal_inputs

        # Create more than 6 missing signals
        source_summary = {
            "a_signal": "stub",
            "b_signal": "stub",
            "c_signal": "none",
            "d_signal": "stub",
            "e_signal": "none",
            "f_signal": "stub",
            "g_signal": "stub",  # 7th
            "h_signal": "none",  # 8th
        }
        missing_reasons = {
            "a_signal": "file not found",
            "b_signal": "file not found",
            "c_signal": "file not found (optional)",
            "d_signal": "malformed JSON",
            "e_signal": "missing required keys",
            "f_signal": "file not found",
            "g_signal": "file not found",
            "h_signal": "file not found (optional)",
        }

        result = _build_signal_inputs(source_summary, missing_reasons)

        # Should be limited to 6
        assert len(result["missing_required_artifacts"]) == 6

    def test_missing_artifacts_deterministic_ordering(self, temp_output_dir, mock_result):
        """missing_required_artifacts has deterministic alphabetical ordering."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _build_signal_inputs

        source_summary = {
            "zebra": "stub",
            "alpha": "stub",
            "beta": "none",
            "gamma": "stub",
        }
        missing_reasons = {
            "zebra": "file not found",
            "alpha": "file not found",
            "beta": "file not found (optional)",
            "gamma": "malformed JSON",
        }

        result = _build_signal_inputs(source_summary, missing_reasons)
        missing = result["missing_required_artifacts"]

        # Should be alphabetically ordered
        signal_names = [e["signal"] for e in missing]
        assert signal_names == sorted(signal_names)
        assert signal_names == ["alpha", "beta", "gamma", "zebra"]

    def test_missing_artifacts_empty_when_all_real(self, temp_output_dir, mock_result):
        """missing_required_artifacts is empty when all signals are real."""
        # Create all signal files
        with open(temp_output_dir / "replay_safety_governance_signal.json", "w") as f:
            json.dump({"status": "OK", "governance_alignment": "aligned", "conflict": False}, f)

        with open(temp_output_dir / "p5_topology_auditor_report.json", "w") as f:
            json.dump({"p5_summary": {"joint_status": "ALIGNED"}}, f)

        with open(temp_output_dir / "budget_calibration_summary.json", "w") as f:
            json.dump({"stability_class": "STABLE", "health_score": 90}, f)

        with open(temp_output_dir / "identity_check.json", "w") as f:
            json.dump({"block_hash_valid": True}, f)

        with open(temp_output_dir / "dag_coherence_check.json", "w") as f:
            json.dump({"dag_coherent": True}, f)

        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_all_real",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        # All real - no missing artifacts
        assert diagnostic["signal_inputs"]["missing_required_artifacts"] == []


# =============================================================================
# Test: Reconciliation Mode (Metric Definitions + Paths + Integrity)
# =============================================================================


class TestReconciliationMode:
    """Test reconciliation mode features: metric definitions version, paths, diagnostic integrity."""

    def test_metric_definitions_version_present(self, temp_output_dir, mock_result):
        """metric_definitions_version field is present in signal_inputs."""
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic, METRIC_DEFINITIONS_VERSION

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_metric_version",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        assert "metric_definitions_version" in diagnostic["signal_inputs"]
        assert diagnostic["signal_inputs"]["metric_definitions_version"] == METRIC_DEFINITIONS_VERSION
        assert "METRIC_DEFINITIONS.md" in diagnostic["signal_inputs"]["metric_definitions_version"]

    def test_expected_paths_tried_in_missing_artifacts(self, temp_output_dir, mock_result):
        """missing_required_artifacts entries include expected_paths_tried."""
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_paths_tried",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        missing_artifacts = diagnostic["signal_inputs"]["missing_required_artifacts"]

        # Find replay entry (should be stub)
        replay_entry = next((e for e in missing_artifacts if e["signal"] == "replay"), None)
        assert replay_entry is not None
        assert "expected_paths_tried" in replay_entry
        assert len(replay_entry["expected_paths_tried"]) <= 3
        assert "replay_safety_governance_signal.json" in replay_entry["expected_paths_tried"]

    def test_selected_paths_for_real_signals(self, temp_output_dir, mock_result):
        """selected_paths is present for real signals."""
        # Create real signal files
        with open(temp_output_dir / "replay_safety_governance_signal.json", "w") as f:
            json.dump({"status": "OK", "governance_alignment": "aligned", "conflict": False}, f)

        with open(temp_output_dir / "topology_bundle.json", "w") as f:
            json.dump({"mode": "STABLE"}, f)

        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_selected_paths",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        signal_inputs = diagnostic["signal_inputs"]

        # Should have selected_paths for real signals
        assert "selected_paths" in signal_inputs
        assert "divergence" in signal_inputs["selected_paths"]
        assert "replay" in signal_inputs["selected_paths"]
        assert "topology" in signal_inputs["selected_paths"]
        assert signal_inputs["selected_paths"]["replay"] == "replay_safety_governance_signal.json"
        assert signal_inputs["selected_paths"]["topology"] == "topology_bundle.json"

    def test_diagnostic_integrity_block_present(self, temp_output_dir, mock_result):
        """diagnostic_integrity block is present with correct fields."""
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_integrity",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        integrity = diagnostic["signal_inputs"]["diagnostic_integrity"]

        assert "uses_only_canonical_sources" in integrity
        assert "coerced_sources_count" in integrity
        assert "missing_required_count" in integrity
        assert isinstance(integrity["uses_only_canonical_sources"], bool)
        assert isinstance(integrity["coerced_sources_count"], int)
        assert isinstance(integrity["missing_required_count"], int)

    def test_diagnostic_integrity_uses_only_canonical_true(self, temp_output_dir, mock_result):
        """uses_only_canonical_sources is true when no coercion needed."""
        # Create all real signals - no coercion needed
        with open(temp_output_dir / "replay_safety_governance_signal.json", "w") as f:
            json.dump({"status": "OK", "governance_alignment": "aligned", "conflict": False}, f)

        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_canonical_true",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        integrity = diagnostic["signal_inputs"]["diagnostic_integrity"]

        # All sources are canonical (real, stub, none) - no coercion
        assert integrity["uses_only_canonical_sources"] is True
        assert integrity["coerced_sources_count"] == 0

    def test_diagnostic_integrity_missing_required_count(self, temp_output_dir, mock_result):
        """missing_required_count matches missing_required_artifacts length."""
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_missing_count",
            cycle_count=1,
        )

        with open(temp_output_dir / "p5_divergence_diagnostic.json", "r") as f:
            diagnostic = json.load(f)

        signal_inputs = diagnostic["signal_inputs"]
        integrity = signal_inputs["diagnostic_integrity"]

        # missing_required_count should match the list length
        assert integrity["missing_required_count"] == len(signal_inputs["missing_required_artifacts"])

    def test_paths_tried_max_3_entries(self):
        """expected_paths_tried is limited to max 3 entries."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _build_signal_inputs

        source_summary = {"signal1": "stub", "signal2": "stub"}
        missing_reasons = {"signal1": "file not found", "signal2": "file not found"}
        paths_tried = {
            "signal1": ["path1.json", "path2.json", "path3.json", "path4.json", "path5.json"],
            "signal2": ["a.json", "b.json"],
        }

        result = _build_signal_inputs(source_summary, missing_reasons, paths_tried, {})

        missing_artifacts = result["missing_required_artifacts"]

        # Find signal1
        signal1_entry = next((e for e in missing_artifacts if e["signal"] == "signal1"), None)
        assert signal1_entry is not None
        assert len(signal1_entry["expected_paths_tried"]) == 3  # Max 3

        # Find signal2
        signal2_entry = next((e for e in missing_artifacts if e["signal"] == "signal2"), None)
        assert signal2_entry is not None
        assert len(signal2_entry["expected_paths_tried"]) == 2  # Actual count

    def test_advisory_log_output(self, temp_output_dir, mock_result, capsys):
        """Advisory line is logged with source counts."""
        divergence_log = temp_output_dir / "divergence_log.jsonl"
        with open(divergence_log, "w") as f:
            f.write(json.dumps({"cycle": 0, "severity": "NONE", "type": "NONE"}) + "\n")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import save_p5_diagnostic

        save_p5_diagnostic(
            output_dir=temp_output_dir,
            divergence_log_path=divergence_log,
            result=mock_result,
            run_id="test_advisory_log",
            cycle_count=1,
        )

        captured = capsys.readouterr()
        assert "[P5 Diagnostic] Source summary:" in captured.out
        assert "real=" in captured.out
        assert "stub=" in captured.out
        assert "none=" in captured.out

    def test_build_signal_inputs_with_unknown_source_coercion(self):
        """_build_signal_inputs correctly handles source coercion in integrity block."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _build_signal_inputs

        # Include an unknown source that will be coerced
        source_summary = {
            "divergence": "real",
            "replay": "unknown_source",  # Will be coerced to stub
            "topology": "stub",
        }
        missing_reasons = {
            "replay": "file not found",
            "topology": "file not found",
        }

        result = _build_signal_inputs(source_summary, missing_reasons, {}, {})

        integrity = result["diagnostic_integrity"]

        # Should have coerced one source
        assert integrity["uses_only_canonical_sources"] is False
        assert integrity["coerced_sources_count"] == 1
        assert "source_coercion_advisories" in result
        assert len(result["source_coercion_advisories"]) == 1


class TestNeverLieAuditCaps:
    """Test explicit cap enforcement from NEVER LIE CONTRACT."""

    def test_paths_tried_cap_uses_constant(self):
        """paths_tried cap uses MAX_PATHS_TRIED_PER_SIGNAL constant."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import (
            _build_signal_inputs,
            MAX_PATHS_TRIED_PER_SIGNAL,
        )

        # Provide more paths than the cap
        excess_paths = MAX_PATHS_TRIED_PER_SIGNAL + 5
        paths_tried = {
            "test_signal": [f"path{i}.json" for i in range(excess_paths)],
        }
        source_summary = {"test_signal": "stub"}
        missing_reasons = {"test_signal": "file not found"}

        result = _build_signal_inputs(source_summary, missing_reasons, paths_tried, {})

        entry = result["missing_required_artifacts"][0]
        # Must never exceed MAX_PATHS_TRIED_PER_SIGNAL
        assert len(entry["expected_paths_tried"]) == MAX_PATHS_TRIED_PER_SIGNAL
        assert len(entry["expected_paths_tried"]) <= MAX_PATHS_TRIED_PER_SIGNAL

    def test_missing_artifacts_cap_uses_constant(self):
        """missing_required_artifacts cap uses MAX_MISSING_ARTIFACTS constant."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import (
            _build_signal_inputs,
            MAX_MISSING_ARTIFACTS,
        )

        # Create more signals than the cap
        excess_signals = MAX_MISSING_ARTIFACTS + 5
        source_summary = {f"signal_{i}": "stub" for i in range(excess_signals)}
        missing_reasons = {f"signal_{i}": "file not found" for i in range(excess_signals)}

        result = _build_signal_inputs(source_summary, missing_reasons, {}, {})

        # Must never exceed MAX_MISSING_ARTIFACTS
        assert len(result["missing_required_artifacts"]) == MAX_MISSING_ARTIFACTS
        assert len(result["missing_required_artifacts"]) <= MAX_MISSING_ARTIFACTS

    def test_reason_codes_top3_uses_constant(self):
        """reason_codes_top3 cap uses MAX_REASON_CODES constant."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import (
            _build_signal_inputs,
            MAX_REASON_CODES,
        )

        # Create multiple signals with different reasons to generate multiple codes
        source_summary = {
            "signal_a": "stub",
            "signal_b": "stub",
            "signal_c": "stub",
            "signal_d": "unknown",  # Will be coerced
            "signal_e": "stub",
        }
        missing_reasons = {
            "signal_a": "file not found",
            "signal_b": "malformed JSON",
            "signal_c": "missing required keys",
            "signal_d": "file not found",
            "signal_e": "file not found",
        }

        result = _build_signal_inputs(source_summary, missing_reasons, {}, {})

        reason_codes = result["diagnostic_integrity"]["reason_codes_top3"]
        # Must never exceed MAX_REASON_CODES
        assert len(reason_codes) <= MAX_REASON_CODES

    def test_reason_codes_top3_deterministic_ordering(self):
        """reason_codes_top3 is sorted by count desc, then code asc."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _build_signal_inputs

        # Create signals with known reason counts
        source_summary = {
            "signal_a": "stub",
            "signal_b": "stub",
            "signal_c": "stub",
            "signal_d": "stub",
        }
        missing_reasons = {
            "signal_a": "file not found",   # FILE_NOT_FOUND
            "signal_b": "file not found",   # FILE_NOT_FOUND (2 total)
            "signal_c": "malformed JSON",   # MALFORMED_JSON (1 total)
            "signal_d": "missing required keys",  # MISSING_KEYS (1 total)
        }

        result = _build_signal_inputs(source_summary, missing_reasons, {}, {})

        reason_codes = result["diagnostic_integrity"]["reason_codes_top3"]
        # FILE_NOT_FOUND should be first (count 2)
        assert reason_codes[0] == "FILE_NOT_FOUND"
        # Remaining codes should be in alphabetical order (both count 1)
        if len(reason_codes) > 1:
            assert reason_codes[1] == "MALFORMED_JSON"
        if len(reason_codes) > 2:
            assert reason_codes[2] == "MISSING_KEYS"

    def test_reason_codes_includes_coercion(self):
        """reason_codes_top3 includes UNKNOWN_SOURCE_COERCED when applicable."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _build_signal_inputs

        source_summary = {
            "signal_a": "unknown_source",  # Coerced
            "signal_b": "bad_source",      # Coerced
            "signal_c": "stub",
        }
        missing_reasons = {
            "signal_a": "file not found",
            "signal_b": "file not found",
            "signal_c": "file not found",
        }

        result = _build_signal_inputs(source_summary, missing_reasons, {}, {})

        reason_codes = result["diagnostic_integrity"]["reason_codes_top3"]
        assert "UNKNOWN_SOURCE_COERCED" in reason_codes

    def test_reason_codes_top3_deterministic_multiple_runs(self):
        """reason_codes_top3 produces identical output on multiple runs."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from usla_first_light_p4_harness import _build_signal_inputs

        source_summary = {
            "z_signal": "stub",
            "a_signal": "stub",
            "m_signal": "unknown",
        }
        missing_reasons = {
            "z_signal": "file not found",
            "a_signal": "malformed JSON",
            "m_signal": "file not found",
        }

        results = [
            _build_signal_inputs(source_summary, missing_reasons, {}, {})
            for _ in range(5)
        ]

        # All runs should produce identical reason_codes_top3
        first_codes = results[0]["diagnostic_integrity"]["reason_codes_top3"]
        for r in results[1:]:
            assert r["diagnostic_integrity"]["reason_codes_top3"] == first_codes
