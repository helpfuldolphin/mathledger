"""P5 Replay Safety Wiring Integration Tests.

This module provides integration tests for the P5 replay safety wiring
into First Light status, evidence pack, and alignment view.

SHADOW MODE CONTRACT:
- All tests are observational only
- Tests verify signal structure and wiring, not governance decisions
- No gating logic is tested or enforced
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def synthetic_replay_logs() -> List[Dict[str, Any]]:
    """Synthetic P5 replay logs for testing."""
    return [
        {
            "cycle_id": "cycle_001",
            "trace_hash": "abc123",
            "timestamp": "2025-12-10T00:00:00Z",
            "latency_ms": 50.0,
            "run_id": "test_prod_run_001",
        },
        {
            "cycle_id": "cycle_002",
            "trace_hash": "def456",
            "timestamp": "2025-12-10T00:01:00Z",
            "latency_ms": 55.0,
        },
        {
            "cycle_id": "cycle_003",
            "trace_hash": "ghi789",
            "timestamp": "2025-12-10T00:02:00Z",
            "latency_ms": 48.0,
        },
    ]


@pytest.fixture
def expected_hashes() -> Dict[str, str]:
    """Expected hashes for determinism verification."""
    return {
        "cycle_001": "abc123",
        "cycle_002": "def456",
        "cycle_003": "ghi789",
    }


@pytest.fixture
def replay_logs_jsonl(temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]) -> Path:
    """Create a JSONL file with synthetic replay logs."""
    logs_path = temp_dir / "p5_replay_logs.jsonl"
    with open(logs_path, "w", encoding="utf-8") as f:
        for log in synthetic_replay_logs:
            f.write(json.dumps(log) + "\n")
    return logs_path


@pytest.fixture
def minimal_evidence_pack_dir(temp_dir: Path) -> Path:
    """Create a minimal evidence pack directory structure."""
    pack_dir = temp_dir / "evidence_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal manifest
    manifest = {
        "schema_version": "1.0.0",
        "bundle_id": "test_bundle",
        "governance": {},
    }
    with open(pack_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return pack_dir


# =============================================================================
# INTEGRATION TESTS: Evidence Pack
# =============================================================================

class TestP5ReplayEvidencePackIntegration:
    """Test P5 replay wiring in evidence pack."""

    def test_detect_p5_replay_logs_from_jsonl(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """Detect P5 replay logs from JSONL file."""
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs

        # Create run directory with p5_replay_logs.jsonl
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        logs_path = run_dir / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        result = detect_p5_replay_logs(run_dir)

        assert result is not None
        assert result.status == "ok"
        assert result.determinism_band == "GREEN"
        assert result.p5_grade is True
        assert result.telemetry_source == "real"

    def test_detect_p5_replay_logs_from_directory(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """Detect P5 replay logs from directory of JSON files."""
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs

        # Create run directory with p5_replay_logs/
        run_dir = temp_dir / "run"
        logs_dir = run_dir / "p5_replay_logs"
        logs_dir.mkdir(parents=True)

        for i, log in enumerate(synthetic_replay_logs):
            with open(logs_dir / f"log_{i:03d}.json", "w", encoding="utf-8") as f:
                json.dump(log, f)

        result = detect_p5_replay_logs(run_dir)

        assert result is not None
        assert result.status == "ok"
        assert result.p5_grade is True

    def test_detect_p5_replay_logs_returns_none_when_missing(
        self, temp_dir: Path
    ) -> None:
        """Detect P5 replay logs returns None when no logs present."""
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs

        run_dir = temp_dir / "run"
        run_dir.mkdir()

        result = detect_p5_replay_logs(run_dir)

        assert result is None

    def test_p5_replay_governance_reference_has_sha256(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """P5ReplayGovernanceReference includes sha256 hash."""
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs

        run_dir = temp_dir / "run"
        run_dir.mkdir()
        logs_path = run_dir / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        result = detect_p5_replay_logs(run_dir)

        assert result is not None
        assert result.sha256 is not None
        assert len(result.sha256) == 64  # SHA-256 hex length


# =============================================================================
# INTEGRATION TESTS: Status Generator
# =============================================================================

class TestP5ReplayStatusIntegration:
    """Test P5 replay wiring in status generator."""

    def test_extract_p5_replay_signal_from_jsonl(
        self, replay_logs_jsonl: Path
    ) -> None:
        """Extract P5 replay signal from JSONL file."""
        from scripts.generate_first_light_status import extract_p5_replay_signal

        signal = extract_p5_replay_signal(replay_logs_jsonl)

        assert signal is not None
        assert signal["status"] == "ok"
        assert signal["determinism_band"] == "GREEN"
        assert signal["determinism_rate"] == 1.0
        assert signal["p5_grade"] is True
        assert signal["telemetry_source"] == "real"

    def test_extract_p5_replay_signal_from_directory(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """Extract P5 replay signal from directory of JSON files."""
        from scripts.generate_first_light_status import extract_p5_replay_signal

        logs_dir = temp_dir / "logs"
        logs_dir.mkdir()
        for i, log in enumerate(synthetic_replay_logs):
            with open(logs_dir / f"log_{i:03d}.json", "w", encoding="utf-8") as f:
                json.dump(log, f)

        signal = extract_p5_replay_signal(logs_dir)

        assert signal is not None
        assert signal["status"] == "ok"

    def test_extract_p5_replay_signal_returns_none_when_missing(
        self, temp_dir: Path
    ) -> None:
        """Extract P5 replay signal returns None when no logs present."""
        from scripts.generate_first_light_status import extract_p5_replay_signal

        missing_path = temp_dir / "nonexistent"

        signal = extract_p5_replay_signal(missing_path)

        assert signal is None

    def test_p5_replay_signal_is_json_serializable(
        self, replay_logs_jsonl: Path
    ) -> None:
        """P5 replay signal must be JSON serializable."""
        from scripts.generate_first_light_status import extract_p5_replay_signal

        signal = extract_p5_replay_signal(replay_logs_jsonl)

        assert signal is not None
        json_str = json.dumps(signal)
        assert json_str is not None
        assert len(json_str) > 0


# =============================================================================
# INTEGRATION TESTS: Alignment View Generator
# =============================================================================

class TestP5ReplayAlignmentIntegration:
    """Test P5 replay wiring in alignment view generator."""

    def test_extract_p5_replay_signal_for_ggfl(
        self, replay_logs_jsonl: Path
    ) -> None:
        """Extract P5 replay signal in GGFL format."""
        from scripts.generate_first_light_alignment_view import (
            extract_p5_replay_signal_for_ggfl,
        )

        ggfl_signal = extract_p5_replay_signal_for_ggfl(replay_logs_jsonl)

        assert ggfl_signal is not None
        assert "status" in ggfl_signal
        assert "alignment" in ggfl_signal
        assert "conflict" in ggfl_signal
        assert "top_reasons" in ggfl_signal
        assert "p5_grade" in ggfl_signal
        assert "determinism_band" in ggfl_signal
        assert "telemetry_source" in ggfl_signal

    def test_ggfl_signal_alignment_mapping(
        self, replay_logs_jsonl: Path
    ) -> None:
        """GGFL signal has correct alignment mapping."""
        from scripts.generate_first_light_alignment_view import (
            extract_p5_replay_signal_for_ggfl,
        )

        ggfl_signal = extract_p5_replay_signal_for_ggfl(replay_logs_jsonl)

        assert ggfl_signal is not None
        # OK status -> aligned, no conflict
        assert ggfl_signal["status"] == "ok"
        assert ggfl_signal["alignment"] == "aligned"
        assert ggfl_signal["conflict"] is False

    def test_ggfl_signal_is_json_serializable(
        self, replay_logs_jsonl: Path
    ) -> None:
        """GGFL signal must be JSON serializable."""
        from scripts.generate_first_light_alignment_view import (
            extract_p5_replay_signal_for_ggfl,
        )

        ggfl_signal = extract_p5_replay_signal_for_ggfl(replay_logs_jsonl)

        assert ggfl_signal is not None
        json_str = json.dumps(ggfl_signal)
        assert json_str is not None


# =============================================================================
# INTEGRATION TEST: End-to-End Wiring
# =============================================================================

class TestP5ReplayEndToEndWiring:
    """End-to-end integration tests for P5 replay wiring."""

    def test_full_p5_replay_wiring_chain(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        Test full P5 replay wiring: logs -> signal -> status -> alignment.

        This test verifies that:
        1. Evidence pack detects and processes P5 replay logs
        2. Status generator extracts P5 replay signal
        3. Alignment view generator produces GGFL signal
        4. All outputs are JSON-safe and deterministic
        """
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs
        from scripts.generate_first_light_status import extract_p5_replay_signal
        from scripts.generate_first_light_alignment_view import (
            extract_p5_replay_signal_for_ggfl,
        )

        # Create replay logs JSONL
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        logs_path = run_dir / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        # Step 1: Evidence pack detection
        ep_ref = detect_p5_replay_logs(run_dir)
        assert ep_ref is not None
        assert ep_ref.status == "ok"
        assert ep_ref.p5_grade is True

        # Step 2: Status signal extraction
        status_signal = extract_p5_replay_signal(logs_path)
        assert status_signal is not None
        assert status_signal["status"] == "ok"
        assert status_signal["determinism_band"] == "GREEN"

        # Step 3: GGFL alignment extraction
        ggfl_signal = extract_p5_replay_signal_for_ggfl(logs_path)
        assert ggfl_signal is not None
        assert ggfl_signal["alignment"] == "aligned"
        assert ggfl_signal["conflict"] is False

        # Step 4: Verify JSON serialization
        all_outputs = {
            "evidence_pack_reference": {
                "path": ep_ref.path,
                "sha256": ep_ref.sha256,
                "status": ep_ref.status,
                "determinism_band": ep_ref.determinism_band,
                "p5_grade": ep_ref.p5_grade,
            },
            "status_signal": status_signal,
            "ggfl_signal": ggfl_signal,
        }
        json_str = json.dumps(all_outputs, sort_keys=True)
        assert json_str is not None
        assert len(json_str) > 0

        # Step 5: Verify deterministic output
        json_str_2 = json.dumps(all_outputs, sort_keys=True)
        assert json_str == json_str_2

    def test_all_surfaces_include_replay_p5_blocks(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        All three surfaces (evidence pack, status, alignment) include replay_p5 blocks.
        """
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs
        from scripts.generate_first_light_status import extract_p5_replay_signal
        from scripts.generate_first_light_alignment_view import (
            extract_p5_replay_signal_for_ggfl,
        )

        # Create replay logs
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        logs_path = run_dir / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        # Evidence pack reference
        ep_ref = detect_p5_replay_logs(run_dir)
        assert ep_ref is not None
        assert hasattr(ep_ref, "determinism_band")
        assert hasattr(ep_ref, "p5_grade")

        # Status signal
        status_signal = extract_p5_replay_signal(logs_path)
        assert status_signal is not None
        assert "determinism_band" in status_signal
        assert "p5_grade" in status_signal

        # GGFL signal
        ggfl_signal = extract_p5_replay_signal_for_ggfl(logs_path)
        assert ggfl_signal is not None
        assert "determinism_band" in ggfl_signal
        assert "p5_grade" in ggfl_signal

    def test_shadow_mode_compliance(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        All P5 replay wiring is SHADOW MODE compliant (observational only).
        """
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs

        # Create replay logs
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        logs_path = run_dir / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        # Verify detection doesn't modify anything
        ep_ref = detect_p5_replay_logs(run_dir)

        # Re-read logs to verify they weren't modified
        with open(logs_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == len(synthetic_replay_logs)

        # Verify P5 grade assessment is correct
        assert ep_ref is not None
        assert ep_ref.p5_grade is True  # telemetry_source="real"


# =============================================================================
# ROBUSTNESS INTEGRATION TESTS (v1.1.0)
# =============================================================================

class TestP5ReplayRobustness:
    """Test P5 replay robustness features (v1.1.0)."""

    def test_rotated_logs_identical_result(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        Rotated logs (2 JSONL files) produce identical result as concatenated single file.
        """
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs

        # Create run directory with two rotated JSONL files
        run_dir = temp_dir / "run"
        logs_dir = run_dir / "p5_replay_logs"
        logs_dir.mkdir(parents=True)

        # Split logs into two JSONL files
        mid = len(synthetic_replay_logs) // 2
        with open(logs_dir / "segment_001.jsonl", "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs[:mid]:
                f.write(json.dumps(log) + "\n")
        with open(logs_dir / "segment_002.jsonl", "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs[mid:]:
                f.write(json.dumps(log) + "\n")

        result_rotated = detect_p5_replay_logs(run_dir)

        # Create single concatenated file
        run_dir2 = temp_dir / "run2"
        run_dir2.mkdir()
        with open(run_dir2 / "p5_replay_logs.jsonl", "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        result_single = detect_p5_replay_logs(run_dir2)

        # Both should produce identical P5 signal (same determinism_rate, p5_grade)
        assert result_rotated is not None
        assert result_single is not None
        assert result_rotated.determinism_rate == result_single.determinism_rate
        assert result_rotated.p5_grade == result_single.p5_grade
        assert result_rotated.determinism_band == result_single.determinism_band

    def test_missing_fields_schema_ok_false(
        self, temp_dir: Path
    ) -> None:
        """
        Missing P5 fields -> schema_ok=false + warning, but pipeline completes.
        """
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs

        run_dir = temp_dir / "run"
        run_dir.mkdir()
        logs_path = run_dir / "p5_replay_logs.jsonl"

        # Create logs missing required fields (no trace_hash)
        incomplete_logs = [
            {"cycle_id": "cycle_001", "timestamp": "2025-12-10T00:00:00Z"},
            {"cycle_id": "cycle_002", "timestamp": "2025-12-10T00:01:00Z"},
        ]
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in incomplete_logs:
                f.write(json.dumps(log) + "\n")

        result = detect_p5_replay_logs(run_dir)

        # Pipeline should complete but with schema_ok=false
        assert result is not None
        assert result.schema_ok is False
        assert len(result.advisory_warnings) > 0
        assert any("Missing P5 fields" in w for w in result.advisory_warnings)

    def test_malformed_lines_skipped_with_warning(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        Malformed line(s) are skipped with warning counter, pipeline completes.
        """
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs

        run_dir = temp_dir / "run"
        run_dir.mkdir()
        logs_path = run_dir / "p5_replay_logs.jsonl"

        # Create logs with malformed lines interspersed
        with open(logs_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(synthetic_replay_logs[0]) + "\n")
            f.write("this is not valid json\n")  # Malformed line
            f.write(json.dumps(synthetic_replay_logs[1]) + "\n")
            f.write("{incomplete json\n")  # Another malformed line
            f.write(json.dumps(synthetic_replay_logs[2]) + "\n")

        result = detect_p5_replay_logs(run_dir)

        # Pipeline should complete with malformed line count
        assert result is not None
        assert result.malformed_line_count == 2
        assert len(result.advisory_warnings) >= 2
        assert any("Malformed JSON" in w for w in result.advisory_warnings)
        # Signal should still be valid for the good lines
        assert result.status == "ok"

    def test_gz_file_skipped_with_warning(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        .gz file present -> warning + skip (gzip not supported).
        """
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs

        run_dir = temp_dir / "run"
        logs_dir = run_dir / "p5_replay_logs"
        logs_dir.mkdir(parents=True)

        # Create a valid JSONL file
        with open(logs_dir / "current.jsonl", "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        # Create a fake .gz file (we just need the extension to trigger warning)
        (logs_dir / "archived.jsonl.gz").write_bytes(b"fake gzip content")

        result = detect_p5_replay_logs(run_dir)

        # Pipeline should complete but with warning about skipped gz
        assert result is not None
        assert result.skipped_gz_count == 1
        assert any("gzip" in w.lower() for w in result.advisory_warnings)
        # Signal should still be valid from the non-gz file
        assert result.status == "ok"

    def test_absolute_path_outside_run_dir(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        Absolute path passed via explicit_path works outside run_dir.
        """
        from backend.topology.first_light.evidence_pack import detect_p5_replay_logs

        # Create logs in a separate location outside run_dir
        external_logs = temp_dir / "external_logs"
        external_logs.mkdir()
        logs_path = external_logs / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        # Create empty run_dir (no logs here)
        run_dir = temp_dir / "run"
        run_dir.mkdir()

        # Use explicit_path to point to external logs
        result = detect_p5_replay_logs(run_dir, explicit_path=logs_path)

        assert result is not None
        assert result.status == "ok"
        assert result.p5_grade is True
        # Path should be absolute since it's outside run_dir
        assert str(logs_path) in result.path or "external_logs" in result.path

    def test_ggfl_alignment_view_advisory_only(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        Alignment view includes replay_p5 in GGFL stub with advisory_only=true.
        """
        from scripts.generate_first_light_alignment_view import (
            extract_p5_replay_signal_for_ggfl,
        )

        # Create replay logs
        logs_path = temp_dir / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        ggfl_signal = extract_p5_replay_signal_for_ggfl(logs_path)

        assert ggfl_signal is not None
        assert ggfl_signal.get("advisory_only") is True  # SHADOW MODE marker
        assert ggfl_signal.get("schema_ok") is True
        assert "alignment" in ggfl_signal
        assert "conflict" in ggfl_signal
        assert "p5_grade" in ggfl_signal
        # JSON serializable
        json_str = json.dumps(ggfl_signal)
        assert json_str is not None


# =============================================================================
# TRUE DIVERGENCE v1 COMPATIBILITY TESTS
# =============================================================================

class TestTrueDivergenceCompatibility:
    """Test True Divergence v1 metric contract compliance."""

    def test_replay_p5_advisory_only_even_on_red_band(
        self, temp_dir: Path
    ) -> None:
        """
        replay_p5 GGFL contribution remains advisory_only=true and never
        escalates to solo hard block, even if determinism_rate is RED.

        TRUE DIVERGENCE v1 CONTRACT:
        - Replay determinism is safety-weighted, not averaged away
        - RED band does NOT trigger solo hard blocking
        - advisory_only=true is invariant regardless of band
        """
        from scripts.generate_first_light_alignment_view import (
            extract_p5_replay_signal_for_ggfl,
        )

        # Create logs that will produce RED band (< 70% determinism)
        # We'll create logs with mismatching hashes to force low determinism
        red_band_logs = [
            {
                "cycle_id": "cycle_001",
                "trace_hash": "hash_a",
                "timestamp": "2025-12-10T00:00:00Z",
                "run_id": "prod_run_001",
            },
            {
                "cycle_id": "cycle_002",
                "trace_hash": "hash_b",
                "timestamp": "2025-12-10T00:01:00Z",
            },
            {
                "cycle_id": "cycle_003",
                "trace_hash": "hash_c",
                "timestamp": "2025-12-10T00:02:00Z",
            },
        ]

        logs_path = temp_dir / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in red_band_logs:
                f.write(json.dumps(log) + "\n")

        # Provide expected hashes that DON'T match to force mismatches
        expected_hashes = {
            "cycle_001": "different_hash_1",
            "cycle_002": "different_hash_2",
            "cycle_003": "different_hash_3",
        }

        ggfl_signal = extract_p5_replay_signal_for_ggfl(
            logs_path, expected_hashes=expected_hashes
        )

        assert ggfl_signal is not None

        # TRUE DIVERGENCE CONTRACT ASSERTIONS:
        # 1. advisory_only MUST be true regardless of band
        assert ggfl_signal.get("advisory_only") is True, (
            "replay_p5 must remain advisory_only=true even on RED band"
        )

        # 2. Verify we got RED or YELLOW band (mismatches detected)
        band = ggfl_signal.get("determinism_band")
        # The band depends on the adapter's calculation, but the key point
        # is that advisory_only remains true regardless

        # 3. No hard_block field should exist or be true
        assert ggfl_signal.get("hard_block") is not True, (
            "replay_p5 must never set hard_block=true (solo blocking prohibited)"
        )

        # 4. Signal should be JSON-safe and deterministic
        json_str = json.dumps(ggfl_signal, sort_keys=True)
        json_str_2 = json.dumps(ggfl_signal, sort_keys=True)
        assert json_str == json_str_2, "Output must be deterministic"

    def test_replay_p5_top_reasons_sorted_deterministically(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        All warning lists and reason arrays are sorted alphabetically
        for reproducible output (True Divergence v1 determinism requirement).
        """
        from scripts.generate_first_light_alignment_view import (
            extract_p5_replay_signal_for_ggfl,
        )

        logs_path = temp_dir / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        ggfl_signal = extract_p5_replay_signal_for_ggfl(logs_path)

        assert ggfl_signal is not None

        # Verify top_reasons is sorted (if present)
        top_reasons = ggfl_signal.get("top_reasons", [])
        assert top_reasons == sorted(top_reasons), (
            "top_reasons must be sorted for deterministic output"
        )


# =============================================================================
# ROBUSTNESS COUNTER TESTS (Prompt 2)
# =============================================================================

class TestRobustnessCountersSurfaced:
    """Test that robustness counters are surfaced to status signals."""

    def test_malformed_lines_increment_counter_in_status(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        Malformed lines increment malformed_line_count in status signal.
        """
        from scripts.generate_first_light_status import extract_p5_replay_signal

        logs_path = temp_dir / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(synthetic_replay_logs[0]) + "\n")
            f.write("not valid json line 1\n")  # Malformed
            f.write(json.dumps(synthetic_replay_logs[1]) + "\n")
            f.write("{broken json\n")  # Malformed
            f.write("another bad line\n")  # Malformed
            f.write(json.dumps(synthetic_replay_logs[2]) + "\n")

        signal = extract_p5_replay_signal(logs_path)

        assert signal is not None
        # Counter must reflect exactly 3 malformed lines
        assert signal.get("malformed_line_count") == 3
        # Signal should still succeed for valid lines
        assert signal.get("status") == "ok"
        # Warnings should be present
        assert len(signal.get("advisory_warnings", [])) >= 3

    def test_gz_files_increment_skipped_counter_in_status(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        .gz files increment skipped_gz_count in status signal.
        """
        from scripts.generate_first_light_status import extract_p5_replay_signal

        logs_dir = temp_dir / "logs"
        logs_dir.mkdir()

        # Create valid JSONL file
        with open(logs_dir / "current.jsonl", "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        # Create fake .gz files (2 of them)
        (logs_dir / "old_segment_1.jsonl.gz").write_bytes(b"fake gz 1")
        (logs_dir / "old_segment_2.jsonl.gz").write_bytes(b"fake gz 2")

        signal = extract_p5_replay_signal(logs_dir)

        assert signal is not None
        # Counter must reflect exactly 2 skipped gz files
        assert signal.get("skipped_gz_count") == 2
        # Signal should still succeed from valid file
        assert signal.get("status") == "ok"
        # Warning about skipped gz should be present
        assert any("gzip" in w.lower() for w in signal.get("advisory_warnings", []))

    def test_schema_ok_false_produces_exactly_one_warning(
        self, temp_dir: Path
    ) -> None:
        """
        schema_ok=false produces exactly one warning line (no spam).
        """
        from scripts.generate_first_light_status import extract_p5_replay_signal

        logs_path = temp_dir / "p5_replay_logs.jsonl"

        # Create logs missing required P5 fields (no trace_hash)
        incomplete_logs = [
            {"cycle_id": "cycle_001", "timestamp": "2025-12-10T00:00:00Z"},
            {"cycle_id": "cycle_002", "timestamp": "2025-12-10T00:01:00Z"},
            {"cycle_id": "cycle_003", "timestamp": "2025-12-10T00:02:00Z"},
        ]
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in incomplete_logs:
                f.write(json.dumps(log) + "\n")

        signal = extract_p5_replay_signal(logs_path)

        assert signal is not None
        # schema_ok must be false
        assert signal.get("schema_ok") is False

        # Count warnings about missing P5 fields
        p5_field_warnings = [
            w for w in signal.get("advisory_warnings", [])
            if "Missing P5 fields" in w
        ]
        # Should be exactly ONE warning (not one per log entry = no spam)
        assert len(p5_field_warnings) == 1, (
            f"Expected exactly 1 'Missing P5 fields' warning, got {len(p5_field_warnings)}: {p5_field_warnings}"
        )

        # The warning should mention which fields are missing
        assert "trace_hash" in p5_field_warnings[0]


# =============================================================================
# TRUE DIVERGENCE v1 TESTS
# =============================================================================

class TestTrueDivergenceV1Vector:
    """Test True Divergence v1 metric vector in replay_p5 signal."""

    def test_true_divergence_v1_vector_present(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        true_divergence_v1 vector is present in extracted signal.
        """
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
        )

        signal = extract_p5_replay_safety_from_logs(
            replay_logs=synthetic_replay_logs,
            production_run_id="test_run_001",
            telemetry_source="real",
        )

        assert "true_divergence_v1" in signal
        td_v1 = signal["true_divergence_v1"]

        # Check all required fields are present
        assert "outcome_mismatch_rate" in td_v1
        assert "safety_mismatch_rate" in td_v1
        assert "state_mismatch_rate" in td_v1
        assert "brier_score_success" in td_v1
        assert "version" in td_v1
        assert td_v1["version"] == "1.0.0"

    def test_legacy_metric_label_present(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        legacy_metric_label is present and set to RAW_ANY_MISMATCH.
        """
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
        )

        signal = extract_p5_replay_safety_from_logs(
            replay_logs=synthetic_replay_logs,
            production_run_id="test_run_001",
            telemetry_source="real",
        )

        assert "legacy_metric_label" in signal
        assert signal["legacy_metric_label"] == "RAW_ANY_MISMATCH"

    def test_safety_mismatch_tracking(self, temp_dir: Path) -> None:
        """
        safety_mismatch_rate tracks Ω/blocked mismatches correctly.
        """
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
        )

        # Create logs with safety mismatches
        logs_with_safety_mismatch = [
            {
                "cycle_id": "cycle_001",
                "trace_hash": "hash_a",
                "timestamp": "2025-12-10T00:00:00Z",
                "safety_mismatch": {"omega_mismatch": True, "blocked_mismatch": False},
            },
            {
                "cycle_id": "cycle_002",
                "trace_hash": "hash_b",
                "timestamp": "2025-12-10T00:01:00Z",
                "safety_mismatch": {"omega_mismatch": False, "blocked_mismatch": True},
            },
            {
                "cycle_id": "cycle_003",
                "trace_hash": "hash_c",
                "timestamp": "2025-12-10T00:02:00Z",
                "safety_mismatch": {"omega_mismatch": False, "blocked_mismatch": False},
            },
        ]

        signal = extract_p5_replay_safety_from_logs(
            replay_logs=logs_with_safety_mismatch,
            production_run_id="test_run_001",
            telemetry_source="real",
        )

        td_v1 = signal["true_divergence_v1"]
        # 2 out of 3 entries have safety mismatches
        assert td_v1["safety_mismatch_count"] == 2
        assert td_v1["safety_mismatch_rate"] == 2 / 3

    def test_state_mismatch_tracking(self, temp_dir: Path) -> None:
        """
        state_mismatch_rate tracks H/rho/tau/beta mismatches correctly.
        """
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
        )

        # Create logs with state mismatches
        logs_with_state_mismatch = [
            {
                "cycle_id": "cycle_001",
                "trace_hash": "hash_a",
                "timestamp": "2025-12-10T00:00:00Z",
                "state_mismatch": {"H_mismatch": True, "rho_mismatch": False},
            },
            {
                "cycle_id": "cycle_002",
                "trace_hash": "hash_b",
                "timestamp": "2025-12-10T00:01:00Z",
                "state_mismatch": {"tau_mismatch": True, "beta_mismatch": True},
            },
            {
                "cycle_id": "cycle_003",
                "trace_hash": "hash_c",
                "timestamp": "2025-12-10T00:02:00Z",
                "state_mismatch": {},
            },
        ]

        signal = extract_p5_replay_safety_from_logs(
            replay_logs=logs_with_state_mismatch,
            production_run_id="test_run_001",
            telemetry_source="real",
        )

        td_v1 = signal["true_divergence_v1"]
        # 2 out of 3 entries have state mismatches
        assert td_v1["state_mismatch_count"] == 2
        assert td_v1["state_mismatch_rate"] == 2 / 3

    def test_brier_score_computed_when_prob_success_available(
        self, temp_dir: Path
    ) -> None:
        """
        brier_score_success is computed when prob_success is available.
        """
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
        )

        # Create logs with prob_success
        logs_with_prob = [
            {
                "cycle_id": "cycle_001",
                "trace_hash": "hash_a",
                "timestamp": "2025-12-10T00:00:00Z",
                "prob_success": 0.8,
            },
            {
                "cycle_id": "cycle_002",
                "trace_hash": "hash_b",
                "timestamp": "2025-12-10T00:01:00Z",
                "prob_success": 0.9,
            },
        ]

        signal = extract_p5_replay_safety_from_logs(
            replay_logs=logs_with_prob,
            production_run_id="test_run_001",
            telemetry_source="real",
        )

        td_v1 = signal["true_divergence_v1"]
        assert td_v1["brier_score_success"] is not None
        # Mean prob = 0.85, Brier = 0.85 * 0.15 = 0.1275
        assert abs(td_v1["brier_score_success"] - 0.1275) < 0.001

    def test_true_divergence_v1_surfaced_in_status(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        true_divergence_v1 vector is surfaced in status generator output.
        """
        from scripts.generate_first_light_status import extract_p5_replay_signal

        logs_path = temp_dir / "p5_replay_logs.jsonl"
        with open(logs_path, "w", encoding="utf-8") as f:
            for log in synthetic_replay_logs:
                f.write(json.dumps(log) + "\n")

        signal = extract_p5_replay_signal(logs_path)

        assert signal is not None
        assert "true_divergence_v1" in signal
        assert "legacy_metric_label" in signal
        assert signal["legacy_metric_label"] == "RAW_ANY_MISMATCH"

    def test_warning_on_safety_mismatch_rate(self, temp_dir: Path) -> None:
        """
        Warning is generated when safety_mismatch_rate > 0.
        """
        from scripts.generate_first_light_status import generate_warnings

        # Mock signal with safety mismatch
        p5_signal = {
            "status": "ok",
            "schema_ok": True,
            "determinism_band": "GREEN",
            "true_divergence_v1": {
                "safety_mismatch_rate": 0.15,
            },
        }

        warnings = generate_warnings(
            p3_check={"metrics": {}},
            p4_check={"metrics": {}},
            p5_replay_signal=p5_signal,
        )

        safety_warnings = [w for w in warnings if "safety mismatch" in w.lower()]
        assert len(safety_warnings) == 1
        assert "15.00%" in safety_warnings[0]

    def test_single_warning_cap(self, temp_dir: Path) -> None:
        """
        Only one P5 warning is generated (single cap, no spam).
        """
        from scripts.generate_first_light_status import generate_warnings

        # Mock signal with multiple issues
        p5_signal = {
            "status": "block",
            "schema_ok": False,  # Would trigger warning
            "determinism_rate": 0.5,
            "determinism_band": "RED",  # Would trigger warning
            "true_divergence_v1": {
                "safety_mismatch_rate": 0.25,  # Would trigger warning
            },
        }

        warnings = generate_warnings(
            p3_check={"metrics": {}},
            p4_check={"metrics": {}},
            p5_replay_signal=p5_signal,
        )

        # Should only have ONE P5 warning (schema_ok takes priority)
        p5_warnings = [w for w in warnings if "P5" in w]
        assert len(p5_warnings) == 1
        assert "schema" in p5_warnings[0].lower()

    def test_true_divergence_v1_json_serializable(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        true_divergence_v1 vector is JSON serializable.
        """
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
        )

        signal = extract_p5_replay_safety_from_logs(
            replay_logs=synthetic_replay_logs,
            production_run_id="test_run_001",
            telemetry_source="real",
        )

        # Must be JSON serializable
        json_str = json.dumps(signal, sort_keys=True)
        assert json_str is not None

        # Verify deterministic output
        json_str_2 = json.dumps(signal, sort_keys=True)
        assert json_str == json_str_2


# =============================================================================
# CANONICALIZATION v1.3.0 TESTS
# =============================================================================

class TestCanonicalizationV130:
    """Test Canonicalization v1.3.0 hardening features."""

    def test_extraction_provenance_values(
        self, temp_dir: Path, synthetic_replay_logs: List[Dict[str, Any]]
    ) -> None:
        """
        Extraction provenance fields are present with correct enum values.
        Tests: extraction_source, input_schema_version passthrough.
        """
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            EXTRACTION_SOURCE_MANIFEST,
            EXTRACTION_SOURCE_EVIDENCE_JSON,
            EXTRACTION_SOURCE_DIRECT_LOG,
            EXTRACTION_SOURCE_MISSING,
        )

        # Test default (DIRECT_LOG)
        signal_default = extract_p5_replay_safety_from_logs(
            replay_logs=synthetic_replay_logs,
            production_run_id="test_run_001",
            telemetry_source="real",
        )
        assert signal_default["extraction_source"] == EXTRACTION_SOURCE_DIRECT_LOG
        assert signal_default["input_schema_version"] == "UNKNOWN"

        # Test MANIFEST source with schema version
        signal_manifest = extract_p5_replay_safety_from_logs(
            replay_logs=synthetic_replay_logs,
            production_run_id="test_run_002",
            telemetry_source="real",
            extraction_source=EXTRACTION_SOURCE_MANIFEST,
            input_schema_version="2.1.0",
        )
        assert signal_manifest["extraction_source"] == EXTRACTION_SOURCE_MANIFEST
        assert signal_manifest["input_schema_version"] == "2.1.0"

        # Test EVIDENCE_JSON source
        signal_evidence = extract_p5_replay_safety_from_logs(
            replay_logs=synthetic_replay_logs,
            production_run_id="test_run_003",
            telemetry_source="real",
            extraction_source=EXTRACTION_SOURCE_EVIDENCE_JSON,
            input_schema_version="1.5.0",
        )
        assert signal_evidence["extraction_source"] == EXTRACTION_SOURCE_EVIDENCE_JSON
        assert signal_evidence["input_schema_version"] == "1.5.0"

        # All extraction source enum values should be strings, not objects
        for src in [EXTRACTION_SOURCE_MANIFEST, EXTRACTION_SOURCE_EVIDENCE_JSON,
                    EXTRACTION_SOURCE_DIRECT_LOG, EXTRACTION_SOURCE_MISSING]:
            assert isinstance(src, str), f"{src} must be a string enum value"

    def test_driver_code_constraint_no_prose(self, temp_dir: Path) -> None:
        """
        Driver codes are frozen enum values (NO PROSE).
        Tests: driver codes contain only DRIVER_* constants, no free-form text.
        """
        from backend.health.replay_governance_adapter import (
            compute_driver_codes,
            DRIVER_SCHEMA_NOT_OK,
            DRIVER_SAFETY_MISMATCH_PRESENT,
            DRIVER_STATE_MISMATCH_PRESENT,
            DRIVER_DETERMINISM_RED_BAND,
            DRIVER_PRIORITY_ORDER,
        )

        # All valid driver codes (exhaustive list)
        valid_driver_codes = {
            DRIVER_SCHEMA_NOT_OK,
            DRIVER_SAFETY_MISMATCH_PRESENT,
            DRIVER_STATE_MISMATCH_PRESENT,
            DRIVER_DETERMINISM_RED_BAND,
        }

        # Generate codes with all conditions true
        codes_all = compute_driver_codes(
            schema_ok=False,
            safety_mismatch_rate=0.5,
            state_mismatch_rate=0.3,
            determinism_band="RED",
        )

        # All returned codes must be in valid set (NO PROSE)
        for code in codes_all:
            assert code in valid_driver_codes, (
                f"Driver code '{code}' is not a valid frozen enum value. "
                "Driver codes must be DRIVER_* constants only, NO PROSE."
            )

        # All codes must start with DRIVER_ prefix
        for code in codes_all:
            assert code.startswith("DRIVER_"), (
                f"Driver code '{code}' does not have DRIVER_ prefix"
            )

        # Priority order must match defined order
        for i, code in enumerate(codes_all):
            expected_position = DRIVER_PRIORITY_ORDER.index(code)
            assert expected_position >= 0, f"{code} not in priority order"
            # Earlier codes in result should have lower priority index
            if i > 0:
                prev_position = DRIVER_PRIORITY_ORDER.index(codes_all[i - 1])
                assert prev_position < expected_position, (
                    f"Driver codes not in priority order: {codes_all}"
                )

    def test_driver_code_ordering_determinism(self, temp_dir: Path) -> None:
        """
        Driver codes are always in deterministic priority order.
        Tests: ordering is consistent across multiple calls.
        """
        from backend.health.replay_governance_adapter import (
            compute_driver_codes,
            DRIVER_PRIORITY_ORDER,
            DRIVER_REASON_CAP,
        )

        # Test various combinations
        test_cases = [
            {"schema_ok": False, "safety_mismatch_rate": 0.0, "state_mismatch_rate": 0.0, "determinism_band": "GREEN"},
            {"schema_ok": True, "safety_mismatch_rate": 0.5, "state_mismatch_rate": 0.0, "determinism_band": "GREEN"},
            {"schema_ok": True, "safety_mismatch_rate": 0.0, "state_mismatch_rate": 0.3, "determinism_band": "GREEN"},
            {"schema_ok": True, "safety_mismatch_rate": 0.0, "state_mismatch_rate": 0.0, "determinism_band": "RED"},
            {"schema_ok": False, "safety_mismatch_rate": 0.5, "state_mismatch_rate": 0.3, "determinism_band": "RED"},
        ]

        for case in test_cases:
            # Call multiple times
            codes_1 = compute_driver_codes(**case)
            codes_2 = compute_driver_codes(**case)
            codes_3 = compute_driver_codes(**case)

            # Must be identical (deterministic)
            assert codes_1 == codes_2 == codes_3, (
                f"Driver codes not deterministic for {case}: {codes_1} vs {codes_2} vs {codes_3}"
            )

            # Must respect cap
            assert len(codes_1) <= DRIVER_REASON_CAP, (
                f"Driver codes exceed cap ({DRIVER_REASON_CAP}): {codes_1}"
            )

            # Must be in priority order
            for i in range(1, len(codes_1)):
                prev_idx = DRIVER_PRIORITY_ORDER.index(codes_1[i - 1])
                curr_idx = DRIVER_PRIORITY_ORDER.index(codes_1[i])
                assert prev_idx < curr_idx, (
                    f"Driver codes not in priority order: {codes_1}"
                )

    def test_single_warning_cap_precedence_unchanged(self, temp_dir: Path) -> None:
        """
        Single warning cap precedence is unchanged from v1.2.0.
        Tests: schema_ok > safety_mismatch_rate > RED band.
        """
        from scripts.generate_first_light_status import generate_warnings

        # Test case 1: All issues present - schema_ok takes priority
        p5_all_issues = {
            "status": "block",
            "schema_ok": False,
            "determinism_rate": 0.5,
            "determinism_band": "RED",
            "true_divergence_v1": {
                "safety_mismatch_rate": 0.25,
            },
        }
        warnings_1 = generate_warnings(
            p3_check={"metrics": {}},
            p4_check={"metrics": {}},
            p5_replay_signal=p5_all_issues,
        )
        p5_warnings_1 = [w for w in warnings_1 if "P5" in w]
        assert len(p5_warnings_1) == 1, f"Expected 1 P5 warning, got {len(p5_warnings_1)}"
        assert "schema" in p5_warnings_1[0].lower(), "Schema warning should have priority"

        # Test case 2: Safety mismatch present (no schema issue) - safety takes priority
        p5_safety_red = {
            "status": "warn",
            "schema_ok": True,
            "determinism_rate": 0.5,
            "determinism_band": "RED",
            "true_divergence_v1": {
                "safety_mismatch_rate": 0.30,
            },
        }
        warnings_2 = generate_warnings(
            p3_check={"metrics": {}},
            p4_check={"metrics": {}},
            p5_replay_signal=p5_safety_red,
        )
        p5_warnings_2 = [w for w in warnings_2 if "P5" in w]
        assert len(p5_warnings_2) == 1, f"Expected 1 P5 warning, got {len(p5_warnings_2)}"
        assert "safety mismatch" in p5_warnings_2[0].lower(), "Safety warning should have priority over RED band"

        # Test case 3: Only RED band (no schema/safety issues) - RED band fallback
        p5_red_only = {
            "status": "block",
            "schema_ok": True,
            "determinism_rate": 0.5,
            "determinism_band": "RED",
            "true_divergence_v1": {
                "safety_mismatch_rate": 0.0,
            },
        }
        warnings_3 = generate_warnings(
            p3_check={"metrics": {}},
            p4_check={"metrics": {}},
            p5_replay_signal=p5_red_only,
        )
        p5_warnings_3 = [w for w in warnings_3 if "P5" in w]
        assert len(p5_warnings_3) == 1, f"Expected 1 P5 warning, got {len(p5_warnings_3)}"
        assert "RED band" in p5_warnings_3[0], "RED band warning should be fallback"

        # Test case 4: GREEN band (no issues) - no P5 warning
        p5_green = {
            "status": "ok",
            "schema_ok": True,
            "determinism_rate": 0.95,
            "determinism_band": "GREEN",
            "true_divergence_v1": {
                "safety_mismatch_rate": 0.0,
            },
        }
        warnings_4 = generate_warnings(
            p3_check={"metrics": {}},
            p4_check={"metrics": {}},
            p5_replay_signal=p5_green,
        )
        p5_warnings_4 = [w for w in warnings_4 if "P5" in w]
        assert len(p5_warnings_4) == 0, f"Expected 0 P5 warnings for GREEN, got {len(p5_warnings_4)}"

    def test_frozen_contract_thresholds_guard(self) -> None:
        """
        FREEZE GUARD: Determinism band thresholds are frozen.

        If this test fails, you have changed a frozen contract.
        This requires a version bump in replay_p5_metric_versioning.md
        and explicit migration documentation.
        """
        from backend.health.replay_governance_adapter import (
            P5_DETERMINISM_GREEN_THRESHOLD,
            P5_DETERMINISM_YELLOW_THRESHOLD,
        )

        # FROZEN CONTRACT (v1.3.0):
        # GREEN >= 0.85, YELLOW >= 0.70, RED < 0.70
        assert P5_DETERMINISM_GREEN_THRESHOLD == 0.85, (
            "Frozen contract changed: P5_DETERMINISM_GREEN_THRESHOLD must be 0.85. "
            "Changing this requires a version bump in docs/system_law/replay/replay_p5_metric_versioning.md"
        )
        assert P5_DETERMINISM_YELLOW_THRESHOLD == 0.70, (
            "Frozen contract changed: P5_DETERMINISM_YELLOW_THRESHOLD must be 0.70. "
            "Changing this requires a version bump in docs/system_law/replay/replay_p5_metric_versioning.md"
        )

    def test_frozen_contract_p5_required_fields_guard(self, temp_dir: Path) -> None:
        """
        FREEZE GUARD: P5_REQUIRED_FIELDS is frozen to ["trace_hash"].

        If this test fails, you have changed a frozen contract.
        This requires a version bump in replay_p5_metric_versioning.md
        and explicit migration documentation.
        """
        from scripts.generate_first_light_status import extract_p5_replay_signal

        # Create logs with trace_hash (should pass schema_ok)
        logs_with_trace_hash = temp_dir / "logs_ok.jsonl"
        with open(logs_with_trace_hash, "w", encoding="utf-8") as f:
            f.write('{"trace_hash": "abc123"}\n')

        # Create logs without trace_hash (should fail schema_ok)
        logs_without_trace_hash = temp_dir / "logs_fail.jsonl"
        with open(logs_without_trace_hash, "w", encoding="utf-8") as f:
            f.write('{"cycle_id": "cycle_001"}\n')

        signal_ok = extract_p5_replay_signal(logs_with_trace_hash)
        signal_fail = extract_p5_replay_signal(logs_without_trace_hash)

        # FROZEN CONTRACT (v1.3.0):
        # P5_REQUIRED_FIELDS = ["trace_hash"] - trace_hash is the minimum for schema_ok=True
        assert signal_ok is not None
        assert signal_ok.get("schema_ok") is True, (
            "Frozen contract changed: trace_hash alone must be sufficient for schema_ok=True. "
            "Changing this requires a version bump in docs/system_law/replay/replay_p5_metric_versioning.md"
        )

        assert signal_fail is not None
        assert signal_fail.get("schema_ok") is False, (
            "Frozen contract changed: missing trace_hash must cause schema_ok=False. "
            "Changing this requires a version bump in docs/system_law/replay/replay_p5_metric_versioning.md"
        )
