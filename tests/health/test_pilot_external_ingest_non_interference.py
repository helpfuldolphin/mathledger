"""
Non-Interference Tests for Pilot External Ingest Adapter

SHADOW MODE CONTRACT:
- Proves external ingestion does not affect existing manifest fields
- Proves no schema changes to evidence pack structure
- Proves deterministic behavior
- Proves isolation from CAL-EXP-2 code paths

These tests validate that pilot external log ingestion is strictly additive
and does not interfere with existing evidence pack functionality.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from backend.health.pilot_external_ingest_adapter import (
    PILOT_INGEST_SCHEMA_VERSION,
    PilotIngestResult,
    PilotIngestSource,
    attach_to_manifest,
    compute_file_sha256,
    copy_to_evidence_pack,
    ingest_external_log,
    validate_external_log_schema,
    wrap_for_evidence_pack,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_external_log() -> Dict[str, Any]:
    """Sample valid external log artifact."""
    return {
        "log_type": "runtime_metrics",
        "timestamp": "2025-12-13T10:00:00Z",
        "entries": [
            {"metric": "cpu_usage", "value": 45.2},
            {"metric": "memory_mb", "value": 1024},
        ],
        "metadata": {"source": "pilot_system_a"},
    }


@pytest.fixture
def sample_manifest() -> Dict[str, Any]:
    """Sample existing evidence pack manifest."""
    return {
        "pack_version": "1.0.0",
        "created_at": "2025-12-13T10:00:00Z",
        "files": [
            {"path": "p3/summary.json", "sha256": "abc123"},
            {"path": "p4/results.json", "sha256": "def456"},
        ],
        "file_count": 2,
        "governance": {
            "p5_calibration": {
                "cal_exp1": {"verdict": "PASS"},
            },
            "rtts_validation_reference": {
                "path": "governance/rtts_validation.json",
                "sha256": "xyz789",
            },
        },
    }


# =============================================================================
# Schema Validation Tests
# =============================================================================

class TestSchemaValidation:
    """Test schema validation for external logs."""

    def test_valid_schema_passes(self, sample_external_log: Dict[str, Any]):
        """Valid external log passes schema validation."""
        is_valid, result, warnings = validate_external_log_schema(sample_external_log)
        assert is_valid is True
        assert result == PilotIngestResult.SUCCESS

    def test_missing_log_type_fails(self):
        """Missing log_type field fails validation."""
        invalid_log = {
            "timestamp": "2025-12-13T10:00:00Z",
            "entries": [],
        }
        is_valid, result, warnings = validate_external_log_schema(invalid_log)
        assert is_valid is False
        assert result == PilotIngestResult.SCHEMA_INVALID
        assert any("log_type" in w for w in warnings)

    def test_unrecognized_fields_warn_but_pass(self):
        """Unrecognized fields produce warnings but still pass."""
        log_with_extra = {
            "log_type": "test",
            "custom_field": "some_value",
            "another_custom": 123,
        }
        is_valid, result, warnings = validate_external_log_schema(log_with_extra)
        assert is_valid is True
        assert result == PilotIngestResult.SUCCESS
        assert len(warnings) == 2
        assert any("custom_field" in w for w in warnings)


# =============================================================================
# Ingest Tests
# =============================================================================

class TestIngestExternalLog:
    """Test external log ingestion."""

    def test_ingest_valid_json(self, sample_external_log: Dict[str, Any]):
        """Ingest valid JSON file successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            result = ingest_external_log(log_file)

            assert result["result"] == PilotIngestResult.SUCCESS
            assert result["source_type"] == PilotIngestSource.EXTERNAL_JSON
            assert result["data"] == sample_external_log
            assert result["sha256"] is not None
            assert result["extraction_source"] == "EXTERNAL_PILOT"
            assert result["mode"] == "SHADOW"
            assert result["action"] == "LOGGED_ONLY"

    def test_ingest_file_not_found(self):
        """Ingest returns FILE_NOT_FOUND for missing file."""
        result = ingest_external_log(Path("/nonexistent/file.json"))

        assert result["result"] == PilotIngestResult.FILE_NOT_FOUND
        assert result["source_type"] == PilotIngestSource.INVALID
        assert result["data"] is None

    def test_ingest_invalid_json(self):
        """Ingest returns PARSE_ERROR for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "invalid.json"
            log_file.write_text("{ not valid json")

            result = ingest_external_log(log_file)

            assert result["result"] == PilotIngestResult.PARSE_ERROR
            assert result["data"] is None

    def test_ingest_sha256_mismatch(self, sample_external_log: Dict[str, Any]):
        """Ingest returns INTEGRITY_MISMATCH for wrong SHA256."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            result = ingest_external_log(log_file, expected_sha256="wronghash")

            assert result["result"] == PilotIngestResult.INTEGRITY_MISMATCH
            assert result["data"] is None
            assert "integrity mismatch" in result["warnings"][0]

    def test_ingest_sha256_match(self, sample_external_log: Dict[str, Any]):
        """Ingest succeeds with correct SHA256."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            actual_sha256 = compute_file_sha256(log_file)
            result = ingest_external_log(log_file, expected_sha256=actual_sha256)

            assert result["result"] == PilotIngestResult.SUCCESS
            assert result["sha256"] == actual_sha256

    def test_ingest_jsonl_file(self):
        """Ingest JSONL file successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "events.jsonl"
            lines = [
                json.dumps({"event": "start", "ts": 1}),
                json.dumps({"event": "stop", "ts": 2}),
            ]
            log_file.write_text("\n".join(lines))

            result = ingest_external_log(log_file)

            assert result["result"] == PilotIngestResult.SUCCESS
            assert result["source_type"] == PilotIngestSource.EXTERNAL_JSONL
            assert result["data"]["log_type"] == "external_jsonl"
            assert result["data"]["entry_count"] == 2
            assert len(result["data"]["entries"]) == 2


# =============================================================================
# Non-Interference Tests
# =============================================================================

class TestNonInterference:
    """Test that pilot ingestion does not interfere with existing manifest."""

    def test_attach_preserves_existing_fields(
        self,
        sample_manifest: Dict[str, Any],
        sample_external_log: Dict[str, Any],
    ):
        """Attaching pilot entries preserves all existing manifest fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            ingest_result = ingest_external_log(log_file)
            entry = wrap_for_evidence_pack(ingest_result, log_file)
            new_manifest = attach_to_manifest(sample_manifest, [entry])

            # Verify all original fields preserved
            assert new_manifest["pack_version"] == sample_manifest["pack_version"]
            assert new_manifest["created_at"] == sample_manifest["created_at"]
            assert new_manifest["files"] == sample_manifest["files"]
            assert new_manifest["file_count"] == sample_manifest["file_count"]

    def test_attach_preserves_existing_governance(
        self,
        sample_manifest: Dict[str, Any],
        sample_external_log: Dict[str, Any],
    ):
        """Attaching pilot entries preserves existing governance fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            ingest_result = ingest_external_log(log_file)
            entry = wrap_for_evidence_pack(ingest_result, log_file)
            new_manifest = attach_to_manifest(sample_manifest, [entry])

            # Verify existing governance fields preserved
            assert "p5_calibration" in new_manifest["governance"]
            assert new_manifest["governance"]["p5_calibration"] == sample_manifest["governance"]["p5_calibration"]
            assert "rtts_validation_reference" in new_manifest["governance"]
            assert new_manifest["governance"]["rtts_validation_reference"] == sample_manifest["governance"]["rtts_validation_reference"]

    def test_attach_adds_external_pilot_section(
        self,
        sample_manifest: Dict[str, Any],
        sample_external_log: Dict[str, Any],
    ):
        """Attaching pilot entries adds external_pilot section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            ingest_result = ingest_external_log(log_file)
            entry = wrap_for_evidence_pack(ingest_result, log_file)
            new_manifest = attach_to_manifest(sample_manifest, [entry])

            # Verify external_pilot section added
            assert "external_pilot" in new_manifest["governance"]
            pilot_section = new_manifest["governance"]["external_pilot"]
            assert pilot_section["schema_version"] == PILOT_INGEST_SCHEMA_VERSION
            assert pilot_section["mode"] == "SHADOW"
            assert pilot_section["action"] == "LOGGED_ONLY"
            assert pilot_section["entry_count"] == 1

    def test_attach_does_not_mutate_original(
        self,
        sample_manifest: Dict[str, Any],
        sample_external_log: Dict[str, Any],
    ):
        """Attaching pilot entries does not mutate original manifest."""
        import copy
        original_copy = copy.deepcopy(sample_manifest)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            ingest_result = ingest_external_log(log_file)
            entry = wrap_for_evidence_pack(ingest_result, log_file)
            _ = attach_to_manifest(sample_manifest, [entry])

            # Original manifest unchanged
            assert sample_manifest == original_copy

    def test_empty_entries_produces_empty_section(
        self,
        sample_manifest: Dict[str, Any],
    ):
        """Empty entries list produces empty external_pilot section."""
        new_manifest = attach_to_manifest(sample_manifest, [])

        assert "external_pilot" in new_manifest["governance"]
        pilot_section = new_manifest["governance"]["external_pilot"]
        assert pilot_section["entry_count"] == 0
        assert pilot_section["entries"] == []


# =============================================================================
# Determinism Tests
# =============================================================================

class TestDeterminism:
    """Test deterministic behavior of pilot ingestion."""

    def test_ingest_deterministic(self, sample_external_log: Dict[str, Any]):
        """Repeated ingestion produces same result (excluding timestamp)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            result1 = ingest_external_log(log_file)
            result2 = ingest_external_log(log_file)

            # Strip timestamps for comparison
            def strip_time(d: Dict) -> Dict:
                return {k: v for k, v in d.items() if "timestamp" not in k.lower()}

            assert strip_time(result1) == strip_time(result2)

    def test_sha256_deterministic(self, sample_external_log: Dict[str, Any]):
        """SHA256 computation is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            sha1 = compute_file_sha256(log_file)
            sha2 = compute_file_sha256(log_file)

            assert sha1 == sha2

    def test_wrap_deterministic(self, sample_external_log: Dict[str, Any]):
        """Wrapping for evidence pack is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            ingest_result = ingest_external_log(log_file)
            entry1 = wrap_for_evidence_pack(ingest_result, log_file)
            entry2 = wrap_for_evidence_pack(ingest_result, log_file)

            # Strip timestamps for comparison
            def strip_time(d: Dict) -> Dict:
                if not isinstance(d, dict):
                    return d
                result = {}
                for k, v in d.items():
                    if "timestamp" in k.lower() or k == "ingested_at":
                        continue
                    elif isinstance(v, dict):
                        result[k] = strip_time(v)
                    else:
                        result[k] = v
                return result

            assert strip_time(entry1) == strip_time(entry2)


# =============================================================================
# Copy Tests
# =============================================================================

class TestCopyToEvidencePack:
    """Test file copy to evidence pack."""

    def test_copy_creates_target_dir(self, sample_external_log: Dict[str, Any]):
        """Copy creates target directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.json"
            source_file.write_text(json.dumps(sample_external_log))

            pack_dir = Path(tmpdir) / "evidence_pack"
            pack_dir.mkdir()

            result = copy_to_evidence_pack(source_file, pack_dir, "external")

            assert result is not None
            assert result.exists()
            assert result.parent.name == "external"

    def test_copy_does_not_overwrite(self, sample_external_log: Dict[str, Any]):
        """Copy does not overwrite existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.json"
            source_file.write_text(json.dumps(sample_external_log))

            pack_dir = Path(tmpdir) / "evidence_pack"
            target_dir = pack_dir / "external"
            target_dir.mkdir(parents=True)

            # Create existing file
            existing = target_dir / "source.json"
            existing.write_text("existing content")

            result = copy_to_evidence_pack(source_file, pack_dir, "external")

            assert result is None
            # Original content preserved
            assert existing.read_text() == "existing content"


# =============================================================================
# CAL-EXP-2 Isolation Tests
# =============================================================================

class TestCALEXP2Isolation:
    """Test isolation from CAL-EXP-2 frozen code paths."""

    def test_no_import_of_rtts_adapter(self):
        """Pilot adapter does not import CAL-EXP-2 RTTS adapter."""
        import backend.health.pilot_external_ingest_adapter as pilot

        # Check module has no reference to RTTS
        module_source = Path(pilot.__file__).read_text()
        assert "rtts_status_adapter" not in module_source
        assert "RTTSStatisticalValidator" not in module_source

    def test_independent_schema_version(self):
        """Pilot adapter has independent schema version."""
        from backend.health.rtts_status_adapter import RTTS_STATUS_SIGNAL_SCHEMA_VERSION

        # Schema versions should be independent
        assert PILOT_INGEST_SCHEMA_VERSION != RTTS_STATUS_SIGNAL_SCHEMA_VERSION

    def test_no_modification_to_rtts_fields(
        self,
        sample_manifest: Dict[str, Any],
        sample_external_log: Dict[str, Any],
    ):
        """Pilot ingestion does not modify rtts_validation_reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "external.json"
            log_file.write_text(json.dumps(sample_external_log))

            ingest_result = ingest_external_log(log_file)
            entry = wrap_for_evidence_pack(ingest_result, log_file)
            new_manifest = attach_to_manifest(sample_manifest, [entry])

            # RTTS reference unchanged
            assert new_manifest["governance"]["rtts_validation_reference"] == {
                "path": "governance/rtts_validation.json",
                "sha256": "xyz789",
            }


# =============================================================================
# E2E Smoke Tests (Full Chain)
# =============================================================================

class TestE2ESmoke:
    """
    End-to-end smoke tests exercising the full chain:
    ingest → wrap_for_evidence_pack → attach_to_manifest → copy_to_evidence_pack

    SHADOW MODE CONTRACT:
    - Only governance.external_pilot changes
    - No CAL-EXP namespaces touched
    - Warnings ≤ 1, single-line, neutral language
    - Deterministic after stripping time keys
    """

    def _strip_time_keys(self, d: Any) -> Any:
        """Recursively strip time-related keys for determinism comparison."""
        if isinstance(d, dict):
            return {
                k: self._strip_time_keys(v)
                for k, v in d.items()
                if not any(t in k.lower() for t in ["timestamp", "time", "ingested_at", "created_at"])
            }
        elif isinstance(d, list):
            return [self._strip_time_keys(item) for item in d]
        return d

    def _build_cal_exp_manifest(self) -> Dict[str, Any]:
        """Build a manifest with CAL-EXP namespaces for non-interference testing."""
        return {
            "pack_version": "1.0.0",
            "created_at": "2025-12-14T00:00:00Z",
            "files": [
                {"path": "p3/summary.json", "sha256": "p3hash"},
                {"path": "p4/results.json", "sha256": "p4hash"},
                {"path": "governance/rtts_validation.json", "sha256": "rttshash"},
            ],
            "file_count": 3,
            "governance": {
                "p5_calibration": {
                    "cal_exp1": {"verdict": "PASS", "cycles": 500},
                    "cal_exp2": {"verdict": "PLATEAUING", "cycles": 1000},
                },
                "rtts_validation_reference": {
                    "path": "governance/rtts_validation.json",
                    "sha256": "rttshash",
                },
                "schema_versioned": {
                    "ledger_guard_summary": {"version": "1.0.0"},
                    "cal_exp_reports": {"cal_exp1": "path/to/report"},
                },
                "prng_regime_timeseries": [
                    {"cycle": 0, "regime": "STABLE"},
                ],
            },
        }

    def test_e2e_json_full_chain(self):
        """E2E: JSON log through full chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create external JSON log
            json_log = {"log_type": "runtime_metrics", "entries": [{"cpu": 42}]}
            json_file = tmpdir_path / "metrics.json"
            json_file.write_text(json.dumps(json_log))

            # Create evidence pack directory
            pack_dir = tmpdir_path / "evidence_pack"
            pack_dir.mkdir()

            # Build manifest with CAL-EXP namespaces
            manifest = self._build_cal_exp_manifest()
            original_manifest = json.loads(json.dumps(manifest))  # Deep copy

            # Full chain: ingest → wrap → attach → copy
            ingest_result = ingest_external_log(json_file)
            assert ingest_result["result"] == PilotIngestResult.SUCCESS

            entry = wrap_for_evidence_pack(ingest_result, json_file)
            assert entry["valid"] is True

            new_manifest = attach_to_manifest(manifest, [entry])

            copied = copy_to_evidence_pack(json_file, pack_dir, "external")
            assert copied is not None
            assert copied.exists()

            # Assert: only governance.external_pilot changes
            assert "external_pilot" in new_manifest["governance"]

            # Assert: no CAL-EXP namespaces touched
            assert new_manifest["governance"]["p5_calibration"] == original_manifest["governance"]["p5_calibration"]
            assert new_manifest["governance"]["rtts_validation_reference"] == original_manifest["governance"]["rtts_validation_reference"]
            assert new_manifest["governance"]["schema_versioned"] == original_manifest["governance"]["schema_versioned"]
            assert new_manifest["governance"]["prng_regime_timeseries"] == original_manifest["governance"]["prng_regime_timeseries"]

            # Assert: warnings ≤ 1
            warnings = new_manifest["governance"]["external_pilot"].get("warnings", [])
            assert len(warnings) <= 1

    def test_e2e_jsonl_full_chain(self):
        """E2E: JSONL log through full chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create external JSONL log
            jsonl_lines = [
                json.dumps({"event": "start", "ts": 1}),
                json.dumps({"event": "process", "ts": 2}),
                json.dumps({"event": "stop", "ts": 3}),
            ]
            jsonl_file = tmpdir_path / "events.jsonl"
            jsonl_file.write_text("\n".join(jsonl_lines))

            # Create evidence pack directory
            pack_dir = tmpdir_path / "evidence_pack"
            pack_dir.mkdir()

            # Build manifest with CAL-EXP namespaces
            manifest = self._build_cal_exp_manifest()

            # Full chain
            ingest_result = ingest_external_log(jsonl_file)
            assert ingest_result["result"] == PilotIngestResult.SUCCESS
            assert ingest_result["source_type"] == PilotIngestSource.EXTERNAL_JSONL

            entry = wrap_for_evidence_pack(ingest_result, jsonl_file)
            new_manifest = attach_to_manifest(manifest, [entry])
            copied = copy_to_evidence_pack(jsonl_file, pack_dir, "external")

            # Assert: entry count correct
            assert new_manifest["governance"]["external_pilot"]["entry_count"] == 1
            assert copied is not None

    def test_e2e_determinism_json(self):
        """E2E: Two runs produce byte-identical normalized outputs (JSON)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            json_log = {"log_type": "test", "entries": [{"a": 1}, {"b": 2}]}
            json_file = tmpdir_path / "test.json"
            json_file.write_text(json.dumps(json_log))

            manifest = self._build_cal_exp_manifest()

            # Run 1
            r1 = ingest_external_log(json_file)
            e1 = wrap_for_evidence_pack(r1, json_file)
            m1 = attach_to_manifest(manifest, [e1])

            # Run 2
            r2 = ingest_external_log(json_file)
            e2 = wrap_for_evidence_pack(r2, json_file)
            m2 = attach_to_manifest(manifest, [e2])

            # Strip time keys and compare
            normalized_m1 = self._strip_time_keys(m1)
            normalized_m2 = self._strip_time_keys(m2)

            # Byte-identical when serialized
            assert json.dumps(normalized_m1, sort_keys=True) == json.dumps(normalized_m2, sort_keys=True)

    def test_e2e_determinism_jsonl(self):
        """E2E: Two runs produce byte-identical normalized outputs (JSONL)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            jsonl_lines = [json.dumps({"x": i}) for i in range(5)]
            jsonl_file = tmpdir_path / "test.jsonl"
            jsonl_file.write_text("\n".join(jsonl_lines))

            manifest = self._build_cal_exp_manifest()

            # Run 1
            r1 = ingest_external_log(jsonl_file)
            e1 = wrap_for_evidence_pack(r1, jsonl_file)
            m1 = attach_to_manifest(manifest, [e1])

            # Run 2
            r2 = ingest_external_log(jsonl_file)
            e2 = wrap_for_evidence_pack(r2, jsonl_file)
            m2 = attach_to_manifest(manifest, [e2])

            # Strip time keys and compare
            normalized_m1 = self._strip_time_keys(m1)
            normalized_m2 = self._strip_time_keys(m2)

            assert json.dumps(normalized_m1, sort_keys=True) == json.dumps(normalized_m2, sort_keys=True)

    def test_e2e_warning_single_line_neutral(self):
        """E2E: Warnings are ≤ 1, single-line, neutral language."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Log with unrecognized field (produces warning)
            json_log = {"log_type": "test", "unknown_field": "value"}
            json_file = tmpdir_path / "with_warning.json"
            json_file.write_text(json.dumps(json_log))

            manifest = self._build_cal_exp_manifest()

            ingest_result = ingest_external_log(json_file)
            entry = wrap_for_evidence_pack(ingest_result, json_file)
            new_manifest = attach_to_manifest(manifest, [entry])

            warnings = new_manifest["governance"]["external_pilot"].get("warnings", [])

            # Warnings ≤ 1 per entry
            assert len(warnings) <= 1

            # Each warning is single-line and neutral
            for w in warnings:
                assert "\n" not in w, "Warning must be single-line"
                # Neutral language check: no exclamation, no caps shouting
                assert "!" not in w, "Warning must use neutral language"
                assert w == w.lower() or not w.isupper(), "Warning must not shout"

    def test_e2e_multiple_files(self):
        """E2E: Multiple files through full chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create multiple external logs
            files = []
            for i in range(3):
                log = {"log_type": f"type_{i}", "entries": [{"idx": i}]}
                f = tmpdir_path / f"log_{i}.json"
                f.write_text(json.dumps(log))
                files.append(f)

            pack_dir = tmpdir_path / "evidence_pack"
            pack_dir.mkdir()

            manifest = self._build_cal_exp_manifest()
            original_cal_exp = json.loads(json.dumps(manifest["governance"]["p5_calibration"]))

            # Process all files
            entries = []
            for f in files:
                r = ingest_external_log(f)
                assert r["result"] == PilotIngestResult.SUCCESS
                e = wrap_for_evidence_pack(r, f)
                entries.append(e)
                copy_to_evidence_pack(f, pack_dir, "external")

            new_manifest = attach_to_manifest(manifest, entries)

            # Assert: all entries recorded
            assert new_manifest["governance"]["external_pilot"]["entry_count"] == 3

            # Assert: CAL-EXP namespaces untouched
            assert new_manifest["governance"]["p5_calibration"] == original_cal_exp

    def test_e2e_no_cal_exp_writes(self):
        """E2E: No writes to CAL-EXP result paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create directory structure mimicking real evidence pack
            pack_dir = tmpdir_path / "evidence_pack"
            (pack_dir / "governance").mkdir(parents=True)
            (pack_dir / "p3").mkdir()
            (pack_dir / "p4").mkdir()

            # Create sentinel files in CAL-EXP paths
            sentinel_governance = pack_dir / "governance" / "sentinel.txt"
            sentinel_p3 = pack_dir / "p3" / "sentinel.txt"
            sentinel_p4 = pack_dir / "p4" / "sentinel.txt"

            sentinel_governance.write_text("UNTOUCHED")
            sentinel_p3.write_text("UNTOUCHED")
            sentinel_p4.write_text("UNTOUCHED")

            # Create external log
            json_log = {"log_type": "test", "entries": []}
            json_file = tmpdir_path / "external.json"
            json_file.write_text(json.dumps(json_log))

            # Full chain with copy to "external" subdirectory
            r = ingest_external_log(json_file)
            e = wrap_for_evidence_pack(r, json_file)
            copy_to_evidence_pack(json_file, pack_dir, "external")

            # Assert: sentinel files untouched
            assert sentinel_governance.read_text() == "UNTOUCHED"
            assert sentinel_p3.read_text() == "UNTOUCHED"
            assert sentinel_p4.read_text() == "UNTOUCHED"

            # Assert: external directory created
            assert (pack_dir / "external" / "external.json").exists()
