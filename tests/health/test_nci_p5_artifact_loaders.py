"""Integration tests for NCI P5 artifact loaders.

Tests cover:
- load_doc_contents_for_nci() with real repository docs
- load_telemetry_schema_from_file() with fixture files
- load_slice_registry_from_file() with fixture files
- run_nci_p5_with_artifacts() in all 3 modes via file presence/absence
- build_nci_p5_compact_signal() output structure

SHADOW MODE CONTRACT:
- All tests are read-only
- No governance decisions are made
- Tests validate structure, not governance logic
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def repo_root() -> Path:
    """Get repository root (assumes tests run from repo root)."""
    # Try to find repo root by looking for pyproject.toml
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to cwd
    return Path.cwd()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test fixtures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_telemetry_schema(temp_dir: Path) -> Path:
    """Create a sample telemetry schema JSON file."""
    schema = {
        "schema_version": "1.2.0",
        "events": {
            "cycle_start": {"fields": ["cycle_id", "timestamp"]},
            "rsi_computed": {"fields": ["rsi", "H", "rho"]},
        },
        "fields": {
            "H": {"type": "float", "description": "Health metric"},
            "rho": {"type": "float", "description": "RSI value"},
            "tau": {"type": "float", "description": "Threshold"},
        },
        "schema_age_hours": 48,
    }
    schema_path = temp_dir / "telemetry_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    return schema_path


@pytest.fixture
def sample_slice_registry(temp_dir: Path) -> Path:
    """Create a sample slice registry JSON file."""
    registry = {
        "slices": {
            "arithmetic_simple": {
                "depth_max": 4,
                "atom_max": 4,
                "theory_id": "PL",
            },
            "propositional_tautology": {
                "depth_max": 6,
                "atom_max": 5,
                "theory_id": "PL",
            },
        },
        "registry_validated": True,
    }
    registry_path = temp_dir / "slice_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2))
    return registry_path


@pytest.fixture
def sample_docs_dir(temp_dir: Path) -> Path:
    """Create sample documentation directory with test files."""
    docs_dir = temp_dir / "docs"
    docs_dir.mkdir()

    # Create system_law subdirectory
    system_law = docs_dir / "system_law"
    system_law.mkdir()

    # File with canonical names (no violations)
    # Note: Only use canonical names here - don't mention variants even in context
    # Variants to avoid: health, h, Ht, H_t (variants of H), rsi, RSI (variants of rho)
    (system_law / "canonical.md").write_text(
        "# Canonical Names\n\n"
        "The H value is the system metric. Use rho for stability measurements.\n"
        "The arithmetic_simple slice handles basic math operations.\n",
        encoding="utf-8",
    )

    # File with TCL violations (non-canonical field names)
    # Use ASCII variants only to avoid Windows encoding issues
    (system_law / "tcl_violations.md").write_text(
        "# TCL Violations\n\n"
        "The Ht value is health. Use RSI for stability.\n"
        "The block_rate parameter controls rate.\n",
        encoding="utf-8",
    )

    # File with SIC violations (non-canonical slice names)
    (system_law / "sic_violations.md").write_text(
        "# SIC Violations\n\n"
        "The ArithmeticSimple slice is for basic math.\n"
        "Use prop_taut for propositional logic.\n",
        encoding="utf-8",
    )

    # File with SIC-004 capability overclaims
    (system_law / "overclaims.md").write_text(
        "# Capability Claims\n\n"
        "This slice supports unlimited depth derivation.\n"
        "There is no limit to proof complexity.\n",
        encoding="utf-8",
    )

    return docs_dir


# =============================================================================
# LOAD_DOC_CONTENTS_FOR_NCI TESTS
# =============================================================================


class TestLoadDocContentsForNci:
    """Tests for load_doc_contents_for_nci()."""

    def test_loads_docs_from_repo(self, repo_root: Path) -> None:
        """Can load docs from actual repository."""
        from backend.health.nci_governance_adapter import load_doc_contents_for_nci

        doc_contents = load_doc_contents_for_nci(repo_root, max_files=10)

        # Should load at least some docs
        assert len(doc_contents) > 0

        # All paths should be relative
        for path in doc_contents.keys():
            assert not Path(path).is_absolute()
            assert "\\" not in path  # Forward slashes only

    def test_loads_from_custom_patterns(self, sample_docs_dir: Path, temp_dir: Path) -> None:
        """Loads docs from custom glob patterns."""
        from backend.health.nci_governance_adapter import load_doc_contents_for_nci

        patterns = ["docs/system_law/*.md"]
        doc_contents = load_doc_contents_for_nci(temp_dir, patterns=patterns)

        # Should load the 4 test files
        assert len(doc_contents) == 4
        assert "docs/system_law/canonical.md" in doc_contents
        assert "docs/system_law/tcl_violations.md" in doc_contents

    def test_respects_max_files_limit(self, repo_root: Path) -> None:
        """Respects max_files limit."""
        from backend.health.nci_governance_adapter import load_doc_contents_for_nci

        doc_contents = load_doc_contents_for_nci(repo_root, max_files=3)

        assert len(doc_contents) <= 3

    def test_returns_empty_for_missing_dir(self, temp_dir: Path) -> None:
        """Returns empty dict for missing directory."""
        from backend.health.nci_governance_adapter import load_doc_contents_for_nci

        doc_contents = load_doc_contents_for_nci(
            temp_dir / "nonexistent",
            patterns=["*.md"],
        )

        assert doc_contents == {}


# =============================================================================
# TELEMETRY SCHEMA LOADER TESTS
# =============================================================================


class TestLoadTelemetrySchemaFromFile:
    """Tests for load_telemetry_schema_from_file()."""

    def test_loads_valid_schema(self, sample_telemetry_schema: Path) -> None:
        """Loads valid telemetry schema JSON."""
        from backend.health.nci_governance_adapter import load_telemetry_schema_from_file

        schema = load_telemetry_schema_from_file(sample_telemetry_schema)

        assert schema is not None
        assert schema["schema_version"] == "1.2.0"
        assert "events" in schema
        assert "fields" in schema
        assert schema["schema_age_hours"] == 48

    def test_returns_none_for_missing_file(self, temp_dir: Path) -> None:
        """Returns None for missing file."""
        from backend.health.nci_governance_adapter import load_telemetry_schema_from_file

        schema = load_telemetry_schema_from_file(temp_dir / "nonexistent.json")

        assert schema is None

    def test_returns_none_for_invalid_json(self, temp_dir: Path) -> None:
        """Returns None for invalid JSON."""
        from backend.health.nci_governance_adapter import load_telemetry_schema_from_file

        bad_file = temp_dir / "bad.json"
        bad_file.write_text("not valid json {{{")

        schema = load_telemetry_schema_from_file(bad_file)

        assert schema is None

    def test_computes_schema_age_if_missing(self, temp_dir: Path) -> None:
        """Computes schema_age_hours from file mtime if not in JSON."""
        from backend.health.nci_governance_adapter import load_telemetry_schema_from_file

        schema_file = temp_dir / "schema_no_age.json"
        schema_file.write_text('{"schema_version": "1.0.0", "events": {}}')

        schema = load_telemetry_schema_from_file(schema_file)

        assert schema is not None
        assert "schema_age_hours" in schema
        assert isinstance(schema["schema_age_hours"], int)


# =============================================================================
# SLICE REGISTRY LOADER TESTS
# =============================================================================


class TestLoadSliceRegistryFromFile:
    """Tests for load_slice_registry_from_file()."""

    def test_loads_valid_registry(self, sample_slice_registry: Path) -> None:
        """Loads valid slice registry JSON."""
        from backend.health.nci_governance_adapter import load_slice_registry_from_file

        registry = load_slice_registry_from_file(sample_slice_registry)

        assert registry is not None
        assert "slices" in registry
        assert "arithmetic_simple" in registry["slices"]
        assert registry["registry_validated"] is True

    def test_returns_none_for_missing_file(self, temp_dir: Path) -> None:
        """Returns None for missing file."""
        from backend.health.nci_governance_adapter import load_slice_registry_from_file

        registry = load_slice_registry_from_file(temp_dir / "nonexistent.json")

        assert registry is None

    def test_adds_empty_slices_if_missing(self, temp_dir: Path) -> None:
        """Adds empty slices dict if not in JSON."""
        from backend.health.nci_governance_adapter import load_slice_registry_from_file

        registry_file = temp_dir / "registry_no_slices.json"
        registry_file.write_text('{"registry_validated": false}')

        registry = load_slice_registry_from_file(registry_file)

        assert registry is not None
        assert "slices" in registry
        assert registry["slices"] == {}


# =============================================================================
# RUN_NCI_P5_WITH_ARTIFACTS TESTS (MODE SELECTION VIA FILE PRESENCE)
# =============================================================================


class TestRunNciP5WithArtifactsModeSelection:
    """Tests for run_nci_p5_with_artifacts() mode selection via file presence."""

    def test_doc_only_mode_without_files(
        self,
        sample_docs_dir: Path,
        temp_dir: Path,
    ) -> None:
        """DOC_ONLY mode when no telemetry schema or slice registry."""
        from backend.health.nci_governance_adapter import run_nci_p5_with_artifacts

        result = run_nci_p5_with_artifacts(
            repo_root=temp_dir,
            telemetry_schema_path=None,
            slice_registry_path=None,
            doc_patterns=["docs/system_law/*.md"],
        )

        assert result["mode"] == "DOC_ONLY"
        assert result["artifact_metadata"]["telemetry_schema_loaded"] is False
        assert result["artifact_metadata"]["slice_registry_loaded"] is False
        assert result["shadow_mode"] is True

    def test_telemetry_checked_mode_with_schema_only(
        self,
        sample_docs_dir: Path,
        sample_telemetry_schema: Path,
        temp_dir: Path,
    ) -> None:
        """TELEMETRY_CHECKED mode when telemetry schema but no slice registry."""
        from backend.health.nci_governance_adapter import run_nci_p5_with_artifacts

        result = run_nci_p5_with_artifacts(
            repo_root=temp_dir,
            telemetry_schema_path=sample_telemetry_schema,
            slice_registry_path=None,
            doc_patterns=["docs/system_law/*.md"],
        )

        assert result["mode"] == "TELEMETRY_CHECKED"
        assert result["artifact_metadata"]["telemetry_schema_loaded"] is True
        assert result["artifact_metadata"]["slice_registry_loaded"] is False

    def test_fully_bound_mode_with_both_files(
        self,
        sample_docs_dir: Path,
        sample_telemetry_schema: Path,
        sample_slice_registry: Path,
        temp_dir: Path,
    ) -> None:
        """FULLY_BOUND mode when both telemetry schema and slice registry present."""
        from backend.health.nci_governance_adapter import run_nci_p5_with_artifacts

        result = run_nci_p5_with_artifacts(
            repo_root=temp_dir,
            telemetry_schema_path=sample_telemetry_schema,
            slice_registry_path=sample_slice_registry,
            doc_patterns=["docs/system_law/*.md"],
        )

        assert result["mode"] == "FULLY_BOUND"
        assert result["artifact_metadata"]["telemetry_schema_loaded"] is True
        assert result["artifact_metadata"]["slice_registry_loaded"] is True

    def test_falls_back_to_doc_only_for_missing_schema_file(
        self,
        sample_docs_dir: Path,
        temp_dir: Path,
    ) -> None:
        """Falls back to DOC_ONLY when telemetry schema file doesn't exist."""
        from backend.health.nci_governance_adapter import run_nci_p5_with_artifacts

        result = run_nci_p5_with_artifacts(
            repo_root=temp_dir,
            telemetry_schema_path=temp_dir / "nonexistent_schema.json",
            slice_registry_path=None,
            doc_patterns=["docs/system_law/*.md"],
        )

        # File doesn't exist, so schema is None, mode is DOC_ONLY
        assert result["mode"] == "DOC_ONLY"
        assert result["artifact_metadata"]["telemetry_schema_loaded"] is False


class TestRunNciP5WithArtifactsViolationDetection:
    """Tests for violation detection in run_nci_p5_with_artifacts()."""

    def test_detects_tcl_violations_from_docs(
        self,
        sample_docs_dir: Path,
        temp_dir: Path,
    ) -> None:
        """Detects TCL violations from documentation files."""
        from backend.health.nci_governance_adapter import run_nci_p5_with_artifacts

        result = run_nci_p5_with_artifacts(
            repo_root=temp_dir,
            doc_patterns=["docs/system_law/tcl_violations.md"],
        )

        assert result["tcl_result"]["aligned"] is False
        violations = result["tcl_result"]["violations"]
        found_names = [v["found"] for v in violations]
        assert "Ht" in found_names or "RSI" in found_names

    def test_detects_sic_violations_from_docs(
        self,
        sample_docs_dir: Path,
        temp_dir: Path,
    ) -> None:
        """Detects SIC violations from documentation files."""
        from backend.health.nci_governance_adapter import run_nci_p5_with_artifacts

        result = run_nci_p5_with_artifacts(
            repo_root=temp_dir,
            doc_patterns=["docs/system_law/sic_violations.md"],
        )

        assert result["sic_result"]["aligned"] is False
        violations = result["sic_result"]["violations"]
        found_names = [v.get("found", "") for v in violations]
        assert "ArithmeticSimple" in found_names or "prop_taut" in found_names

    def test_detects_sic_004_overclaims(
        self,
        sample_docs_dir: Path,
        temp_dir: Path,
    ) -> None:
        """Detects SIC-004 capability overclaims."""
        from backend.health.nci_governance_adapter import run_nci_p5_with_artifacts

        result = run_nci_p5_with_artifacts(
            repo_root=temp_dir,
            doc_patterns=["docs/system_law/overclaims.md"],
        )

        sic_004_violations = [
            v for v in result["sic_result"]["violations"]
            if v.get("violation_type") == "SIC-004"
        ]
        assert len(sic_004_violations) >= 1

    def test_no_violations_for_canonical_docs(
        self,
        sample_docs_dir: Path,
        temp_dir: Path,
    ) -> None:
        """No violations for docs using only canonical names."""
        from backend.health.nci_governance_adapter import run_nci_p5_with_artifacts

        result = run_nci_p5_with_artifacts(
            repo_root=temp_dir,
            doc_patterns=["docs/system_law/canonical.md"],
        )

        assert result["tcl_result"]["aligned"] is True
        assert result["sic_result"]["aligned"] is True


# =============================================================================
# BUILD_NCI_P5_COMPACT_SIGNAL TESTS
# =============================================================================


class TestBuildNciP5CompactSignal:
    """Tests for build_nci_p5_compact_signal()."""

    def test_produces_compact_structure(
        self,
        sample_docs_dir: Path,
        temp_dir: Path,
    ) -> None:
        """Produces compact signal structure."""
        from backend.health.nci_governance_adapter import (
            run_nci_p5_with_artifacts,
            build_nci_p5_compact_signal,
        )

        result = run_nci_p5_with_artifacts(
            repo_root=temp_dir,
            doc_patterns=["docs/system_law/*.md"],
        )
        signal = build_nci_p5_compact_signal(result)

        # Verify required fields
        assert signal["schema_version"] == "1.0.0"
        assert signal["signal_type"] == "SIG-NAR"
        assert signal["mode"] == "DOC_ONLY"
        assert "global_nci" in signal
        assert "confidence" in signal
        assert "slo_status" in signal
        assert "recommendation" in signal
        assert "tcl_aligned" in signal
        assert "sic_aligned" in signal
        assert "tcl_violation_count" in signal
        assert "sic_violation_count" in signal
        assert "warning_count" in signal
        assert signal["shadow_mode"] is True

    def test_compact_signal_is_json_serializable(
        self,
        sample_docs_dir: Path,
        temp_dir: Path,
    ) -> None:
        """Compact signal is JSON serializable."""
        from backend.health.nci_governance_adapter import (
            run_nci_p5_with_artifacts,
            build_nci_p5_compact_signal,
        )

        result = run_nci_p5_with_artifacts(
            repo_root=temp_dir,
            doc_patterns=["docs/system_law/*.md"],
        )
        signal = build_nci_p5_compact_signal(result)

        json_str = json.dumps(signal, sort_keys=True)
        assert len(json_str) > 50

        parsed = json.loads(json_str)
        assert parsed["signal_type"] == "SIG-NAR"


# =============================================================================
# REAL REPO INTEGRATION TEST
# =============================================================================


class TestRealRepoIntegration:
    """Integration tests against actual repository docs."""

    def test_runs_against_real_repo(self, repo_root: Path) -> None:
        """Can run against actual repository documentation."""
        from backend.health.nci_governance_adapter import run_nci_p5_with_artifacts

        # Skip if not in a valid repo
        if not (repo_root / "docs").exists():
            pytest.skip("Not in a valid repository with docs/")

        result = run_nci_p5_with_artifacts(
            repo_root=repo_root,
            mock_global_nci=0.85,
        )

        # Should complete without error
        assert result["mode"] in ("DOC_ONLY", "TELEMETRY_CHECKED", "FULLY_BOUND")
        assert result["shadow_mode"] is True
        assert result["artifact_metadata"]["doc_count"] > 0

    def test_compact_signal_from_real_repo(self, repo_root: Path) -> None:
        """Can produce compact signal from real repo."""
        from backend.health.nci_governance_adapter import (
            run_nci_p5_with_artifacts,
            build_nci_p5_compact_signal,
        )

        # Skip if not in a valid repo
        if not (repo_root / "docs").exists():
            pytest.skip("Not in a valid repository with docs/")

        result = run_nci_p5_with_artifacts(repo_root=repo_root)
        signal = build_nci_p5_compact_signal(result)

        # Should produce valid compact signal
        assert signal["signal_type"] == "SIG-NAR"
        assert signal["shadow_mode"] is True
