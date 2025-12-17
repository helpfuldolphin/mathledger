"""E2E tests for identity preflight extraction_source in status generator.

Tests the full CLI-to-output flow for extraction source hierarchy:
CLI > MANIFEST > LEGACY_FILE > RUN_CONFIG > MISSING

SHADOW MODE CONTRACT:
- All outputs are observational only
- No gating or enforcement logic
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


class TestExtractionSourceCLIOverride:
    """Test CLI --identity-preflight takes highest priority."""

    def test_cli_override_extraction_source(self, tmp_path: Path) -> None:
        """CLI identity preflight should set extraction_source=CLI."""
        from scripts.generate_first_light_status import generate_status

        # CLI identity takes priority even with manifest present
        manifest = {
            "governance": {
                "slice_identity": {
                    "p5_preflight_reference": {
                        "status": "OK",
                        "sha256": "abc123",
                    }
                }
            }
        }

        cli_identity = {"status": "INVESTIGATE", "fingerprint_match": False}

        status = generate_status(
            manifest=manifest,
            cli_identity=cli_identity,
        )

        identity = status["signals"]["p5_identity_preflight"]
        assert identity["extraction_source"] == "CLI"
        assert identity["status"] == "INVESTIGATE"
        assert identity["mode"] == "SHADOW"

    def test_cli_override_with_legacy_file(self, tmp_path: Path) -> None:
        """CLI should take priority over legacy file."""
        from scripts.generate_first_light_status import generate_status

        # Create legacy file
        preflight_path = tmp_path / "p5_identity_preflight.json"
        preflight_path.write_text(json.dumps({"status": "OK"}))

        cli_identity = {"status": "BLOCK"}

        status = generate_status(
            results_dir=tmp_path,
            cli_identity=cli_identity,
        )

        identity = status["signals"]["p5_identity_preflight"]
        assert identity["extraction_source"] == "CLI"
        assert identity["status"] == "BLOCK"


class TestExtractionSourceManifest:
    """Test MANIFEST extraction source."""

    def test_manifest_extraction_source(self, tmp_path: Path) -> None:
        """Manifest governance.slice_identity.p5_preflight_reference extraction."""
        from scripts.generate_first_light_status import generate_status

        manifest = {
            "governance": {
                "slice_identity": {
                    "p5_preflight_reference": {
                        "status": "OK",
                        "sha256": "abc123def456",
                        "fingerprint_match": True,
                    }
                }
            }
        }

        status = generate_status(manifest=manifest)

        identity = status["signals"]["p5_identity_preflight"]
        assert identity["extraction_source"] == "MANIFEST"
        assert identity["status"] == "OK"
        assert identity["sha256"] == "abc123def456"


class TestExtractionSourceLegacyFile:
    """Test LEGACY_FILE extraction source."""

    def test_legacy_file_extraction_source(self, tmp_path: Path) -> None:
        """p5_identity_preflight.json in results_dir extraction."""
        from scripts.generate_first_light_status import generate_status

        # Create legacy file
        preflight_path = tmp_path / "p5_identity_preflight.json"
        preflight_data = {"status": "INVESTIGATE", "fingerprint_match": False}
        preflight_path.write_text(json.dumps(preflight_data))

        status = generate_status(results_dir=tmp_path)

        identity = status["signals"]["p5_identity_preflight"]
        assert identity["extraction_source"] == "LEGACY_FILE"
        assert identity["status"] == "INVESTIGATE"
        assert "sha256" in identity  # File hash computed


class TestExtractionSourceRunConfig:
    """Test RUN_CONFIG extraction source."""

    def test_run_config_extraction_source(self, tmp_path: Path) -> None:
        """run_config.json identity_preflight field extraction."""
        from scripts.generate_first_light_status import generate_status

        # Create run_config with identity_preflight
        config_path = tmp_path / "run_config.json"
        config_data = {
            "telemetry_adapter": "mock",
            "identity_preflight": {
                "status": "OK",
                "fingerprint_match": True,
            }
        }
        config_path.write_text(json.dumps(config_data))

        status = generate_status(results_dir=tmp_path)

        identity = status["signals"]["p5_identity_preflight"]
        assert identity["extraction_source"] == "RUN_CONFIG"
        assert identity["status"] == "OK"


class TestExtractionSourceMissing:
    """Test MISSING extraction source fallback."""

    def test_missing_extraction_source(self, tmp_path: Path) -> None:
        """No identity source found -> MISSING with OK default."""
        from scripts.generate_first_light_status import generate_status

        # Empty results dir, no manifest, no CLI
        status = generate_status(results_dir=tmp_path)

        identity = status["signals"]["p5_identity_preflight"]
        assert identity["extraction_source"] == "MISSING"
        assert identity["status"] == "OK"  # Default
        assert identity["mode"] == "SHADOW"


class TestWarningsWithExtractionSource:
    """Test warnings include extraction_source context."""

    def test_block_warning_includes_source(self, tmp_path: Path) -> None:
        """BLOCK status warning includes extraction source."""
        from scripts.generate_first_light_status import generate_status

        cli_identity = {"status": "BLOCK"}
        status = generate_status(cli_identity=cli_identity)

        warnings = status["warnings"]
        # Should have exactly one identity warning with source info
        identity_warnings = [w for w in warnings if "identity" in w.lower()]
        assert len(identity_warnings) == 1
        assert "source=CLI" in identity_warnings[0]
        assert "BLOCK" in identity_warnings[0]
        assert "(advisory)" in identity_warnings[0]

    def test_investigate_warning_includes_source(self, tmp_path: Path) -> None:
        """INVESTIGATE status warning includes extraction source."""
        from scripts.generate_first_light_status import generate_status

        # Create legacy file with INVESTIGATE
        preflight_path = tmp_path / "p5_identity_preflight.json"
        preflight_path.write_text(json.dumps({"status": "INVESTIGATE"}))

        status = generate_status(results_dir=tmp_path)

        warnings = status["warnings"]
        identity_warnings = [w for w in warnings if "identity" in w.lower()]
        assert len(identity_warnings) == 1
        assert "source=LEGACY_FILE" in identity_warnings[0]
        assert "INVESTIGATE" in identity_warnings[0]

    def test_ok_status_no_warning(self, tmp_path: Path) -> None:
        """OK status should not generate identity warning."""
        from scripts.generate_first_light_status import generate_status

        cli_identity = {"status": "OK"}
        status = generate_status(cli_identity=cli_identity)

        warnings = status["warnings"]
        identity_warnings = [w for w in warnings if "identity" in w.lower()]
        assert len(identity_warnings) == 0


class TestExtractionSourceHierarchy:
    """Test full extraction source hierarchy precedence."""

    def test_hierarchy_cli_over_all(self, tmp_path: Path) -> None:
        """CLI > MANIFEST > LEGACY_FILE > RUN_CONFIG."""
        from scripts.generate_first_light_status import generate_status

        # Create all sources
        manifest = {
            "governance": {
                "slice_identity": {
                    "p5_preflight_reference": {"status": "INVESTIGATE"}
                }
            }
        }

        preflight_path = tmp_path / "p5_identity_preflight.json"
        preflight_path.write_text(json.dumps({"status": "BLOCK"}))

        config_path = tmp_path / "run_config.json"
        config_path.write_text(json.dumps({
            "identity_preflight": {"status": "OK"}
        }))

        cli_identity = {"status": "CLI_STATUS"}

        status = generate_status(
            results_dir=tmp_path,
            manifest=manifest,
            cli_identity=cli_identity,
        )

        identity = status["signals"]["p5_identity_preflight"]
        assert identity["extraction_source"] == "CLI"
        assert identity["status"] == "CLI_STATUS"

    def test_hierarchy_manifest_over_file(self, tmp_path: Path) -> None:
        """MANIFEST takes priority over LEGACY_FILE."""
        from scripts.generate_first_light_status import generate_status

        manifest = {
            "governance": {
                "slice_identity": {
                    "p5_preflight_reference": {"status": "MANIFEST_STATUS"}
                }
            }
        }

        preflight_path = tmp_path / "p5_identity_preflight.json"
        preflight_path.write_text(json.dumps({"status": "FILE_STATUS"}))

        status = generate_status(
            results_dir=tmp_path,
            manifest=manifest,
        )

        identity = status["signals"]["p5_identity_preflight"]
        assert identity["extraction_source"] == "MANIFEST"
        assert identity["status"] == "MANIFEST_STATUS"

    def test_hierarchy_file_over_config(self, tmp_path: Path) -> None:
        """LEGACY_FILE takes priority over RUN_CONFIG."""
        from scripts.generate_first_light_status import generate_status

        preflight_path = tmp_path / "p5_identity_preflight.json"
        preflight_path.write_text(json.dumps({"status": "FILE_STATUS"}))

        config_path = tmp_path / "run_config.json"
        config_path.write_text(json.dumps({
            "identity_preflight": {"status": "CONFIG_STATUS"}
        }))

        status = generate_status(results_dir=tmp_path)

        identity = status["signals"]["p5_identity_preflight"]
        assert identity["extraction_source"] == "LEGACY_FILE"
        assert identity["status"] == "FILE_STATUS"
