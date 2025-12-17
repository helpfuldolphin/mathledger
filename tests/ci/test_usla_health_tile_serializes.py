"""
Phase X P2: CI Smoke Test for USLA Health Tile Serialization

This test verifies that the USLA health tile can be produced and serialized
without error. It does NOT test governance logic, simulator semantics, or
CDI/invariant behavior.

SHADOW MODE CONTRACT:
- This test only verifies serialization and structural stability
- No governance decisions are tested or modified
- No simulator stepping or real cycles are run
- The test is purely for observability validation

Test requirements (per Phase X P2 spec):
1. Create a mock USLAIntegration in SHADOW mode (enabled=True)
2. Create a USLAHealthTileProducer bound to that integration
3. Call produce()
4. Assert: isinstance(tile, dict)
5. Assert: json.dumps(tile) does not raise
"""

from __future__ import annotations

import json
import tempfile
from typing import Any, Dict

import pytest


class TestUSLAHealthTileSerializes:
    """
    CI smoke tests for USLA health tile serialization.

    SHADOW MODE: These tests verify serialization only.
    No governance logic is tested.
    """

    def test_usla_tile_serializes_without_error(self) -> None:
        """
        Verify USLA health tile can be produced and serialized.

        This is the primary CI gate test per Phase X P2 spec.
        """
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_integration import USLAIntegration, RunnerType

        # 1. Create a mock USLAIntegration in SHADOW mode (enabled=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            integration = USLAIntegration.create_for_runner(
                runner_type=RunnerType.RFL,
                runner_id="ci_smoke_test",
                enabled=True,  # SHADOW mode enabled
                log_dir=tmpdir,
            )

            try:
                # 2. Create a USLAHealthTileProducer
                producer = USLAHealthTileProducer()

                # 3. Call produce_from_integration()
                tile = producer.produce_from_integration(integration)

                # 4. Assert: isinstance(tile, dict)
                assert tile is not None, "Tile should not be None for enabled integration"
                assert isinstance(tile, dict), f"Tile should be dict, got {type(tile)}"

                # 5. Assert: json.dumps(tile) does not raise
                json_str = json.dumps(tile)
                assert json_str is not None
                assert len(json_str) > 0

                # Verify round-trip
                parsed = json.loads(json_str)
                assert isinstance(parsed, dict)

            finally:
                integration.close()

    def test_usla_tile_has_required_fields(self) -> None:
        """Verify tile contains required fields per schema."""
        from backend.topology.usla_health_tile import (
            USLAHealthTileProducer,
            HEALTH_TILE_SCHEMA_VERSION,
        )
        from backend.topology.usla_simulator import USLAState

        producer = USLAHealthTileProducer()
        state = USLAState.initial()

        tile = producer.produce(state=state, hard_ok=True)

        # Required fields per Phase X P2 spec
        required_fields = [
            "schema_version",
            "tile_type",
            "timestamp",
            "state_summary",
            "hard_mode_status",
            "headline",
        ]

        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

        # Verify schema version
        assert tile["schema_version"] == HEALTH_TILE_SCHEMA_VERSION

    def test_usla_tile_state_summary_structure(self) -> None:
        """Verify state_summary contains expected fields."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_simulator import USLAState

        producer = USLAHealthTileProducer()
        state = USLAState.initial()

        tile = producer.produce(state=state, hard_ok=True)

        state_summary = tile.get("state_summary", {})
        expected_keys = ["H", "rho", "tau", "beta", "J", "C", "Gamma"]

        for key in expected_keys:
            assert key in state_summary, f"Missing state_summary key: {key}"

    def test_disabled_integration_returns_none(self) -> None:
        """Verify disabled integration produces None tile."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_integration import USLAIntegration, RunnerType

        producer = USLAHealthTileProducer()

        # Disabled integration
        integration = USLAIntegration.create_for_runner(
            runner_type=RunnerType.U2,
            runner_id="disabled_test",
            enabled=False,  # Disabled
        )

        tile = producer.produce_from_integration(integration)
        assert tile is None, "Disabled integration should return None tile"


class TestGlobalHealthSurfaceUSLAIntegration:
    """
    Tests for USLA tile integration with GlobalHealthSurface.

    SHADOW MODE: These tests verify the tile attachment mechanism only.
    """

    def test_set_usla_producer(self) -> None:
        """Verify set_usla_producer installs producer correctly."""
        from backend.health.global_surface import (
            set_usla_producer,
            clear_usla_producer,
            _usla_producer,
        )
        from backend.topology.usla_health_tile import USLAHealthTileProducer

        try:
            producer = USLAHealthTileProducer()
            set_usla_producer(producer)

            # Check module-level variable (import again to get updated value)
            from backend.health import global_surface
            assert global_surface._usla_producer is producer

        finally:
            clear_usla_producer()

    def test_build_global_health_surface_without_usla(self) -> None:
        """Verify build works without USLA producer configured."""
        import os
        from backend.health.global_surface import (
            build_global_health_surface,
            clear_usla_producer,
        )

        # Ensure clean state and SHADOW mode off
        clear_usla_producer()
        old_env = os.environ.get("USLA_SHADOW_ENABLED")
        os.environ["USLA_SHADOW_ENABLED"] = "false"

        try:
            payload = build_global_health_surface()

            assert isinstance(payload, dict)
            assert "schema_version" in payload
            assert "dynamics" in payload
            assert "usla" not in payload  # Should not be present

        finally:
            if old_env is not None:
                os.environ["USLA_SHADOW_ENABLED"] = old_env
            else:
                os.environ.pop("USLA_SHADOW_ENABLED", None)

    def test_build_global_health_surface_with_usla_shadow_enabled(self) -> None:
        """Verify USLA tile attached when SHADOW mode enabled."""
        import os
        from backend.health.global_surface import (
            build_global_health_surface,
            set_usla_producer,
            clear_usla_producer,
        )
        from backend.topology.usla_health_tile import USLAHealthTileProducer

        old_env = os.environ.get("USLA_SHADOW_ENABLED")
        os.environ["USLA_SHADOW_ENABLED"] = "true"

        try:
            producer = USLAHealthTileProducer()
            set_usla_producer(producer)

            payload = build_global_health_surface()

            assert isinstance(payload, dict)
            assert "usla" in payload, "USLA tile should be present when SHADOW enabled"

            usla_tile = payload["usla"]
            assert isinstance(usla_tile, dict)
            assert "headline" in usla_tile

            # Verify serializable
            json_str = json.dumps(payload)
            assert len(json_str) > 0

        finally:
            clear_usla_producer()
            if old_env is not None:
                os.environ["USLA_SHADOW_ENABLED"] = old_env
            else:
                os.environ.pop("USLA_SHADOW_ENABLED", None)

    def test_usla_tile_does_not_affect_dynamics(self) -> None:
        """Verify USLA tile presence doesn't change dynamics tile."""
        import os
        from backend.health.global_surface import (
            build_global_health_surface,
            set_usla_producer,
            clear_usla_producer,
        )
        from backend.topology.usla_health_tile import USLAHealthTileProducer

        old_env = os.environ.get("USLA_SHADOW_ENABLED")

        try:
            # Build without USLA
            os.environ["USLA_SHADOW_ENABLED"] = "false"
            clear_usla_producer()
            payload_without = build_global_health_surface()

            # Build with USLA
            os.environ["USLA_SHADOW_ENABLED"] = "true"
            producer = USLAHealthTileProducer()
            set_usla_producer(producer)
            payload_with = build_global_health_surface()

            # Dynamics should be identical
            assert payload_without["dynamics"] == payload_with["dynamics"]

        finally:
            clear_usla_producer()
            if old_env is not None:
                os.environ["USLA_SHADOW_ENABLED"] = old_env
            else:
                os.environ.pop("USLA_SHADOW_ENABLED", None)
