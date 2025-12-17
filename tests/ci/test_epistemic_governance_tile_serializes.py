"""CI serialization tests for epistemic governance tile.

PHASE X — EPISTEMIC GOVERNANCE INTEGRATION

Tests that the epistemic governance tile:
- Builds from synthetic inputs
- Serializes to JSON successfully
- Contains all required fields
- Is deterministic across repeated runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.health.epistemic_alignment_adapter import (
    build_epistemic_governance_tile,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_alignment_tensor() -> Dict[str, Any]:
    """Create sample alignment tensor."""
    return {
        "schema_version": "1.0.0",
        "tensor_id": "test_tensor_123",
        "generated_at": "2025-01-01T00:00:00Z",
        "slice_axis": {
            "slice_easy_fo": 0.8,
            "slice_medium": 0.6,
            "slice_hard": 0.4,
        },
        "system_axes": {
            "semantic": 0.9,
            "curriculum": 0.8,
            "metrics": 0.7,
            "drift": 0.6,
        },
        "alignment_tensor_norm": 0.75,
        "misalignment_hotspots": ["slice_hard"],
    }


@pytest.fixture
def sample_misalignment_forecast() -> Dict[str, Any]:
    """Create sample misalignment forecast."""
    return {
        "schema_version": "1.0.0",
        "forecast_id": "test_forecast_123",
        "generated_at": "2025-01-01T00:00:00Z",
        "predicted_band": "MEDIUM",
        "confidence": 0.75,
        "time_to_drift_event": 10,
        "neutral_explanation": [
            "Current alignment tensor norm: 0.75",
            "Trend analysis based on 3 historical evaluations",
        ],
    }


@pytest.fixture
def sample_director_panel() -> Dict[str, Any]:
    """Create sample director panel."""
    return {
        "schema_version": "1.0.0",
        "panel_id": "test_panel_123",
        "generated_at": "2025-01-01T00:00:00Z",
        "status_light": "YELLOW",
        "alignment_band": "MEDIUM",
        "forecast_band": "MEDIUM",
        "structural_band": "MEDIUM",
        "headline": "Epistemic alignment shows mixed signals across domains.",
        "flags": [
            "Alignment tensor norm below threshold: 0.75",
            "Misalignment hotspots detected: 1 slice(s)",
        ],
    }


# ==============================================================================
# TESTS
# ==============================================================================


def test_tile_builds_from_synthetic_inputs(
    sample_alignment_tensor: Dict[str, Any],
    sample_misalignment_forecast: Dict[str, Any],
    sample_director_panel: Dict[str, Any],
) -> None:
    """Test that tile builds from synthetic alignment_tensor, forecast, and director_panel."""
    tile = build_epistemic_governance_tile(
        alignment_tensor=sample_alignment_tensor,
        misalignment_forecast=sample_misalignment_forecast,
        director_panel=sample_director_panel,
    )

    assert isinstance(tile, dict)
    assert "schema_version" in tile


def test_tile_serializes_to_json(
    sample_alignment_tensor: Dict[str, Any],
    sample_misalignment_forecast: Dict[str, Any],
    sample_director_panel: Dict[str, Any],
) -> None:
    """Test that json.dumps(tile) succeeds."""
    tile = build_epistemic_governance_tile(
        alignment_tensor=sample_alignment_tensor,
        misalignment_forecast=sample_misalignment_forecast,
        director_panel=sample_director_panel,
    )

    # Should not raise
    json_str = json.dumps(tile)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Should be able to parse back
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert parsed["schema_version"] == tile["schema_version"]


def test_tile_has_required_fields(
    sample_alignment_tensor: Dict[str, Any],
    sample_misalignment_forecast: Dict[str, Any],
    sample_director_panel: Dict[str, Any],
) -> None:
    """Test that status_light, alignment_band, forecast_band, tensor_norm, misalignment_hotspots, headline, flags, schema_version are present."""
    tile = build_epistemic_governance_tile(
        alignment_tensor=sample_alignment_tensor,
        misalignment_forecast=sample_misalignment_forecast,
        director_panel=sample_director_panel,
    )

    required_fields = {
        "status_light",
        "alignment_band",
        "forecast_band",
        "tensor_norm",
        "misalignment_hotspots",
        "headline",
        "flags",
        "schema_version",
    }

    assert required_fields.issubset(set(tile.keys()))

    # Verify types
    assert isinstance(tile["status_light"], str)
    assert tile["status_light"] in {"GREEN", "YELLOW", "RED"}
    assert isinstance(tile["alignment_band"], str)
    assert tile["alignment_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(tile["forecast_band"], str)
    assert tile["forecast_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(tile["tensor_norm"], (int, float))
    assert isinstance(tile["misalignment_hotspots"], list)
    assert isinstance(tile["headline"], str)
    assert isinstance(tile["flags"], list)
    assert tile["schema_version"] == "1.0.0"


def test_tile_is_deterministic(
    sample_alignment_tensor: Dict[str, Any],
    sample_misalignment_forecast: Dict[str, Any],
    sample_director_panel: Dict[str, Any],
) -> None:
    """Test that output is deterministic across repeated runs."""
    tile1 = build_epistemic_governance_tile(
        alignment_tensor=sample_alignment_tensor,
        misalignment_forecast=sample_misalignment_forecast,
        director_panel=sample_director_panel,
    )

    tile2 = build_epistemic_governance_tile(
        alignment_tensor=sample_alignment_tensor,
        misalignment_forecast=sample_misalignment_forecast,
        director_panel=sample_director_panel,
    )

    # Compare JSON serializations for exact match
    json1 = json.dumps(tile1, sort_keys=True)
    json2 = json.dumps(tile2, sort_keys=True)

    assert json1 == json2
    assert tile1 == tile2


def test_tile_extracts_correct_values(
    sample_alignment_tensor: Dict[str, Any],
    sample_misalignment_forecast: Dict[str, Any],
    sample_director_panel: Dict[str, Any],
) -> None:
    """Test that tile extracts correct values from inputs."""
    tile = build_epistemic_governance_tile(
        alignment_tensor=sample_alignment_tensor,
        misalignment_forecast=sample_misalignment_forecast,
        director_panel=sample_director_panel,
    )

    # status_light from director_panel
    assert tile["status_light"] == sample_director_panel["status_light"]

    # alignment_band derived from tensor_norm
    assert tile["alignment_band"] in {"LOW", "MEDIUM", "HIGH"}

    # forecast_band from misalignment_forecast
    assert tile["forecast_band"] == sample_misalignment_forecast["predicted_band"]

    # tensor_norm from alignment_tensor
    assert tile["tensor_norm"] == sample_alignment_tensor["alignment_tensor_norm"]

    # misalignment_hotspots from alignment_tensor
    assert tile["misalignment_hotspots"] == sample_alignment_tensor["misalignment_hotspots"]

    # headline from director_panel
    assert tile["headline"] == sample_director_panel["headline"]

    # flags from director_panel
    assert tile["flags"] == sample_director_panel["flags"]


def test_tile_handles_missing_optional_fields() -> None:
    """Test that tile handles missing optional fields gracefully."""
    minimal_alignment_tensor = {
        "alignment_tensor_norm": 0.5,
        "misalignment_hotspots": [],
    }
    minimal_forecast = {
        "predicted_band": "MEDIUM",
    }
    minimal_panel = {
        "status_light": "GREEN",
        "headline": "Test headline",
        "flags": [],
    }

    tile = build_epistemic_governance_tile(
        alignment_tensor=minimal_alignment_tensor,
        misalignment_forecast=minimal_forecast,
        director_panel=minimal_panel,
    )

    # Should still have all required fields
    assert "schema_version" in tile
    assert "status_light" in tile
    assert "alignment_band" in tile
    assert "forecast_band" in tile
    assert "tensor_norm" in tile
    assert "misalignment_hotspots" in tile
    assert "headline" in tile
    assert "flags" in tile

