import json

import pytest

from backend.health.canonicalize import canonicalize_global_health
from backend.health.global_schema import (
    GLOBAL_HEALTH_SCHEMA_VERSION,
    SchemaValidationError,
    validate_global_health,
)
from backend.health.tda_adapter import TDA_HEALTH_TILE_SCHEMA_VERSION


def test_canonicalize_global_health_sorts_and_compacts_keys() -> None:
    payload = {
        "status": "OK",
        "fm_ok": True,
        "alignment_status": "SPARSE",
        "coverage_pct": 12.5,
        "external_only_labels": 0,
        "schema_version": GLOBAL_HEALTH_SCHEMA_VERSION,
    }

    canonical = canonicalize_global_health(payload)

    expected = (
        '{"alignment_status":"SPARSE",'
        '"coverage_pct":12.5,'
        '"external_only_labels":0,'
        '"fm_ok":true,'
        f'"schema_version":"{GLOBAL_HEALTH_SCHEMA_VERSION}",'
        '"status":"OK"}'
    )
    assert canonical == expected


def test_schema_violation_requires_block_status() -> None:
    payload = {
        "status": "WARN",
        "fm_ok": False,
        "alignment_status": "CONCENTRATED",
        "coverage_pct": 50.0,
        "external_only_labels": 2,
        "schema_version": GLOBAL_HEALTH_SCHEMA_VERSION,
    }

    with pytest.raises(SchemaValidationError):
        validate_global_health(payload)


def test_canonicalization_preserves_tda_and_golden_tiles() -> None:
    payload = {
        "status": "BLOCK",
        "fm_ok": False,
        "alignment_status": "WELL_DISTRIBUTED",
        "coverage_pct": 88.4,
        "external_only_labels": 1,
        "schema_version": GLOBAL_HEALTH_SCHEMA_VERSION,
        "tda": {
            "schema_version": TDA_HEALTH_TILE_SCHEMA_VERSION,
            "tda_status": "ATTENTION",
            "block_rate": 0.08,
            "mean_hss": 0.67,
            "hss_trend": "STABLE",
            "governance_signal": "WARN",
            "notes": ["block_rate=8% elevated above 10%", "governance WARN"],
        },
        "golden_runs": {
            "schema_version": "golden-runs-1.0.0",
            "status": "OK",
        },
    }

    canonical = canonicalize_global_health(payload)
    parsed = json.loads(canonical)

    assert parsed["tda"]["schema_version"] == TDA_HEALTH_TILE_SCHEMA_VERSION
    assert parsed["golden_runs"]["status"] == "OK"

    order = [
        '"alignment_status"',
        '"coverage_pct"',
        '"external_only_labels"',
        '"fm_ok"',
        '"golden_runs"',
        '"schema_version"',
        '"status"',
        '"tda"',
    ]
    indices = [canonical.index(token) for token in order]
    assert indices == sorted(indices), "top-level keys must be sorted"
