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


def test_tda_tile_is_validated_and_preserved() -> None:
    payload = {
        "status": "BLOCK",
        "fm_ok": False,
        "alignment_status": "CONCENTRATED",
        "coverage_pct": 42.0,
        "external_only_labels": 2,
        "schema_version": GLOBAL_HEALTH_SCHEMA_VERSION,
        "tda": {
            "schema_version": TDA_HEALTH_TILE_SCHEMA_VERSION,
            "tda_status": "ATTENTION",
            "block_rate": 0.12,
            "mean_hss": 0.44,
            "hss_trend": "STABLE",
            "governance_signal": "WARN",
            "notes": ["block_rate=12% elevated above 10%"],
        },
    }

    canonical = canonicalize_global_health(payload)
    parsed = json.loads(canonical)

    assert parsed["tda"]["schema_version"] == TDA_HEALTH_TILE_SCHEMA_VERSION
    assert parsed["tda"]["tda_status"] == "ATTENTION"
