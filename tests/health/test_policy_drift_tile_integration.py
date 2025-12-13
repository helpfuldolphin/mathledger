import json
from pathlib import Path

from backend.health.policy_drift_tile import (
    attach_policy_drift_tile,
    build_policy_drift_summary,
)
from scripts.policy_drift_lint import summarize_policy_drift_for_global_health


def _validate_against_schema(instance: dict, schema: dict) -> None:
    assert schema["type"] == "object"
    assert isinstance(instance, dict)

    required_fields = schema.get("required", [])
    for field in required_fields:
        assert field in instance, f"missing required field {field}"

    properties = schema.get("properties", {})
    allow_additional = schema.get("additionalProperties", True)
    if not allow_additional:
        extra = set(instance.keys()) - set(properties.keys())
        assert not extra, f"unexpected properties present: {sorted(extra)}"

    for key, subschema in properties.items():
        if key not in instance:
            continue
        value = instance[key]
        schema_type = subschema.get("type")
        if schema_type == "string":
            assert isinstance(value, str), f"{key} must be a string"
            if "enum" in subschema:
                assert value in subschema["enum"], f"{key} outside enum"
        elif schema_type == "boolean":
            assert isinstance(value, bool), f"{key} must be a boolean"
        elif schema_type == "number":
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"{key} must be numeric"
            )


def test_policy_drift_tile_attachment_and_schema_validation() -> None:
    base_health = {
        "status": "ok",
        "components": ["policy"],
        "metadata": {"run_id": "test-run"},
    }
    report = {
        "status": "WARN",
        "breaking_changes": [],
        "soft_changes": [
            {
                "category": "learning_rates",
                "path": "trainer.lr",
                "change": "changed",
                "old": 0.1,
                "new": 0.2,
            }
        ],
    }

    tile = summarize_policy_drift_for_global_health(report)
    updated_health = attach_policy_drift_tile(base_health, tile)

    assert base_health is not updated_health
    assert "policy_drift" in updated_health
    assert updated_health["policy_drift"] == tile
    assert updated_health["signals"]["policy_drift"] == build_policy_drift_summary(tile)
    assert "policy_drift" not in base_health

    schema = json.loads(Path("schemas/policy_drift_tile.schema.json").read_text(encoding="utf-8"))
    _validate_against_schema(tile, schema)

    serialized = json.dumps(updated_health)
    assert isinstance(serialized, str)
