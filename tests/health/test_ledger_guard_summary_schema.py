import copy

import pytest

from scripts.build_ledger_guard_summary import build_summary_from_chain

jsonschema = pytest.importorskip("jsonschema")


LEDGER_GUARD_SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "status_light",
        "violation_count",
        "violation_counts",
        "headline",
        "generated_at",
    ],
    "properties": {
        "schema_version": {"type": "string", "enum": ["1.0.0"]},
        "status_light": {"type": "string", "enum": ["GREEN", "YELLOW", "RED", "UNKNOWN"]},
        "violation_count": {"type": "integer", "minimum": 0},
        "violation_counts": {"type": "integer", "minimum": 0},
        "headline": {"type": "string"},
        "generated_at": {"type": "string"},
    },
}


def _sample_headers():
    return [
        {
            "height": 0,
            "prev_hash": "0" * 64,
            "root_hash": "1" * 64,
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "height": 0,
            "prev_hash": "1" * 64,
            "root_hash": "2" * 64,
            "timestamp": "2024-01-01T00:01:00Z",
        },
    ]


def test_ledger_guard_summary_schema_accepts_expected_shape():
    summary = build_summary_from_chain(_sample_headers())
    jsonschema.validate(instance=summary, schema=LEDGER_GUARD_SUMMARY_SCHEMA)
    assert summary["violation_counts"] == summary["violation_count"]


def test_ledger_guard_summary_schema_rejects_extra_fields():
    summary = build_summary_from_chain(_sample_headers())
    mutated = copy.deepcopy(summary)
    mutated["unexpected_field"] = "nope"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=mutated, schema=LEDGER_GUARD_SUMMARY_SCHEMA)


def test_ledger_guard_summary_schema_requires_keys():
    summary = build_summary_from_chain(_sample_headers())
    mutated = copy.deepcopy(summary)
    mutated.pop("headline", None)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=mutated, schema=LEDGER_GUARD_SUMMARY_SCHEMA)
