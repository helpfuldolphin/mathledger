"""
Tests for P3 Pathology Annotation Schema.

These tests ensure the optional P3 pathology annotation is valid and that
`jsonschema.validate()` can run against the schema when jsonschema is installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

pytestmark = pytest.mark.unit

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture
def schema() -> Dict[str, Any]:
    schema_path = (
        Path(__file__).parent.parent.parent
        / "docs"
        / "system_law"
        / "schemas"
        / "evidence_pack"
        / "p3_pathology.schema.json"
    )
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_valid_pathology_annotation_validates(schema: Dict[str, Any]) -> None:
    record = {
        "pathology": "spike",
        "pathology_params": {"magnitude": 0.75, "at": 3},
    }
    jsonschema.validate(instance=record, schema=schema)


def test_invalid_pathology_enum_rejected(schema: Dict[str, Any]) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={"pathology": "alien"}, schema=schema)


def test_additional_properties_rejected(schema: Dict[str, Any]) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={"unexpected": True}, schema=schema)

