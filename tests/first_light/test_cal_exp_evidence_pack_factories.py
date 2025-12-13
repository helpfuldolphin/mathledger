from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.factories.first_light_factories import (
    emit_cal_exp1_report_for_evidence_pack,
    emit_cal_exp_reports_for_evidence_pack,
    make_cal_exp1_report,
)


_CAL_EXP1_REPORT_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version",
        "mode",
        "run_id",
        "timestamp",
        "total_cycles",
        "window_size",
        "window_count",
        "windows",
        "summary",
        "provisional_verdict",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "mode": {"const": "SHADOW"},
        "run_id": {"type": "string"},
        "timestamp": {"type": "string"},
        "total_cycles": {"type": "integer", "minimum": 0},
        "window_size": {"type": "integer", "minimum": 1},
        "window_count": {"type": "integer", "minimum": 1},
        "windows": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "window_index",
                    "window_start",
                    "window_end",
                    "cycles_in_window",
                    "divergence_count",
                    "divergence_rate",
                    "mean_delta_p",
                    "delta_bias",
                    "delta_variance",
                    "phase_lag_xcorr",
                    "pattern_tag",
                ],
                "properties": {
                    "window_index": {"type": "integer", "minimum": 0},
                    "window_start": {"type": "integer", "minimum": 0},
                    "window_end": {"type": "integer", "minimum": -1},
                    "cycles_in_window": {"type": "integer", "minimum": 0},
                    "divergence_count": {"type": "integer", "minimum": 0},
                    "divergence_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "mean_delta_p": {"type": "number"},
                    "delta_bias": {"type": "number"},
                    "delta_variance": {"type": "number", "minimum": 0.0},
                    "phase_lag_xcorr": {"type": "number"},
                    "pattern_tag": {"type": "string"},
                },
            },
        },
        "summary": {
            "type": "object",
            "required": [
                "final_divergence_rate",
                "final_delta_bias",
                "mean_divergence_over_run",
                "pattern_progression",
            ],
            "properties": {
                "final_divergence_rate": {"type": ["number", "null"]},
                "final_delta_bias": {"type": ["number", "null"]},
                "mean_divergence_over_run": {"type": ["number", "null"]},
                "pattern_progression": {"type": "array", "items": {"type": "string"}},
            },
        },
        "provisional_verdict": {
            "type": "object",
            "required": ["verdict", "reason", "enforcement"],
            "properties": {
                "verdict": {"type": "string"},
                "reason": {"type": "string"},
                "enforcement": {"const": "SHADOW_ONLY"},
            },
        },
    },
}

_CAL_EXP2_REPORT_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "mode", "generated_at", "timestamp", "trials"],
    "properties": {
        "schema_version": {"const": "0.1.0"},
        "mode": {"const": "SHADOW"},
        "generated_at": {"type": "string"},
        "timestamp": {"type": "string"},
        "trials": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["lr", "divergence_trajectory"],
                "properties": {
                    "lr": {"type": "number"},
                    "divergence_trajectory": {"type": "array", "items": {"type": "number"}},
                },
            },
        },
    },
}

_CAL_EXP3_REPORT_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version",
        "mode",
        "generated_at",
        "params",
        "pre_change",
        "post_change",
        "timestamp",
    ],
    "properties": {
        "schema_version": {"const": "0.1.0"},
        "mode": {"const": "SHADOW"},
        "generated_at": {"type": "string"},
        "timestamp": {"type": "string"},
        "params": {
            "type": "object",
            "required": ["cycles", "change_after", "delta_H", "seed"],
            "properties": {
                "cycles": {"type": "integer"},
                "change_after": {"type": "integer"},
                "delta_H": {"type": "number"},
                "seed": {"type": "integer"},
            },
        },
        "pre_change": {
            "type": "object",
            "required": ["divergence_rate", "sample_size"],
            "properties": {
                "divergence_rate": {"type": "number"},
                "sample_size": {"type": "integer"},
            },
        },
        "post_change": {
            "type": "object",
            "required": ["divergence_rate", "sample_size"],
            "properties": {
                "divergence_rate": {"type": "number"},
                "sample_size": {"type": "integer"},
            },
        },
    },
}


def test_emit_cal_exp_reports_for_evidence_pack_writes_expected_layout(tmp_path: Path) -> None:
    paths = emit_cal_exp_reports_for_evidence_pack(tmp_path, seed=123)

    assert paths["cal_exp1_report"] == tmp_path / "calibration" / "cal_exp1_report.json"
    assert paths["cal_exp2_report"] == tmp_path / "calibration" / "cal_exp2_report.json"
    assert paths["cal_exp3_report"] == tmp_path / "calibration" / "cal_exp3_report.json"

    for path in paths.values():
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" in payload

    payload_1 = json.loads(paths["cal_exp1_report"].read_text(encoding="utf-8"))
    payload_2 = json.loads(paths["cal_exp2_report"].read_text(encoding="utf-8"))
    payload_3 = json.loads(paths["cal_exp3_report"].read_text(encoding="utf-8"))

    assert payload_1["schema_version"] == "1.0.0"
    assert payload_1["mode"] == "SHADOW"
    assert payload_2["schema_version"] == "0.1.0"
    assert payload_2["mode"] == "SHADOW"
    assert payload_3["schema_version"] == "0.1.0"
    assert payload_3["mode"] == "SHADOW"


def test_cal_exp_reports_validate_against_evidence_pack_schemas_when_jsonschema_present(
    tmp_path: Path,
) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    paths = emit_cal_exp_reports_for_evidence_pack(tmp_path, seed=123)
    payload_1 = json.loads(paths["cal_exp1_report"].read_text(encoding="utf-8"))
    payload_2 = json.loads(paths["cal_exp2_report"].read_text(encoding="utf-8"))
    payload_3 = json.loads(paths["cal_exp3_report"].read_text(encoding="utf-8"))

    jsonschema.validate(instance=payload_1, schema=_CAL_EXP1_REPORT_SCHEMA)
    jsonschema.validate(instance=payload_2, schema=_CAL_EXP2_REPORT_SCHEMA)
    jsonschema.validate(instance=payload_3, schema=_CAL_EXP3_REPORT_SCHEMA)


def test_emit_cal_exp1_report_for_evidence_pack_matches_factory_payload(tmp_path: Path) -> None:
    expected = make_cal_exp1_report(cycles=200, window_size=50, seed=42)
    path = emit_cal_exp1_report_for_evidence_pack(tmp_path, cycles=200, window_size=50, seed=42)
    assert json.loads(path.read_text(encoding="utf-8")) == expected
