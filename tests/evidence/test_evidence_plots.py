import json
from pathlib import Path

from tools.evidence_plots import (
    plot_delta_p,
    plot_omega_occupancy,
    plot_rsi,
)


def test_plot_delta_p_generates_svg(tmp_path):
    jsonl_path = tmp_path / "delta.jsonl"
    lines = [
        {"cycle": 2, "delta_p": 0.08},
        {"cycle": 1, "delta_p": -0.02},
        {"cycle": 3, "delta_p": 0.12},
    ]
    jsonl_path.write_text("\n".join(json.dumps(item) for item in lines))

    output = tmp_path / "delta.svg"
    plot_delta_p(str(jsonl_path), str(output))

    _assert_svg_created(output)


def test_plot_rsi_supports_vector_payload(tmp_path):
    json_path = tmp_path / "rsi.json"
    payload = {"cycles": [0, 1, 2], "rsi": [35, 48, 58]}
    json_path.write_text(json.dumps(payload))

    output = tmp_path / "rsi.svg"
    plot_rsi(str(json_path), str(output))

    _assert_svg_created(output)


def test_plot_omega_occupancy_handles_nested_data(tmp_path):
    json_path = tmp_path / "omega.json"
    payload = {
        "data": [
            {"cycle": 0, "omega_occupancy": 0.15},
            {"cycle": 1, "omega_occupancy": 0.22},
            {"cycle": 2, "omega_occupancy": 0.3},
        ]
    }
    json_path.write_text(json.dumps(payload))

    output = tmp_path / "omega.svg"
    plot_omega_occupancy(str(json_path), str(output))

    _assert_svg_created(output)


def _assert_svg_created(path: Path) -> None:
    assert path.exists(), "SVG file was not created"
    raw = path.read_text()
    assert "<svg" in raw, "SVG content appears to be empty"

