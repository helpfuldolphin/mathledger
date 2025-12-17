import json

import matplotlib

matplotlib.use("Agg")

from tools.evidence_plots import plot_delta_p


def test_plot_delta_p_integration_creates_nonempty_svg(tmp_path):
    jsonl_path = tmp_path / "delta_p.jsonl"
    records = [
        {
            "cycle": idx + 1,
            "delta_p": (idx + 1) * 0.01,
            "flags": {"success_diverged": bool(idx % 2)},
        }
        for idx in range(10)
    ]
    jsonl_path.write_text("\n".join(json.dumps(record) for record in records))

    output_path = tmp_path / "delta_plot.svg"
    plot_delta_p(str(jsonl_path), str(output_path))

    assert output_path.exists()
    contents = output_path.read_text()
    assert "<svg" in contents
