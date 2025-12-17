import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from experiments import noise_guard_health


def test_noise_guard_health_script_outputs_sorted_json(tmp_path: Path) -> None:
    snapshot = {
        "window_size": 64,
        "window_id": 7,
        "channels": {"timeout": 0.02, "tier": 0.0, "queue": 0.0, "flip": 0.0},
        "epsilon_total": 0.05,
        "timeout_noisy": False,
        "unstable_buckets": [],
        "delta_h_bound": 0.12,
        "timeout_cusum": 0.01,
    }
    metrics_path = tmp_path / "snapshot.json"
    metrics_path.write_text(json.dumps(snapshot))

    argv = [
        "experiments/noise_guard_health.py",
        "--metrics",
        str(metrics_path),
        "--print",
    ]

    buf = io.StringIO()
    original_argv = noise_guard_health.sys.argv
    noise_guard_health.sys.argv = argv
    try:
        with redirect_stdout(buf):
            noise_guard_health.main()
    finally:
        noise_guard_health.sys.argv = original_argv

    payload = json.loads(buf.getvalue())
    assert "noise_guard" in payload
    tile = payload["noise_guard"]
    assert tile["summary"]["status"] in {"OK", "ATTENTION", "BLOCK"}

    canonical = json.dumps(payload, indent=2, sort_keys=True)
    assert buf.getvalue().strip() == canonical.strip()
