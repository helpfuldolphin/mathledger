#!/usr/bin/env python3
"""Phase X P3 demo orchestrator.

Runs the real First Light harness in a temp directory, builds quick evidence
plots, and prints the SVG locations plus a short metrics summary.

Policy: this demo intentionally invokes the canonical harness only (no legacy
wrapper fallback) to avoid accidental dependence on deprecated entrypoints and
to preserve audit visibility.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import plotting
from rfl.prng.governance import build_first_light_prng_summary

REAL_HARNESS_SCRIPT = Path(__file__).parent / "usla_first_light_harness.py"


def run_first_light_harness(
    base_output_dir: Path,
    *,
    cycles: int = 200,
    seed: int = 42,
    pathology: str = "none",
) -> Path:
    """Invoke the real First Light harness and return the run directory."""
    base_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REAL_HARNESS_SCRIPT),
        "--cycles",
        str(cycles),
        "--seed",
        str(seed),
        "--output-dir",
        str(base_output_dir),
        "--pathology",
        pathology,
    ]

    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "[p3-demo] Real harness invocation failed (no legacy fallback).\n"
            f"[p3-demo] Command: {' '.join(cmd)}\n"
            f"[p3-demo] Failure reason: {exc}\n"
            "[p3-demo] Canonical entrypoint: uv run python scripts/usla_first_light_harness.py ..."
        ) from exc

    run_dirs = sorted(
        [p for p in base_output_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {base_output_dir}")
    return run_dirs[0]


def load_windows(run_dir: Path) -> list[dict]:
    """Load windowed metrics produced by the harness."""
    windows_path = run_dir / "metrics_windows.json"
    data = json.loads(windows_path.read_text())
    windows = data.get("windows", [])
    if not windows:
        raise ValueError(f"No windows found in {windows_path}")
    return windows


def load_summary_metrics(run_dir: Path) -> Tuple[float, float]:
    """Extract success rate and omega occupancy from stability_report.json."""
    report_path = run_dir / "stability_report.json"
    report = json.loads(report_path.read_text())
    metrics = report.get("metrics", {})
    success_rate = metrics.get("success_rate")
    omega_occupancy = metrics.get("omega", {}).get("occupancy_rate")
    if success_rate is None or omega_occupancy is None:
        raise ValueError(f"Missing success_rate or omega occupancy in {report_path}")
    return float(success_rate), float(omega_occupancy)


def load_prng_summary(run_dir: Path) -> dict:
    """Build a PRNG summary using the demo helper and recorded seed."""
    run_config_path = run_dir / "run_config.json"
    seed = None
    if run_config_path.exists():
        config = json.loads(run_config_path.read_text())
        seed = config.get("seed")
    return build_first_light_prng_summary(seed=seed)


def load_run_metadata(run_dir: Path) -> Dict[str, str]:
    """Load minimal run metadata for reporting."""
    meta = {"run_id": run_dir.name, "pathology": "unknown", "seed": "unknown"}
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        config = json.loads(run_config_path.read_text())
        args = config.get("args", {})
        meta["run_id"] = config.get("run_id", meta["run_id"])
        meta["pathology"] = args.get("pathology", config.get("pathology", meta["pathology"]))
        meta["seed"] = args.get("seed", meta["seed"])
    return meta


def generate_plots(run_dir: Path) -> list[Path]:
    """Create SVG plots (delta-p proxy, RSI, omega occupancy) from windowed metrics."""
    plotting.setup_style()

    windows = load_windows(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _end_cycle(window: dict, fallback_index: int) -> int:
        return int(window.get("end_cycle", window.get("start_cycle", fallback_index)))

    delta_cycles: list[int] = []
    delta_values: list[float] = []
    rsi_cycles: list[int] = []
    rsi_values: list[float] = []
    omega_cycles: list[int] = []
    omega_values: list[float] = []

    for idx, window in enumerate(windows):
        metrics = window.get("metrics", {})
        cycle_x = _end_cycle(window, idx)

        success_rate = metrics.get("success_rate")
        abstention_rate = metrics.get("abstention_rate")
        if success_rate is not None and abstention_rate is not None:
            delta_cycles.append(cycle_x)
            delta_values.append(float(success_rate) - float(abstention_rate))

        mean_rsi = metrics.get("mean_rsi")
        if mean_rsi is not None:
            rsi_cycles.append(cycle_x)
            rsi_values.append(float(mean_rsi))

        omega_occ = metrics.get("omega_occupancy")
        if omega_occ is not None:
            omega_cycles.append(cycle_x)
            omega_values.append(float(omega_occ))

    svg_paths: list[Path] = []

    fig_delta, ax_delta = plt.subplots()
    ax_delta.plot(delta_cycles, delta_values, color="black", marker="o", linewidth=2.0)
    ax_delta.set_xlabel("Cycle")
    ax_delta.set_ylabel("Delta-p (success - abstention)")
    ax_delta.set_title("Delta-p Trajectory (windowed)")
    ax_delta.grid(True, alpha=0.3, linewidth=0.5)
    delta_svg = plot_dir / "delta_p.svg"
    fig_delta.savefig(delta_svg, format="svg", bbox_inches="tight")
    plt.close(fig_delta)
    svg_paths.append(delta_svg)

    fig_rsi, ax_rsi = plt.subplots()
    ax_rsi.plot(rsi_cycles, rsi_values, color="black", marker="o", linewidth=2.0)
    ax_rsi.set_xlabel("Cycle")
    ax_rsi.set_ylabel("RSI (rho)")
    ax_rsi.set_title("RSI Trajectory (windowed)")
    ax_rsi.grid(True, alpha=0.3, linewidth=0.5)
    rsi_svg = plot_dir / "rsi.svg"
    fig_rsi.savefig(rsi_svg, format="svg", bbox_inches="tight")
    plt.close(fig_rsi)
    svg_paths.append(rsi_svg)

    fig_omega, ax_omega = plt.subplots()
    ax_omega.plot(omega_cycles, omega_values, color="black", marker="o", linewidth=2.0)
    ax_omega.set_xlabel("Cycle")
    ax_omega.set_ylabel("Omega occupancy")
    ax_omega.set_title("Omega Occupancy Trajectory (windowed)")
    ax_omega.grid(True, alpha=0.3, linewidth=0.5)
    omega_svg = plot_dir / "omega_occupancy.svg"
    fig_omega.savefig(omega_svg, format="svg", bbox_inches="tight")
    plt.close(fig_omega)
    svg_paths.append(omega_svg)

    return svg_paths


def generate_tda_plots(run_dir: Path) -> list[Path]:
    """Plot TDA metrics trajectories if available."""
    tda_path = run_dir / "tda_metrics.json"
    if not tda_path.exists():
        return []

    payload = json.loads(tda_path.read_text())
    series = payload.get("metrics", [])
    if not series:
        return []

    window_indices = [int(entry.get("window_index", idx)) for idx, entry in enumerate(series)]
    metrics = {
        "SNS": [float(entry.get("SNS", 0.0)) for entry in series],
        "PCS": [float(entry.get("PCS", 0.0)) for entry in series],
        "DRS": [float(entry.get("DRS", 0.0)) for entry in series],
        "HSS": [float(entry.get("HSS", 0.0)) for entry in series],
    }

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for name, values in metrics.items():
        ax.plot(window_indices, values, marker="o", linewidth=2.0, label=name)
    ax.set_xlabel("Window index")
    ax.set_ylabel("TDA metric value")
    ax.set_title("TDA Metrics Trajectories")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend()

    tda_svg = plot_dir / "tda_metrics.svg"
    fig.savefig(tda_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    return [tda_svg]


def _series_from_windows(windows: Iterable[dict], metric_key: str) -> Tuple[list[int], list[float]]:
    cycles: list[int] = []
    values: list[float] = []
    for idx, window in enumerate(windows):
        metrics = window.get("metrics", {})
        value = metrics.get(metric_key)
        if value is None:
            continue
        cycle_x = int(window.get("end_cycle", window.get("start_cycle", idx)))
        cycles.append(cycle_x)
        values.append(float(value))
    return cycles, values


def generate_comparison_plots(run_a: Path, run_b: Path, label_a: str, label_b: str) -> list[Path]:
    """Create overlay SVGs comparing two runs for delta-p proxy, RSI, and omega occupancy."""
    plotting.setup_style()
    windows_a = load_windows(run_a)
    windows_b = load_windows(run_b)

    comp_dir = run_a.parent / "comparison_plots"
    comp_dir.mkdir(parents=True, exist_ok=True)

    svg_paths: list[Path] = []

    # Delta-p proxy
    def _delta_series(windows: list[dict]) -> Tuple[list[int], list[float]]:
        cycles: list[int] = []
        values: list[float] = []
        for idx, window in enumerate(windows):
            metrics = window.get("metrics", {})
            success = metrics.get("success_rate")
            abstention = metrics.get("abstention_rate")
            if success is None or abstention is None:
                continue
            cycle_x = int(window.get("end_cycle", window.get("start_cycle", idx)))
            cycles.append(cycle_x)
            values.append(float(success) - float(abstention))
        return cycles, values

    series = {
        "delta_p": (_delta_series, "Delta-p (success - abstention)", "delta_p_comparison.svg"),
        "rsi": (lambda w: _series_from_windows(w, "mean_rsi"), "RSI (rho)", "rsi_comparison.svg"),
        "omega": (lambda w: _series_from_windows(w, "omega_occupancy"), "Omega occupancy", "omega_comparison.svg"),
    }

    for _, (builder, ylabel, filename) in series.items():
        cycles_a, values_a = builder(windows_a)
        cycles_b, values_b = builder(windows_b)

        fig, ax = plt.subplots()
        ax.plot(cycles_a, values_a, color="black", marker="o", linewidth=2.0, label=label_a)
        ax.plot(cycles_b, values_b, color="red", marker="s", linewidth=2.0, linestyle="--", label=label_b)
        ax.set_xlabel("Cycle")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} (comparison)")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend()

        out_path = comp_dir / filename
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg_paths.append(out_path)

    return svg_paths


def generate_tda_comparison_plots(run_a: Path, run_b: Path, label_a: str, label_b: str) -> list[Path]:
    """Compare TDA metrics across two runs."""
    tda_a = run_a / "tda_metrics.json"
    tda_b = run_b / "tda_metrics.json"
    if not tda_a.exists() or not tda_b.exists():
        return []

    data_a = json.loads(tda_a.read_text()).get("metrics", [])
    data_b = json.loads(tda_b.read_text()).get("metrics", [])
    if not data_a or not data_b:
        return []

    metrics = ("SNS", "PCS", "DRS", "HSS")
    comp_dir = run_a.parent / "comparison_plots"
    comp_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for metric in metrics:
        values_a = [float(entry.get(metric, 0.0)) for entry in data_a]
        values_b = [float(entry.get(metric, 0.0)) for entry in data_b]
        indices_a = [int(entry.get("window_index", idx)) for idx, entry in enumerate(data_a)]
        indices_b = [int(entry.get("window_index", idx)) for idx, entry in enumerate(data_b)]
        ax.plot(indices_a, values_a, linewidth=2.0, label=f"{metric} ({label_a})")
        ax.plot(indices_b, values_b, linewidth=2.0, linestyle="--", label=f"{metric} ({label_b})")

    ax.set_xlabel("Window index")
    ax.set_ylabel("TDA metric value")
    ax.set_title("TDA Metrics Comparison")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend()

    out_path = comp_dir / "tda_comparison.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return [out_path]


def write_comparison_summary(
    run_a: Path,
    run_b: Path,
    summary_path: Path,
) -> None:
    """Write a compact JSON comparison summary."""
    success_a, omega_a = load_summary_metrics(run_a)
    success_b, omega_b = load_summary_metrics(run_b)
    prng_a = load_prng_summary(run_a)
    prng_b = load_prng_summary(run_b)
    meta_a = load_run_metadata(run_a)
    meta_b = load_run_metadata(run_b)

    summary = {
        "run_a": {
            "path": str(run_a),
            "run_id": meta_a["run_id"],
            "pathology": meta_a["pathology"],
            "seed": meta_a["seed"],
            "success_rate": success_a,
            "omega_occupancy": omega_a,
            "prng": prng_a,
        },
        "run_b": {
            "path": str(run_b),
            "run_id": meta_b["run_id"],
            "pathology": meta_b["pathology"],
            "seed": meta_b["seed"],
            "success_rate": success_b,
            "omega_occupancy": omega_b,
            "prng": prng_b,
        },
        "delta": {
            "success_rate": success_b - success_a,
            "omega_occupancy": omega_b - omega_a,
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))


def main() -> int:
    """Run paired harnesses (baseline vs pathology) and emit comparison plots."""
    parser = argparse.ArgumentParser(description="Phase X P3 demo orchestrator")
    parser.add_argument("--with-tda", action="store_true", help="Generate TDA metric plots if available")
    parser.add_argument("--cycles", type=int, default=200, help="Cycles per run (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for both runs (default: 42)")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="phase_x_p3_demo_"))
    harness_output_dir = workdir / "runs"

    print(f"[p3-demo] Working directory: {workdir}")
    print("Run A: noise baseline (no pathology). Run B: spike pathology (stress test).")

    base_a = harness_output_dir / "run_a"
    base_b = harness_output_dir / "run_b"
    run_a = run_first_light_harness(base_a, cycles=args.cycles, seed=args.seed, pathology="none")
    run_b = run_first_light_harness(base_b, cycles=args.cycles, seed=args.seed, pathology="spike")

    # Per-run plots
    svg_paths = generate_plots(run_a) + generate_plots(run_b)

    # Comparison plots
    comparison_paths = generate_comparison_plots(run_a, run_b, "Run A (baseline)", "Run B (spike)")
    if args.with_tda:
        comparison_paths.extend(generate_tda_comparison_plots(run_a, run_b, "Run A (baseline)", "Run B (spike)"))

    summary_path = harness_output_dir / "comparison_plots" / "comparison_summary.json"
    write_comparison_summary(run_a, run_b, summary_path)

    success_rate_a, omega_occupancy_a = load_summary_metrics(run_a)
    success_rate_b, omega_occupancy_b = load_summary_metrics(run_b)
    prng_summary_b = load_prng_summary(run_b)

    print("[p3-demo] Generated SVGs:")
    for svg_path in svg_paths + comparison_paths:
        print(f"  {svg_path}")
    print(f"[p3-demo] Run A directory: {run_a}")
    print(f"[p3-demo] Run B directory: {run_b}")
    print(f"[p3-demo] Summary A: success_rate={success_rate_a:.3f}, omega_occupancy={omega_occupancy_a:.3f}")
    print(
        f"[p3-demo] Summary B: success_rate={success_rate_b:.3f}, "
        f"omega_occupancy={omega_occupancy_b:.3f}, prng_status={prng_summary_b.get('status')}"
    )
    print(f"[p3-demo] Comparison summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
