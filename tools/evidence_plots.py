from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KeySpec = Union[str, Sequence[str]]


def plot_delta_p(jsonl_path: str, output_svg: str) -> None:
    """Plot delta-p across cycles from a JSONL file."""
    cycles, values = _load_series(
        Path(jsonl_path),
        value_keys=("delta_p", "deltaP", "delta"),
        cycle_keys=("cycle", "cycles", "real_cycle"),
        is_jsonl=True,
    )
    _render_line_plot(
        cycles,
        values,
        Path(output_svg),
        ylabel="Delta p",
        title="Delta p vs Cycles",
    )


def plot_rsi(json_path: str, output_svg: str) -> None:
    """Plot RSI across cycles from a JSON file."""
    cycles, values = _load_series(
        Path(json_path),
        value_keys=("rsi", "RSI", "metrics.mean_rsi"),
        cycle_keys=("cycle", "cycles", "end_cycle", "window_index"),
        is_jsonl=False,
    )
    _render_line_plot(
        cycles,
        values,
        Path(output_svg),
        ylabel="RSI",
        title="RSI vs Cycles",
    )


def plot_omega_occupancy(json_path: str, output_svg: str) -> None:
    """Plot omega occupancy across cycles from a JSON file."""
    cycles, values = _load_series(
        Path(json_path),
        value_keys=("omega_occupancy", "omega", "occupancy", "metrics.omega_occupancy"),
        cycle_keys=("cycle", "cycles", "end_cycle", "window_index"),
        is_jsonl=False,
    )
    _render_line_plot(
        cycles,
        values,
        Path(output_svg),
        ylabel="Omega occupancy",
        title="Omega Occupancy vs Cycles",
    )


def _load_series(
    path: Path,
    *,
    value_keys: Sequence[str],
    cycle_keys: Sequence[str],
    is_jsonl: bool,
) -> Tuple[List[float], List[float]]:
    if is_jsonl:
        records = _load_jsonl_records(path)
        return _records_to_series(records, cycle_keys, value_keys)

    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return _records_to_series(payload, cycle_keys, value_keys)

    if isinstance(payload, dict):
        for key in ("data", "series", "points", "windows", "records"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return _records_to_series(nested, cycle_keys, value_keys)
        return _vector_payload_to_series(payload, cycle_keys, value_keys)

    raise ValueError(f"Unsupported payload format in {path}")


def _load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def _records_to_series(
    records: Iterable[object],
    cycle_keys: Sequence[KeySpec],
    value_keys: Sequence[KeySpec],
) -> Tuple[List[float], List[float]]:
    cycles: List[float] = []
    values: List[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        cycle = _first_available(record, cycle_keys)
        value = _first_available(record, value_keys)
        if cycle is None or value is None:
            continue
        cycles.append(cycle)
        values.append(value)
    if not cycles or not values:
        raise ValueError("No usable datapoints found")
    return _sort_series(cycles, values)


def _vector_payload_to_series(
    payload: dict,
    cycle_keys: Sequence[KeySpec],
    value_keys: Sequence[KeySpec],
) -> Tuple[List[float], List[float]]:
    cycle_series = _first_available_list(payload, cycle_keys)
    value_series = _first_available_list(payload, value_keys)
    if cycle_series is not None and value_series is not None:
        if len(cycle_series) != len(value_series):
            raise ValueError("Cycle and value vectors must be the same length")
        return _sort_series(list(cycle_series), list(value_series))

    # Allow simple mapping {cycle: value}
    if not cycle_series and not value_series:
        points = []
        for key, val in payload.items():
            if isinstance(val, (int, float)):
                try:
                    numeric_key = float(key)
                except (TypeError, ValueError):
                    continue
                points.append((numeric_key, val))
        if points:
            points.sort(key=lambda pair: pair[0])
            cycles, values = zip(*points)
            return list(cycles), list(values)

    raise ValueError("Unable to interpret payload as a time series")


def _first_available(record: dict, keys: Sequence[KeySpec]):
    for key in keys:
        value = _extract_value(record, key)
        if value is not None:
            return value
    return None


def _first_available_list(record: dict, keys: Sequence[KeySpec]):
    for key in keys:
        value = _extract_value(record, key)
        if isinstance(value, list):
            return value
    return None


def _extract_value(record: Any, key: KeySpec):
    if isinstance(key, (list, tuple)):
        current: Any = record
        for part in key:
            if not isinstance(current, dict):
                return None
            if part not in current:
                return None
            current = current[part]
        return current
    if not isinstance(key, str):
        return None
    parts = key.split(".")
    current: Any = record
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _sort_series(cycles: List[float], values: List[float]) -> Tuple[List[float], List[float]]:
    try:
        paired = sorted(zip(cycles, values), key=lambda pair: pair[0])
    except TypeError:
        paired = list(zip(cycles, values))
    sorted_cycles = [pair[0] for pair in paired]
    sorted_values = [pair[1] for pair in paired]
    return sorted_cycles, sorted_values


def _render_line_plot(
    cycles: List[float],
    values: List[float],
    output_path: Path,
    *,
    ylabel: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cycles, values, color="#1f77b4", linewidth=2.0, marker="o", markersize=4)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
