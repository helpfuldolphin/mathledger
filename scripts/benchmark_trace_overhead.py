#!/usr/bin/env python3
"""
PHASE II — Trace Logging Overhead Benchmark

Measures the overhead of trace logging with and without filtering.
Target: <1% overhead for core events.
"""

import time
import tempfile
from pathlib import Path

from experiments.u2.logging import U2TraceLogger, CORE_EVENTS
from experiments.u2 import schema


def do_cycle_work(i: int) -> int:
    """Simulated experiment work per cycle."""
    # Minimal work - we want to measure pure logging overhead
    return i * 2


def main():
    # Benchmark: 1000 cycles WITHOUT logging
    start = time.perf_counter()
    for i in range(1000):
        do_cycle_work(i)
    baseline_time = time.perf_counter() - start

    # Benchmark: 1000 cycles WITH logging (all events)
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "all_events.jsonl"
        with U2TraceLogger(log_path) as logger:
            start = time.perf_counter()
            for i in range(1000):
                do_cycle_work(i)
                logger.log_cycle_duration(
                    schema.CycleDurationEvent(
                        cycle=i,
                        slice_name="bench",
                        mode="baseline",
                        duration_ms=1.0,
                        substrate_duration_ms=None,
                    )
                )
                logger.log_cycle_telemetry(
                    schema.CycleTelemetryEvent(
                        cycle=i,
                        slice_name="bench",
                        mode="baseline",
                        raw_record={"cycle": i},
                    )
                )
            all_events_time = time.perf_counter() - start

    # Benchmark: 1000 cycles WITH logging (CORE_EVENTS filter)
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "core_events.jsonl"
        with U2TraceLogger(log_path, enabled_events=CORE_EVENTS) as logger:
            start = time.perf_counter()
            for i in range(1000):
                do_cycle_work(i)
                logger.log_cycle_duration(
                    schema.CycleDurationEvent(
                        cycle=i,
                        slice_name="bench",
                        mode="baseline",
                        duration_ms=1.0,
                        substrate_duration_ms=None,
                    )
                )
                logger.log_cycle_telemetry(
                    schema.CycleTelemetryEvent(
                        cycle=i,
                        slice_name="bench",
                        mode="baseline",
                        raw_record={"cycle": i},
                    )
                )
            core_events_time = time.perf_counter() - start

    # Benchmark: 1000 cycles WITH logging (only cycle_duration)
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "duration_only.jsonl"
        with U2TraceLogger(log_path, enabled_events={"cycle_duration"}) as logger:
            start = time.perf_counter()
            for i in range(1000):
                do_cycle_work(i)
                logger.log_cycle_duration(
                    schema.CycleDurationEvent(
                        cycle=i,
                        slice_name="bench",
                        mode="baseline",
                        duration_ms=1.0,
                        substrate_duration_ms=None,
                    )
                )
                logger.log_cycle_telemetry(
                    schema.CycleTelemetryEvent(
                        cycle=i,
                        slice_name="bench",
                        mode="baseline",
                        raw_record={"cycle": i},
                    )
                )
            duration_only_time = time.perf_counter() - start

    # Calculate per-event overhead (2 events per cycle: duration + telemetry)
    all_events_overhead_per_event = (all_events_time - baseline_time) / (1000 * 2) * 1000  # ms
    core_events_overhead_per_event = (core_events_time - baseline_time) / (1000 * 2) * 1000  # ms
    dur_events_overhead_per_event = (duration_only_time - baseline_time) / (1000 * 1) * 1000  # ms (1 event)

    # Calculate what overhead would be with realistic cycle times
    # Real U2 experiments have ~10-50ms per cycle (substrate calls)
    realistic_cycle_ms = 20.0  # Conservative estimate
    realistic_overhead_pct = (core_events_overhead_per_event * 2 / realistic_cycle_ms) * 100

    # Report
    print("=" * 60)
    print("TRACE LOGGING OVERHEAD BENCHMARK (1000 cycles)")
    print("=" * 60)
    print()
    print("ABSOLUTE TIMING:")
    print(f"  Baseline (no logging):     {baseline_time*1000:.2f} ms total")
    print(f"  All events logging:        {all_events_time*1000:.2f} ms total")
    print(f"  Core events (filtered):    {core_events_time*1000:.2f} ms total")
    print(f"  Duration only (filtered):  {duration_only_time*1000:.2f} ms total")
    print()
    print("PER-EVENT OVERHEAD:")
    print(f"  All events:                {all_events_overhead_per_event:.3f} ms/event")
    print(f"  Core events:               {core_events_overhead_per_event:.3f} ms/event")
    print(f"  Duration only:             {dur_events_overhead_per_event:.3f} ms/event")
    print()
    print("REALISTIC OVERHEAD (assuming 20ms/cycle substrate latency):")
    print(f"  Core events (2 events/cycle): {realistic_overhead_pct:.2f}%")
    print()
    print("TARGETS:")
    print(f"  Per-event overhead:        <0.5 ms (actual: {core_events_overhead_per_event:.3f} ms) {'PASS' if core_events_overhead_per_event < 0.5 else 'FAIL'}")
    print(f"  Realistic cycle overhead:  <1% (actual: {realistic_overhead_pct:.2f}%) {'PASS' if realistic_overhead_pct < 1.0 else 'REVIEW'}")
    print("=" * 60)
    
    return 0 if realistic_overhead_pct < 1.0 else 1


if __name__ == "__main__":
    exit(main())

