# U2 Planner — Search Runtime & Derivation Engine

**Status**: Phase II Implementation  
**Owner**: Manus-F (Planner / Derivation / Search Runtime Engineer)  
**Date**: 2025-12-06

---

## Overview

The **U2 Planner** is a deterministic search runtime for MathLedger's derivation engine. It implements guided search, frontier management, and beam allocation with complete determinism guarantees for reproducibility and RFL evidence generation.

### Key Features

1. **Deterministic Execution**
   - MDAP (Master-Derived-Atomic-Path) seeding hierarchy
   - Platform-independent randomness
   - Stable replay across machines and OS

2. **Guided Search**
   - Policy-driven candidate selection (Baseline / RFL)
   - Priority-based frontier queue
   - Beam search with dynamic allocation

3. **Telemetry Export**
   - RFL Evidence Packs
   - Per-cycle trace logging
   - Determinism verification hashes

4. **Snapshot & Replay**
   - Checkpointing for long experiments
   - Resume from failure
   - Cycle-level replay debugging

---

## Architecture

### Component Hierarchy

```
U2 Planner
├── PRNG (Deterministic Random)
│   ├── Master Seed
│   ├── Slice PRNGs (per experiment slice)
│   └── Cycle PRNGs (per execution cycle)
│
├── Frontier Manager
│   ├── Priority Queue (min-heap)
│   ├── Deduplication (seen set)
│   └── Depth Tracking
│
├── Beam Allocator
│   ├── Budget Tracking
│   ├── Depth-Aware Allocation
│   └── Exhaustion Detection
│
├── Search Policy
│   ├── Baseline (random)
│   └── RFL (feedback-driven)
│
├── Runner
│   ├── Cycle Execution
│   ├── Snapshot Management
│   └── State Serialization
│
└── Telemetry
    ├── Trace Logger (JSONL)
    ├── Evidence Pack Generator
    └── Determinism Verifier
```

---

## INVARIANTS

### Determinism Guarantees

1. **PRNG Seeding**
   - Master seed → Slice seed → Cycle seed (hierarchical)
   - Same seed path always produces same random sequence
   - Seed canonicalization ensures cross-platform stability

2. **Frontier Operations**
   - Tie-breaking uses deterministic PRNG
   - Push/pop order is reproducible
   - State serialization preserves exact ordering

3. **Policy Evaluation**
   - Ranking is deterministic given same PRNG seed
   - Feature extraction is stable
   - Score computation is platform-independent

4. **Trace Canonicalization**
   - Events sorted by timestamp
   - JSON with sorted keys and stable formatting
   - SHA-256 hash for verification

### Planner Stability

1. **Randomness Control**
   - All randomness flows through MDAP seeding
   - No system RNG or time-based seeds
   - Slice index determines PRNG branch

2. **Replay Stability**
   - Same config + seed → same execution
   - Snapshot restore produces identical continuation
   - Trace hashes match across replays

3. **Search Trace Canonicalization**
   - Events are timestamped monotonically
   - Traces are hashed after sorting
   - Hash comparison verifies determinism

---

## Usage

### Basic Experiment

```python
from pathlib import Path
from experiments.u2 import U2Runner, U2Config

# Configure experiment
config = U2Config(
    experiment_id="arithmetic_baseline",
    slice_name="arithmetic_simple",
    mode="baseline",  # or "rfl"
    total_cycles=100,
    master_seed=42,
    max_beam_width=100,
    max_depth=10,
)

# Create runner
runner = U2Runner(config)

# Seed initial frontier
runner.frontier.push("axiom_0", priority=1.0, depth=0)

# Define execution function
def execute_candidate(item, seed):
    # Your derivation logic here
    success = verify_statement(item)
    result = {"outcome": "VERIFIED" if success else "FAILED"}
    return success, result

# Run cycles
for cycle in range(config.total_cycles):
    result = runner.run_cycle(cycle, execute_candidate)
    print(f"Cycle {cycle}: {result.candidates_processed} processed")
```

### With Trace Logging

```python
from experiments.u2 import run_with_traces

trace_path = Path("output/trace.jsonl")

results = run_with_traces(
    config=config,
    execute_fn=execute_candidate,
    trace_path=trace_path,
)

# Extract telemetry
from experiments.u2.telemetry import extract_telemetry_from_trace, export_telemetry

telemetry = extract_telemetry_from_trace(trace_path)
export_telemetry(telemetry, Path("output/telemetry.json"))
```

### With Snapshots

```python
config = U2Config(
    experiment_id="long_experiment",
    slice_name="complex_slice",
    mode="rfl",
    total_cycles=1000,
    master_seed=42,
    snapshot_interval=100,  # Save every 100 cycles
    snapshot_dir=Path("snapshots"),
)

runner = U2Runner(config)
runner.frontier.push("seed_item", priority=1.0, depth=0)

for cycle in range(config.total_cycles):
    result = runner.run_cycle(cycle, execute_candidate)
    
    # Snapshot saved automatically every 100 cycles
    if (cycle + 1) % config.snapshot_interval == 0:
        print(f"Snapshot saved at cycle {cycle}")
```

### Resume from Snapshot

```python
from experiments.u2.snapshots import find_latest_snapshot, load_snapshot

# Find latest snapshot
snapshot_path = find_latest_snapshot(
    Path("snapshots"),
    experiment_id="long_experiment"
)

# Load and restore
snapshot = load_snapshot(snapshot_path, verify_hash=True)
runner = U2Runner(config)
runner.restore_state(snapshot)

# Continue from where we left off
start_cycle = snapshot.current_cycle + 1
for cycle in range(start_cycle, config.total_cycles):
    result = runner.run_cycle(cycle, execute_candidate)
```

---

## Telemetry & Evidence Packs

### Evidence Pack Structure

```
evidence_pack_<experiment_id>/
├── metadata.json          # Experiment metadata
├── telemetry.json         # Aggregated statistics
└── trace.jsonl            # Full execution trace
```

### Creating Evidence Pack

```python
from experiments.u2.telemetry import create_evidence_pack

pack_dir = create_evidence_pack(
    trace_path=Path("output/trace.jsonl"),
    output_dir=Path("evidence_packs"),
    include_trace=True,
)

print(f"Evidence pack created: {pack_dir}")
```

### Verifying Determinism

```python
from experiments.u2.logging import verify_trace_determinism

# Run experiment twice
trace1 = Path("run1/trace.jsonl")
trace2 = Path("run2/trace.jsonl")

is_deterministic = verify_trace_determinism(trace1, trace2)
print(f"Deterministic: {is_deterministic}")
```

---

## Testing

### Run Determinism Tests

```bash
cd /home/ubuntu/mathledger
python3.11 tests/run_u2_tests.py
```

### Test Coverage

- ✅ PRNG same seed same sequence
- ✅ PRNG hierarchical isolation
- ✅ PRNG hierarchical determinism
- ✅ PRNG state serialization
- ✅ PRNG integer seed conversion
- ✅ Frontier push/pop determinism
- ✅ Frontier state serialization
- ✅ Baseline policy determinism
- ✅ RFL policy determinism

---

## Integration with Derivation Engine

The U2 planner integrates with MathLedger's existing derivation engine:

```python
from backend.axiom_engine.derive_core import DerivationEngine
from experiments.u2 import U2Runner, U2Config

# Initialize derivation engine
engine = DerivationEngine(
    db_url="postgresql://...",
    redis_url="redis://...",
    max_depth=3,
    max_breadth=100,
)

# Create U2 runner
config = U2Config(
    experiment_id="derive_experiment",
    slice_name="propositional_logic",
    mode="rfl",
    total_cycles=50,
    master_seed=42,
)

runner = U2Runner(config)

# Execution function calls derivation engine
def execute_with_engine(item, seed):
    # Derive new statements
    result = engine.derive_statements(steps=1)
    success = result["n_new"] > 0
    return success, result

# Run cycles
for cycle in range(config.total_cycles):
    runner.run_cycle(cycle, execute_with_engine)
```

---

## RFL Policy

The RFL (Reflexive Formal Learning) policy uses **verifiable feedback only**:

- ✅ Success rates from previous runs (ground truth)
- ✅ Structural features (depth, complexity)
- ✅ Derivation statistics (proof depth, breadth)
- ❌ NO RLHF (Reinforcement Learning from Human Feedback)
- ❌ NO preferences or proxy rewards
- ❌ NO unverifiable signals

### Feedback Format

```json
{
  "item_123": {
    "success_rate": 0.85,
    "avg_time_s": 1.2,
    "avg_depth": 2.3
  }
}
```

---

## Performance Considerations

### Beam Width

- Larger beam → more exploration, slower cycles
- Smaller beam → faster cycles, may miss solutions
- Recommended: 100-500 for most experiments

### Snapshot Interval

- More frequent → faster recovery, more disk I/O
- Less frequent → slower recovery, less overhead
- Recommended: Every 50-100 cycles

### Trace Logging

- Full traces → complete audit trail, large files
- Filtered traces → smaller files, less detail
- Use `event_filter` to log only critical events

---

## Troubleshooting

### Non-Deterministic Results

1. Check PRNG seeding (must use same master seed)
2. Verify no external randomness (system RNG, time-based)
3. Ensure same execution order (no parallel processing)
4. Compare trace hashes to find divergence point

### Snapshot Corruption

1. Verify hash on load (`verify_hash=True`)
2. Check disk space during save
3. Use snapshot rotation to keep backups

### Slow Execution

1. Reduce beam width
2. Increase snapshot interval
3. Filter trace events (log only CORE_EVENTS)
4. Profile execution function

---

## Future Enhancements

- [ ] Distributed frontier (multi-node search)
- [ ] Adaptive beam allocation (dynamic budget)
- [ ] Policy learning from RFL feedback
- [ ] Incremental snapshot compression
- [ ] Real-time telemetry streaming

---

## References

- **MDAP Seeding**: `rfl/prng/deterministic_prng.py`
- **Frontier Management**: `experiments/u2/frontier.py`
- **Search Policies**: `experiments/u2/policy.py`
- **Trace Schema**: `experiments/u2/schema.py`
- **Telemetry Export**: `experiments/u2/telemetry.py`

---

**Acceptance Criteria**:
- ✅ Determinism guaranteed across platforms
- ✅ Planner outputs stable under replay
- ✅ Search traces canonicalized before hashing
- ✅ Telemetry exported for RFL Evidence Packs
- ✅ Snapshot/restore maintains determinism
- ✅ All tests passing (9/9)
