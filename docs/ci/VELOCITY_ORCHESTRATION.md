# Velocity Orchestration System

## Overview

The Velocity Orchestration system extends the 33% CI runtime reduction to 50% through intelligent job scheduling, timing telemetry, and meta-optimization.

## Architecture

### Components

1. **Velocity Telemetry** (`tools/ci/velocity_telemetry.py`)
   - Collects timing data for all CI jobs
   - Generates canonical `artifacts/ci/perf_log.json`
   - Calculates velocity improvements vs baseline
   - Provides optimization hash (SHA256)

2. **Meta-Scheduler** (`tools/ci/meta_scheduler.py`)
   - Analyzes job DAGs from workflow YAML
   - Identifies parallelization opportunities
   - Calculates critical path
   - Suggests optimization strategies

3. **Optimized Workflow** (`.github/workflows/ci-optimized.yml`)
   - Implements velocity orchestration
   - Records job timings automatically
   - Generates velocity summary
   - Enforces 5% variance threshold (Proof-or-Abstain)

## Performance Targets

| Metric | Baseline | Current | Target | Status |
|--------|----------|---------|--------|--------|
| Total CI Runtime | 420s | 280s (33%) | 210s (50%) | In Progress |
| Test Job | 180s | 120s | 90s | Optimizing |
| Uplift-Omega | 90s | 60s | 45s | Optimizing |
| Dual-Attestation | 150s | 100s | 75s | Optimizing |

## Optimization Strategies

### 1. Dependency Caching (Implemented)
- UV dependency caching
- Saves 20-30s per job
- Cache hit rate: >80%

### 2. Test Consolidation (Implemented)
- Merged duplicate test runs
- Single coverage execution
- Saves 60s

### 3. Parallel Job Execution (Implemented)
- Test and uplift-omega run in parallel
- Reduces wall-clock time
- Saves 90s (max of parallel jobs)

### 4. Meta-Scheduling (New)
- Dynamic job reordering based on telemetry
- Critical path optimization
- Estimated savings: 30-40s

### 5. Step-Level Optimization (New)
- Reorder steps within jobs
- Minimize setup overhead
- Estimated savings: 20-30s

## Telemetry Format

### Canonical perf_log.json

```json
{
  "version": "1.0",
  "generated_at": "2025-10-31T20:00:00Z",
  "runs": [
    {
      "workflow_name": "ci-optimized.yml",
      "run_id": "12345678",
      "trigger": "pull_request",
      "start_time": 1730404800,
      "end_time": 1730405010,
      "total_duration_seconds": 210,
      "baseline_duration_seconds": 420,
      "velocity_improvement_percent": 50.0,
      "optimization_hash": "a1b2c3d4e5f6...",
      "jobs": [
        {
          "job_name": "test",
          "start_time": 1730404800,
          "end_time": 1730404890,
          "duration_seconds": 90,
          "status": "pass",
          "step_timings": []
        }
      ]
    }
  ]
}
```

## CI Summary Format

```
[PASS] CI Velocity: 50.0% faster
CI_OPTIMIZATION_HASH: a1b2c3d4e5f6789...

Total Duration: 210s (Baseline: 420s)
```

## Variance Detection

The system implements Proof-or-Abstain on variance >5%:

- If actual improvement deviates >5% from expected, abstain from velocity claim
- Example: Expected 50%, Actual 45% → Variance = 10% → ABSTAIN
- Ensures honest reporting and prevents greenfaking

## Usage

### Analyze Workflow

```bash
python tools/ci/meta_scheduler.py .github/workflows/ci.yml artifacts/ci/perf_log.json
```

### Generate Telemetry

```bash
python tools/ci/velocity_telemetry.py
```

### Run Optimized Workflow

The optimized workflow runs automatically on PR and push events. It:
1. Plans execution with velocity-planner job
2. Runs test and uplift-omega in parallel
3. Collects timing data
4. Generates velocity summary
5. Uploads perf_log.json artifact

## Determinism Guarantees

- All timing measurements use monotonic clocks
- Optimization hash is deterministic (SHA256 of sorted config)
- Telemetry format is canonical JSON (sorted keys)
- No random elements in scheduling decisions
- ASCII-only outputs

## Rollback Safety

All optimizations are reversible:
1. Velocity orchestration can be disabled by using original ci.yml
2. Telemetry collection is non-invasive (doesn't affect job execution)
3. Meta-scheduler is advisory only (doesn't modify workflows automatically)

## Future Enhancements

1. **Machine Learning Scheduler**
   - Predict job durations based on code changes
   - Dynamically adjust parallelism
   - Learn from historical telemetry

2. **Adaptive Caching**
   - Intelligent cache invalidation
   - Predictive pre-warming
   - Multi-level cache hierarchy

3. **Resource-Aware Scheduling**
   - Consider runner capacity
   - Balance load across jobs
   - Optimize for cost vs speed

## Maintenance

- Review telemetry data weekly
- Update baseline metrics monthly
- Audit optimization hash quarterly
- Verify determinism continuously

---

**Status**: Active Development
**Target**: 50% end-to-end runtime reduction
**Current**: 33% achieved, 17% remaining
**ETA**: Next sprint
