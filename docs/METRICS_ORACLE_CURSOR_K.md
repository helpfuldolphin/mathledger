# Cursor K — Metrics Oracle

## Overview
- Deterministic aggregation across PostgreSQL, performance passports, uplift experiments, and synthetic queue telemetry.
- Canonical payloads emitted at `artifacts/metrics/latest.json` plus per-session archives and history/trend ledgers.
- Dual attestation: per-run Merkle hash (`provenance.merkle_hash`) and cross-run Merkle hash (`provenance.history_merkle`).
- Pure ASCII outputs for downstream reporters (`latest_report.txt`, Markdown daily snapshots).

## Data Flow
1. **Collectors**  
   - `database`: proofs, statements, blocks, runs (gracefully skipped if DB absent).  
   - `performance_passport`: latency and memory distribution.  
   - `uplift`: guided vs baseline proof throughput.  
   - `queue`: synthetic fallback (extend when live queue metrics exist).
2. **Aggregator** (`backend/metrics_cartographer.py`)  
   - Normalizes metrics, merges provenance, records warnings.  
   - Maintains rolling history (`history.json`) and trend synthesis (`trends.json`).  
   - Computes variance against configurable epsilon (default `0.01`).
3. **Exports**  
   - `latest.json`, `session_{id}.json`, `history.json`, `trends.json`.  
   - ASCII reporter (`backend/metrics_reporter.py`) renders human report.  
   - Markdown reporter (`backend/metrics_md_report.py`) stores daily briefing.

## Running the Pipeline
```bash
# Full pipeline (collect + validate + report)
uv run python tools/metrics_cartographer_cli.py full

# Collection only (detects warnings, writes history)
uv run python tools/metrics_cartographer_cli.py collect

# ASCII dashboard
uv run python backend/metrics_reporter.py

# Markdown report (writes docs to reports/)
uv run python backend/metrics_md_report.py
```

### Environment knobs
- `MATHLEDGER_DB_URL` or `DATABASE_URL`: override PostgreSQL connection string.
- `METRICS_EPSILON`: adjust variance tolerance without code changes.

### Nightly automation
`run-nightly.ps1` now invokes the Cursor K pipeline (collect → ASCII report → Markdown report) on every run. Nightly CI therefore keeps `latest.json`, `history.json`, and the daily markdown briefing fresh—if any step fails, the nightly job aborts.

## First Organism Observability

The First Organism integration test is now a first-class citizen in the metrics pipeline, providing daily "vital signs" of the organism's health.

### Telemetry Flow
1. **Runner** (`scripts/run_first_organism.py`) executes the standalone or full integration test
2. **Emitter** (`backend/metrics/first_organism_telemetry.py`) pushes metrics to Redis
3. **Collector** (`FirstOrganismCollector`) reads from Redis during metrics aggregation
4. **Reports** display "First Organism Vital Signs" section with trends

### Redis Keys
- `ml:metrics:first_organism:runs_total` — total run count
- `ml:metrics:first_organism:last_ht` — last composite root Hₜ (short hash)
- `ml:metrics:first_organism:duration_seconds` — last run duration
- `ml:metrics:first_organism:last_abstentions` — abstention count in last run
- `ml:metrics:first_organism:last_run_timestamp` — ISO 8601 timestamp
- `ml:metrics:first_organism:last_status` — "success" or "failure"
- `ml:metrics:first_organism:duration_history` — rolling window (last 20)
- `ml:metrics:first_organism:abstention_history` — rolling window (last 20)
- `ml:metrics:first_organism:success_history` — rolling window (last 20)

### Metrics Collected
| Field | Description |
|-------|-------------|
| `runs_total` | Total number of FO runs |
| `last_ht_hash` | Composite attestation root Hₜ (short) |
| `last_duration_seconds` | Duration of most recent run |
| `average_duration_seconds` | Mean duration from history |
| `median_duration_seconds` | Median duration from history |
| `abstention_count` | Abstentions in last run |
| `success_rate` | Success percentage from history |
| `duration_delta` | Change from previous run duration |
| `abstention_delta` | Change from previous run abstentions |
| `last_status` | "success" or "failure" |

### Report Output
The ASCII and Markdown reports include a "First Organism Vital Signs" section:
- Status indicator (ALIVE/WARN/UNKNOWN)
- Last run timestamp and Hₜ short hash
- Duration with trend delta and sparkline visualization
- Abstention count with trend delta
- Success rate percentage

### Nightly Integration
`run-nightly.ps1` now includes Step 1.5 (First Organism Vital Signs) which:
1. Runs `scripts/run_first_organism.py --standalone --verbose`
2. Emits telemetry to Redis (if available)
3. Continues to Step 5 (Metrics Collection) which reads the fresh FO data

This ensures the daily metrics digest reflects the organism's current health state.

## Verification Checklist
1. **Schema compliance**  
   - `uv run python tools/metrics_cartographer_cli.py validate`  
   - Ensures `schema_v1.json` matches payload (including `trends` and `queue` sections).
2. **Deterministic digests**  
   - `jq '.provenance.merkle_hash' artifacts/metrics/latest.json`  
   - Re-running without new data should keep hash stable.
3. **History continuity**  
   - Verify `artifacts/metrics/history.json` retains the latest `N` sessions (default 30).  
   - `jq '.sessions | length' artifacts/metrics/history.json`
4. **Trend integrity**  
   - `jq '.trends' artifacts/metrics/trends.json`  
   - Confirm moving averages update after multiple runs.
5. **Reporter outputs**  
   - ASCII: `artifacts/metrics/latest_report.txt`.  
   - Markdown: `reports/metrics_YYYY-MM-DD.md` (ASCII-only, CI-diff friendly).

## Extension Hooks
- Add live queue collector (Redis, Rabbit, etc.) by implementing `BaseCollector`.
- Plug new experiment streams by returning `CollectorResult` with metrics + provenance.
- Adjust retention windows in `MetricsConfig` (short vs long averages, history length).

## Troubleshooting
- **Missing schema/performance passport** → pipeline aborts with `[ABSTAIN]`. Ensure `artifacts/metrics/schema_v1.json` and `performance_passport.json` exist.  
- **Database offline** → warning recorded; other collectors still run. History merkle remains valid.  
- **Variance exceedance** → CLI exits with non-zero; inspect `variance` section and recent history for regressions.

