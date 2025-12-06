# First Organism Environment Hardening Plan

**Security Officer:** CLAUDE N
**Date:** 2025-11-30
**Classification:** Internal - Security Configuration

---

> **STATUS: SECURITY HARDENING GUIDELINES**
>
> This document contains both **currently enforced** thresholds and **recommended targets**.
>
> | Credential | Current Enforcer (Phase I) | Recommended Target (Phase II) |
> |------------|---------------------------|-------------------------------|
> | PostgreSQL Password | 12 characters minimum | 16 characters minimum |
> | Redis Password | 12 characters minimum | 16 characters minimum |
> | API Key | 16 characters minimum | 32 characters minimum |
> | API Key Entropy | 6 unique characters | 10 unique characters |
>
> **Source:** `backend/security/first_organism_enforcer.py` lines 85-86 define:
> ```python
> MIN_PASSWORD_LENGTH = 12
> MIN_API_KEY_LENGTH = 16
> ```
>
> The stricter recommendations in Section 2.1 are **Phase II targets** not yet implemented in code.

---

## RFL Evidence & Environment

**Clarification for Reviewer-2:**

- **RFL logs do not hit infrastructure.** The Reflective Feedback Loop (RFL) operates on in-memory abstention decisions and does not write to PostgreSQL, Redis, or any external service during Phase I runs.
- **RFL Phase I operates hermetically.** The 1000-cycle Dyno Chart data (baseline vs RFL) was generated without database connections; abstention counts are computed purely from derivation state.
- **Environment hardening does not affect RFL runs.** The security enforcer validates credentials for First Organism closed-loop tests that exercise the full stack (DB + Redis + API). RFL experiments are orthogonal—they run in-process without invoking the enforcer or requiring `.env.first_organism`.

This document pertains to **First Organism integration tests**, not RFL evidence collection.

---

## 1. Executive Summary

This document establishes security requirements for the First Organism integration test environment. The goal is to prevent accidental test execution against production databases while maintaining a secure, reproducible test harness.

---

## 2. Security Recommendations

### 2.1 Password Entropy Requirements

> **Phase II Recommendations** — The table below shows hardening targets.
> Current enforcer (Phase I) uses: passwords ≥12 chars, API keys ≥16 chars, entropy ≥6 unique chars.

| Credential Type | Current (Phase I) | Recommended (Phase II) | Character Set Recommended |
|-----------------|-------------------|------------------------|---------------------------|
| PostgreSQL Password | 12 characters | 16 characters | Uppercase + Lowercase + Digits + Symbols |
| Redis Password | 12 characters | 16 characters | Uppercase + Lowercase + Digits |
| API Key | 16 characters | 32 characters | Hex characters (0-9, a-f) or alphanumeric |

**Entropy Calculation (Phase II target, not currently enforced):**
- Passwords should have at least **10 unique characters** (current: not checked for passwords)
- API keys must have at least **6 unique characters** (current: enforced)
- No dictionary words or common patterns (current: banned list only, no pattern detection)

**Generation Commands:**

```powershell
# PostgreSQL Password (32 chars, mixed with symbols)
$pg_pass = -join ((65..90) + (97..122) + (48..57) + (33,35,37,38,42,64,94) | Get-Random -Count 32 | ForEach-Object {[char]$_})
Write-Host "POSTGRES_PASSWORD=$pg_pass"

# Redis Password (24 chars, alphanumeric)
$redis_pass = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 24 | ForEach-Object {[char]$_})
Write-Host "REDIS_PASSWORD=$redis_pass"

# API Key (64 hex chars)
$api_key = -join ((48..57) + (97..102) | Get-Random -Count 64 | ForEach-Object {[char]$_})
Write-Host "LEDGER_API_KEY=$api_key"
```

### 2.2 Rotation Interval

| Credential | Rotation Interval | Trigger Events |
|------------|-------------------|----------------|
| PostgreSQL Password | 90 days | Personnel change, suspected breach |
| Redis Password | 90 days | Personnel change, suspected breach |
| API Key | 30 days | Key exposure, deployment to new environment |
| TLS Certificates | 365 days | Before expiration |

**Rotation Procedure:**
1. Generate new credential using commands above
2. Update `.env.first_organism` file
3. Restart Docker Compose stack
4. Verify connectivity with health checks
5. Archive old credential hash (not plaintext) in rotation log

### 2.3 Allowed Networks

| Service | Allowed Source | Rationale |
|---------|----------------|-----------|
| PostgreSQL (5432) | `127.0.0.1` only | Local testing only |
| Redis (6380) | `127.0.0.1` only | Local testing only |
| API (8000) | `127.0.0.1`, `localhost` | Local development UI |

**Network Restrictions:**
- **NEVER** expose First Organism services on `0.0.0.0`
- **NEVER** allow external network access to test databases
- **NEVER** run First Organism tests on production networks

### 2.4 Localhost Binding Rules

All services MUST bind to localhost (`127.0.0.1`) exclusively:

```yaml
# CORRECT - localhost only
ports:
  - "127.0.0.1:5432:5432"

# INCORRECT - exposes to all interfaces
ports:
  - "5432:5432"
  - "0.0.0.0:5432:5432"
```

**Verification Command:**
```powershell
# Verify no services bound to 0.0.0.0
netstat -an | Select-String ":5432|:6379|:6380" | Select-String "0.0.0.0" | ForEach-Object {
    Write-Host "SECURITY VIOLATION: $_" -ForegroundColor Red
}
```

### 2.5 Docker Profile Locking

**Required Security Options:**
```yaml
security_opt:
  - no-new-privileges:true    # Prevent privilege escalation
```

**Resource Limits (prevent DoS from runaway tests):**
```yaml
deploy:
  resources:
    limits:
      memory: 512M            # PostgreSQL
      cpus: '1.0'
    reservations:
      memory: 256M
```

**Container Isolation:**
- Use dedicated Docker network (`first_organism_network`)
- No shared volumes with host system (except data volumes)
- Read-only root filesystem where possible

### 2.6 Secret Scanning CI Rules

Add to CI pipeline (`.github/workflows/security.yml` or equivalent):

```yaml
name: Secret Scanning
on: [push, pull_request]

jobs:
  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Scan for secrets
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          extra_args: --only-verified

      - name: Check for hardcoded credentials
        run: |
          # Fail if any .env files contain non-placeholder passwords
          if grep -r "POSTGRES_PASSWORD=" --include="*.env" --include="*.env.*" . | grep -v "REPLACE" | grep -v ".template"; then
            echo "ERROR: Hardcoded PostgreSQL password detected"
            exit 1
          fi
          if grep -r "REDIS_PASSWORD=" --include="*.env" --include="*.env.*" . | grep -v "REPLACE" | grep -v ".template"; then
            echo "ERROR: Hardcoded Redis password detected"
            exit 1
          fi
          if grep -r "LEDGER_API_KEY=" --include="*.env" --include="*.env.*" . | grep -v "REPLACE" | grep -v ".template"; then
            echo "ERROR: Hardcoded API key detected"
            exit 1
          fi
```

**Pre-commit Hook (`scripts/pre-commit-secret-check.ps1`):**
```powershell
# Check staged files for secrets before commit
$staged = git diff --cached --name-only
foreach ($file in $staged) {
    if ($file -match "\.env" -and $file -notmatch "\.template") {
        Write-Host "WARNING: Attempting to commit $file" -ForegroundColor Yellow
        Write-Host "Ensure no secrets are exposed" -ForegroundColor Yellow
    }
}
```

---

## 3. Banned Credential Values

The following values are **ALWAYS REJECTED** by the First Organism enforcer:

### PostgreSQL Passwords (Banned) — Currently Enforced
- `postgres`, `password`, `mlpass`, `ml`, `mathledger`
- `admin`, `root`, `test`, `secret`, `changeme`
- Any password < 12 characters (enforcer constant: `MIN_PASSWORD_LENGTH = 12`)

### Redis Passwords (Banned) — Currently Enforced
- `redis`, `password`, `secret`, `changeme`, `test`
- Empty string
- Any password < 12 characters (enforcer constant: `MIN_PASSWORD_LENGTH = 12`)

### API Keys (Banned) — Currently Enforced
- `devkey`, `dev`, `test`, `secret`, `changeme`
- `api-key`, `apikey`
- Any key < 16 characters (enforcer constant: `MIN_API_KEY_LENGTH = 16`)
- Any key with < 6 unique characters (enforcer check: `len(set(key)) < 6`)

### CORS Origins (Banned) — Currently Enforced
- `*` (wildcard)
- Empty value

---

## 4. Compliance Checklist

Before running First Organism tests, verify:

**Minimum Requirements (Currently Enforced by Phase I Enforcer):**
- [ ] `.env.first_organism` exists and is NOT committed to git
- [ ] All `<REPLACE_...>` placeholders have been replaced
- [ ] PostgreSQL password is 12+ characters, not in banned list
- [ ] Redis password is 12+ characters, not in banned list
- [ ] API key is 16+ characters with 6+ unique characters
- [ ] CORS origins are explicit (no wildcards)

**Recommended (Phase II Targets, Not Currently Enforced):**
- [ ] PostgreSQL password is 16+ characters with mixed character types
- [ ] Redis password is 16+ characters
- [ ] API key is 32+ characters with 10+ unique characters
- [ ] Docker services bind to `127.0.0.1` only
- [ ] `no-new-privileges` security option is enabled
- [ ] Resource limits are configured

---

## 5. Incident Response

If credentials are exposed:

1. **Immediately rotate** all affected credentials
2. **Stop** all First Organism containers
3. **Check** git history for committed secrets
4. **Run** `git filter-branch` or BFG Repo-Cleaner if secrets were committed
5. **Notify** team members who may have pulled exposed credentials
6. **Document** incident in security log

---

## Appendix A: Enforcer Integration

The First Organism enforcer (`backend/security/first_organism_enforcer.py`) automatically validates:
- Database URL format and password strength (≥12 chars, banned list)
- Redis URL format and password strength (≥12 chars, banned list)
- API key length (≥16 chars) and entropy (≥6 unique chars)
- CORS origin policy (no wildcards)
- Runtime environment marker (`test_hardened`, `first_organism`, or `integration`)

Run enforcer validation manually:
```powershell
uv run python -c "from backend.security.first_organism_enforcer import enforce_first_organism_env; enforce_first_organism_env()"
```

---

## Phase II: RFL Uplift Experiments & Environment Requirements

> **STATUS: FUTURE WORK — NOT IMPLEMENTED**
>
> This section describes environment hardening requirements that will be necessary
> when RFL transitions from hermetic in-memory operation to DB-backed uplift experiments.
> None of these requirements apply to Phase I evidence (1000-cycle Dyno Chart, attestation.json).

### Phase I vs Phase II RFL: Security Model Comparison

| Aspect | Phase I (Current) | Phase II (Future) |
|--------|-------------------|-------------------|
| Data persistence | In-memory only | PostgreSQL + Redis |
| Attack surface | None (hermetic) | DB injection, credential exposure |
| Network exposure | None | localhost DB connections |
| Credential requirements | None | Full FO enforcer + RFL-specific |
| Migration dependencies | None | RFL schema migrations required |
| Evidence integrity | File hashes only | DB transaction integrity |

### Extra Constraints for Phase II RFL

When RFL begins writing to infrastructure, the following hardening requirements become mandatory:

**1. RFL-Specific Database Tables**
- Dedicated `rfl_cycles`, `rfl_abstentions`, `rfl_feedback` tables
- Schema must be migrated BEFORE any RFL uplift run
- Migration order: core schema → FO schema → RFL schema
- Rollback scripts required for each migration

**2. Connection String Scrutiny**
- `RFL_DB_URL` must be validated separately from `DATABASE_URL`
- Phase II enforcer must verify:
  - RFL connects to correct database (not production ledger)
  - Connection uses dedicated RFL user with restricted permissions
  - `sslmode=require` for any non-localhost connection

**3. SSL/TLS Expectations**
- Localhost connections: `sslmode=disable` acceptable
- Any remote connection: `sslmode=require` mandatory
- Certificate validation for production-adjacent environments

**4. Redis Isolation**
- Separate Redis database index for RFL queues (e.g., `/1` vs `/0`)
- Or separate `RFL_REDIS_URL` entirely
- Queue key namespacing: `ml:rfl:*` vs `ml:jobs:*`

**5. Environment Mode Marker**
- New variable: `RFL_ENV_MODE`
- Valid values: `phase1-hermetic` (current), `phase2-uplift` (future)
- Enforcer rejects Phase II operations unless `RFL_ENV_MODE=phase2-uplift`

**6. Audit Trail Requirements**
- All RFL DB writes must be logged with timestamp and cycle ID
- Transaction isolation level: `SERIALIZABLE` for feedback writes
- Integrity checks: cycle counts must match between memory and DB

### Migration Checklist (Phase II Readiness)

Before any DB-backed RFL experiment:

- [ ] RFL schema migrations applied and verified
- [ ] `RFL_DB_URL` configured with dedicated credentials
- [ ] `RFL_REDIS_URL` or isolated database index configured
- [ ] `RFL_ENV_MODE=phase2-uplift` set explicitly
- [ ] SSL mode verified for connection type
- [ ] Rollback procedure documented and tested
- [ ] Audit logging enabled for RFL tables
- [ ] Backup taken before first uplift run

### What This Means for Evidence Pack

- **Phase I evidence (current):** No DB hardening required. RFL logs are file-based, hermetic, and independently verifiable.
- **Phase II evidence (future):** DB hardening is prerequisite. No uplift experiment results are trustworthy without the above constraints verified.

---

## 6. PHASE II — Environment Modes for Uplift Experiments

> **CLAUDE N — Environment & Hardening Nonfunctional Requirements**
>
> **Revision:** 2025-12-05 | **Author:** CLAUDE N | **Scope:** Phase II U2 Runner

This section outlines the critical environment and non-functional requirements for Phase II, ensuring that Uplift experiments are conducted with absolute integrity, reproducibility, and isolation from Phase I systems.

**Key Phase II Requirements:**
- **New Environment Modes:** Introduction of `uplift_experiment` and `uplift_analysis` modes for granular control over RFL operations (detailed in Section 6.1).
- **Isolation, Logging, & Deterministic Seeds:** Strict requirements for experiment isolation, structured logging, and deterministic Pseudo-Random Number Generators (PRNGs) to ensure reproducibility and auditability (detailed in Sections 6.2 and 6.3).
- **Resource and Timeout Limits:** Implementation of robust resource and timeout limits to prevent interference and ensure stable experiment execution (detailed in Section 2.5).
- **"No Change to Phase I Pathways" Guarantee:** Absolute safeguard ensuring that Phase II development does not alter the behavior or integrity of Phase I systems (as stated in the "Phase II Goal" preamble and "Phase II vs Phase I RFL: Security Model Comparison").

**Nonfunctional Requirements Summary (NFR):**

| NFR ID | Requirement | Verification Method |
|--------|-------------|---------------------|
| NFR-001 | Environment mode must be explicitly declared | Enforcer startup check |
| NFR-002 | Cache directories must be isolated per-run | Path validation, symlink prohibition |
| NFR-003 | All PRNGs must derive from preregistered master seed | Seed derivation audit |
| NFR-004 | Logs must conform to structured schema | Schema validation on emit |
| NFR-005 | Snapshots must include integrity hashes | Hash verification on restore |
| NFR-006 | Determinism must be verifiable via replay | Checkpoint comparison |
| NFR-007 | No banned randomness sources in execution path | Static analysis + runtime guards |

---

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> This section defines environment modes, configuration requirements, and telemetry infrastructure
> for U2 uplift experiments. All artifacts and runs under this section are Phase II only.

### 6.1 Environment Mode Definitions

Phase II introduces two new environment modes beyond `phase1-hermetic`:

| Mode | `RFL_ENV_MODE` Value | Purpose | DB Access | Cache Mode |
|------|---------------------|---------|-----------|------------|
| **Phase I Hermetic** | `phase1-hermetic` | In-memory RFL (baseline, no DB) | None | N/A |
| **Uplift Experiment** | `uplift_experiment` | Execute U2 runs with asymmetric environments | Read/Write | Isolated |
| **Uplift Analysis** | `uplift_analysis` | Post-hoc analysis of completed U2 runs | Read-Only | Shared |

**Mode Transition Rules:**

```
phase1-hermetic ──[explicit migration]──> uplift_experiment
                                              │
                                              ▼
                                        uplift_analysis
```

- Transition from `phase1-hermetic` to `uplift_experiment` requires:
  - All Phase II migrations applied
  - Preregistration hash verified against `PREREG_UPLIFT_U2.yaml`
  - Cache root directories created and validated
- Transition from `uplift_experiment` to `uplift_analysis` is automatic after run completion
- **No reverse transitions allowed** (analysis cannot modify experiment state)

**Mode Detection Logic:**

```python
# backend/rfl/env_mode.py (PHASE II)
import os
from enum import Enum
from typing import NoReturn

class RFLEnvMode(Enum):
    PHASE1_HERMETIC = "phase1-hermetic"
    UPLIFT_EXPERIMENT = "uplift_experiment"
    UPLIFT_ANALYSIS = "uplift_analysis"

def get_rfl_env_mode() -> RFLEnvMode:
    """Get current RFL environment mode with strict validation."""
    mode_str = os.environ.get("RFL_ENV_MODE", "phase1-hermetic")
    try:
        return RFLEnvMode(mode_str)
    except ValueError:
        raise ValueError(
            f"Invalid RFL_ENV_MODE: {mode_str!r}. "
            f"Valid values: {[m.value for m in RFLEnvMode]}"
        )

def require_mode(required: RFLEnvMode) -> None:
    """Enforce specific mode or raise."""
    current = get_rfl_env_mode()
    if current != required:
        raise RuntimeError(
            f"Operation requires RFL_ENV_MODE={required.value}, "
            f"but current mode is {current.value}"
        )
```

**Mode Enforcement Points:**

| Operation | Required Mode | Rationale |
|-----------|---------------|-----------|
| U2 cycle execution | `uplift_experiment` | Writes feedback, policy updates |
| Snapshot creation | `uplift_experiment` | Creates reproducibility artifacts |
| Evidence export | `uplift_experiment` or `uplift_analysis` | Read access sufficient for export |
| Metric computation | `uplift_analysis` | Read-only aggregation |
| Cross-run comparison | `uplift_analysis` | No modification allowed |

### 6.2 Configuration Requirements

#### 6.2.1 Cache Isolation

> **NFR-002 Implementation**

U2 runs MUST use isolated caches to prevent cross-contamination between:
- Different asymmetric environments (A1, A2, A3, A4)
- Baseline vs treatment runs
- Sequential experiment batches

**Isolation Guarantees:**

| Guarantee | Enforcement | Failure Mode |
|-----------|-------------|--------------|
| No shared mutable state | Per-run directory structure | Abort if path collision detected |
| No cache poisoning | Hash verification on read | Reject stale/corrupted entries |
| No cross-environment leakage | Environment ID in all paths | Fail-fast on mismatch |
| No symlink exploitation | Real path validation | Reject symlinks in cache tree |

**Cache Isolation Configuration:**

```yaml
# config/u2_experiment.yaml (PHASE II)
cache:
  isolation_strategy: "per_run"  # Options: per_run, per_environment, shared

  # Per-run isolation (REQUIRED for U2)
  directories:
    base: "${MATHLEDGER_CACHE_ROOT}/u2"
    pattern: "${base}/${run_id}/${environment_id}"

  # Cache components requiring isolation
  components:
    statement_cache: true      # Derived statements per environment
    policy_cache: true         # Policy weights per environment
    prng_state: true           # PRNG checkpoints
    lean_cache: false          # Lean build cache can be shared (read-only)

  # Cleanup policy
  retention:
    completed_runs: 30         # Days to retain completed run caches
    failed_runs: 7             # Days to retain failed run caches
    orphaned: 1                # Days before orphan cache cleanup
```

**Enforcer Validation:**

```python
# backend/security/first_organism_enforcer.py additions (PHASE II)
def validate_cache_isolation(run_id: str, env_id: str) -> None:
    """Ensure cache directory is isolated and writable."""
    from backend.rfl.env_mode import require_mode, RFLEnvMode
    require_mode(RFLEnvMode.UPLIFT_EXPERIMENT)

    cache_root = os.environ.get("MATHLEDGER_CACHE_ROOT", "./cache")
    run_cache = Path(cache_root) / "u2" / run_id / env_id

    # Prevent path traversal
    if ".." in str(run_cache):
        raise SecurityError("Path traversal detected in cache path")

    # Ensure isolation (no symlinks to shared locations)
    if run_cache.exists() and run_cache.is_symlink():
        raise SecurityError("Cache directory must not be a symlink")

    # Create with restricted permissions
    run_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
```

#### 6.2.2 Deterministic PRNGs

> **NFR-003, NFR-006, NFR-007 Implementation**

All randomness in U2 experiments MUST be deterministic and reproducible.

**Determinism Contract:**

```
Master Seed (from PREREG_UPLIFT_U2.yaml hash)
    │
    ├─► derive(run_id, env_id="A1", component="policy")    → PRNG_A1_policy
    ├─► derive(run_id, env_id="A1", component="sampling")  → PRNG_A1_sampling
    ├─► derive(run_id, env_id="A2", component="policy")    → PRNG_A2_policy
    └─► ... (all environment × component combinations)
```

**Determinism Verification:**
1. **Replay Check:** Re-run with same seed must produce identical PRNG sequences
2. **Checkpoint Comparison:** PRNG state at cycle N must match across replays
3. **Call Count Audit:** Each PRNG tracks call count; mismatch indicates non-determinism

**PRNG Architecture:**

```python
# backend/rfl/determinism.py (PHASE II)
import hashlib
import struct
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class PRNGSeed:
    """Immutable, derivable PRNG seed for U2 experiments."""
    master_seed: bytes          # 32-byte master seed from preregistration
    run_id: str                 # Unique run identifier
    environment_id: str         # A1, A2, A3, A4, or baseline
    component: str              # "policy", "sampling", "tiebreaker"

    def derive(self) -> int:
        """Derive deterministic seed for this context."""
        material = (
            self.master_seed +
            self.run_id.encode() +
            self.environment_id.encode() +
            self.component.encode()
        )
        digest = hashlib.sha256(material).digest()
        return struct.unpack(">Q", digest[:8])[0]

class DeterministicPRNG:
    """PRNG with mandatory checkpointing for reproducibility."""

    def __init__(self, seed: PRNGSeed):
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed.derive()))
        self._call_count = 0
        self._checkpoints: dict[int, bytes] = {}

    def checkpoint(self) -> bytes:
        """Create serializable checkpoint of PRNG state."""
        state = self.rng.bit_generator.state
        checkpoint = {
            "seed": self.seed,
            "call_count": self._call_count,
            "state": state
        }
        serialized = pickle.dumps(checkpoint)
        self._checkpoints[self._call_count] = serialized
        return serialized

    @classmethod
    def restore(cls, checkpoint: bytes) -> "DeterministicPRNG":
        """Restore PRNG from checkpoint."""
        data = pickle.loads(checkpoint)
        prng = cls(data["seed"])
        prng.rng.bit_generator.state = data["state"]
        prng._call_count = data["call_count"]
        return prng

    def random(self) -> float:
        """Generate random float with call tracking."""
        self._call_count += 1
        return self.rng.random()

    def choice(self, items: list, size: int = 1) -> list:
        """Deterministic choice with call tracking."""
        self._call_count += 1
        indices = self.rng.choice(len(items), size=size, replace=False)
        return [items[i] for i in indices]
```

**PRNG Configuration:**

```yaml
# config/u2_prng.yaml (PHASE II)
prng:
  # Master seed from PREREG_UPLIFT_U2.yaml (SHA256 of preregistration doc)
  master_seed_source: "preregistration_hash"

  # Component-specific PRNG instances (all derived from master)
  components:
    policy_update:
      algorithm: "PCG64"
      checkpoint_interval: 100    # Checkpoint every 100 calls

    candidate_sampling:
      algorithm: "PCG64"
      checkpoint_interval: 50

    tiebreaker:
      algorithm: "PCG64"
      checkpoint_interval: 10     # Fine-grained for reproducibility

  # Validation requirements
  validation:
    require_checkpoint_on_cycle_end: true
    require_state_in_snapshot: true
    detect_nondeterminism: true   # Fail if same seed produces different results
```

**Banned Sources of Randomness:**

| Banned | Replacement | Rationale |
|--------|-------------|-----------|
| `random.random()` | `DeterministicPRNG.random()` | Not seedable per-component |
| `time.time()` as seed | `PRNGSeed.derive()` | Non-deterministic |
| `os.urandom()` | Preregistered master seed | Non-reproducible |
| `uuid.uuid4()` | `uuid.uuid5(namespace, deterministic_name)` | Random by design |
| `hash()` on objects | `hashlib.sha256()` | Python hash is randomized |

### 6.3 Logging & Snapshotting Requirements

> **NFR-004, NFR-005 Implementation**

All U2 runs produce structured logs and periodic snapshots to enable:
- **Reproducibility:** Any cycle can be replayed from the nearest snapshot
- **Auditability:** Complete event trail from run start to completion
- **Debugging:** Errors can be traced to exact PRNG state and inputs
- **Analysis:** Metrics extraction without re-running experiments

#### 6.3.1 Structured Logging Schema

> **Logging Requirements:**
> - All events MUST use `U2LogEntry` schema
> - Events MUST be written to JSONL files (append-only)
> - Each cycle MUST emit all `REQUIRED_CYCLE_EVENTS`
> - Log files MUST be flushed after each cycle (no buffering across cycles)

All U2 runs MUST emit structured logs conforming to this schema:

```python
# backend/rfl/logging_schema.py (PHASE II)
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Literal, Optional, Any
import json

@dataclass
class U2LogEntry:
    """Structured log entry for U2 uplift experiments."""

    # Identity
    timestamp: str                  # ISO8601 UTC
    run_id: str                     # Unique run identifier
    environment_id: str             # A1, A2, A3, A4, baseline
    cycle: int                      # Cycle number within run

    # Event classification
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "METRIC"]
    category: Literal[
        "lifecycle",        # Run start/stop/checkpoint
        "derivation",       # Statement derivation events
        "policy",           # Policy updates
        "feedback",         # Feedback computation
        "snapshot",         # Snapshot creation
        "validation",       # Determinism/integrity checks
        "error"             # Errors and failures
    ]
    event: str                      # Specific event name

    # Payload
    data: dict[str, Any]            # Event-specific structured data

    # Provenance
    component: str                  # Source component (e.g., "policy_engine")
    prng_call_count: Optional[int]  # PRNG state for reproducibility

    def to_jsonl(self) -> str:
        """Serialize to JSON Lines format."""
        return json.dumps(asdict(self), sort_keys=True, default=str)

# Required events per cycle
REQUIRED_CYCLE_EVENTS = [
    ("lifecycle", "cycle_start"),
    ("derivation", "candidates_generated"),
    ("policy", "candidates_ranked"),
    ("derivation", "statement_selected"),
    ("feedback", "feedback_computed"),
    ("policy", "policy_updated"),
    ("lifecycle", "cycle_end"),
]
```

**Log File Structure:**

```
logs/u2/
├── {run_id}/
│   ├── manifest.json           # Run metadata and configuration
│   ├── events.jsonl            # All structured events (append-only)
│   ├── metrics.jsonl           # Metric events only (filtered view)
│   ├── errors.jsonl            # Error events only
│   └── environments/
│       ├── A1/
│       │   ├── cycles.jsonl    # Per-environment cycle events
│       │   └── policy_trace.jsonl
│       ├── A2/
│       ├── A3/
│       ├── A4/
│       └── baseline/
```

#### 6.3.2 Snapshotting Requirements

> **Snapshot Requirements:**
> - Snapshots MUST be created at configured intervals (cycle count, time, on error)
> - Snapshots MUST include complete PRNG state for all components
> - Snapshots MUST include content hash for integrity verification
> - Snapshot restore MUST verify hash before loading state
> - Minimum 2 snapshots retained for any in-progress run

U2 runs MUST create snapshots at defined intervals for reproducibility and failure recovery:

**Snapshot Configuration:**

```yaml
# config/u2_snapshots.yaml (PHASE II)
snapshots:
  # When to create snapshots
  triggers:
    cycle_interval: 100           # Every N cycles
    time_interval_minutes: 30     # Every N minutes
    on_error: true                # On any error
    on_completion: true           # At run end
    on_signal: true               # On SIGUSR1 (manual trigger)

  # What to include
  contents:
    prng_state: true              # All PRNG checkpoints
    policy_weights: true          # Current policy parameters
    statement_frontier: true      # Active statement set
    derivation_queue: true        # Pending derivations
    cycle_counter: true           # Current cycle number
    metrics_accumulator: true     # Running metric totals

  # Storage
  storage:
    format: "msgpack"             # Binary for efficiency
    compression: "zstd"           # Fast compression
    directory: "${MATHLEDGER_SNAPSHOT_ROOT}/u2/${run_id}"
    filename_pattern: "snapshot_{cycle:08d}_{timestamp}.snap"

  # Retention
  retention:
    keep_last_n: 10               # Always keep last N snapshots
    keep_interval_cycles: 1000    # Keep one every N cycles
    max_age_days: 90              # Maximum retention
```

**Snapshot Schema:**

```python
# backend/rfl/snapshot.py (PHASE II)
from dataclasses import dataclass
from typing import Any
import msgpack
import zstandard as zstd
import hashlib

@dataclass
class U2Snapshot:
    """Complete state snapshot for U2 run reproducibility."""

    # Identity
    run_id: str
    environment_id: str
    cycle: int
    timestamp: str                  # ISO8601 UTC

    # State components
    prng_checkpoints: dict[str, bytes]   # component -> serialized state
    policy_weights: dict[str, float]     # parameter -> value
    statement_frontier: list[str]        # Statement hashes in frontier
    derivation_queue: list[dict]         # Pending derivation tasks
    metrics_accumulator: dict[str, Any]  # Running totals

    # Integrity
    content_hash: str                    # SHA256 of serialized content

    def serialize(self) -> bytes:
        """Serialize snapshot with compression."""
        # Compute hash before compression
        content = msgpack.packb(self.__dict__, use_bin_type=True)
        self.content_hash = hashlib.sha256(content).hexdigest()

        # Compress
        cctx = zstd.ZstdCompressor(level=3)
        return cctx.compress(content)

    @classmethod
    def deserialize(cls, data: bytes) -> "U2Snapshot":
        """Deserialize and validate snapshot."""
        dctx = zstd.ZstdDecompressor()
        content = dctx.decompress(data)

        obj = msgpack.unpackb(content, raw=False)
        snapshot = cls(**obj)

        # Validate integrity
        expected_hash = snapshot.content_hash
        snapshot.content_hash = ""
        actual_content = msgpack.packb(snapshot.__dict__, use_bin_type=True)
        actual_hash = hashlib.sha256(actual_content).hexdigest()

        if actual_hash != expected_hash:
            raise IntegrityError(
                f"Snapshot integrity check failed: "
                f"expected {expected_hash}, got {actual_hash}"
            )

        snapshot.content_hash = expected_hash
        return snapshot
```

#### 6.3.3 Telemetry Export

U2 runs MUST export telemetry in a format suitable for analysis:

**Export Formats:**

| Format | Use Case | Contents |
|--------|----------|----------|
| `events.jsonl` | Raw event stream | All structured events |
| `summary.json` | Run overview | Aggregate metrics, config |
| `policy_trajectory.parquet` | Policy analysis | Policy weights over time |
| `derivation_dag.parquet` | DAG analysis | Statement lineage |
| `reproducibility.tar.zst` | Full reproduction | Snapshots + config + logs |

**Telemetry Configuration:**

```yaml
# config/u2_telemetry.yaml (PHASE II)
telemetry:
  # Real-time streaming (optional)
  streaming:
    enabled: false                # Enable for production monitoring
    endpoint: null                # Telemetry collector endpoint
    batch_size: 100
    flush_interval_seconds: 10

  # Local export (required)
  export:
    directory: "${MATHLEDGER_EXPORT_ROOT}/u2/${run_id}"
    formats:
      - jsonl                     # Always required
      - parquet                   # For analysis

    # What to export
    include:
      events: true
      metrics: true
      snapshots: true
      config: true
      preregistration: true       # Copy of PREREG_UPLIFT_U2.yaml

  # Validation
  validation:
    require_complete_cycle_events: true
    require_snapshot_hashes: true
    require_prng_determinism_check: true
```

### 6.4 Environment Variable Requirements

> **PHASE II — NOT RUN IN PHASE I**

The following environment variables are required for U2 uplift experiments:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RFL_ENV_MODE` | Yes | `phase1-hermetic` | Environment mode selector |
| `MATHLEDGER_CACHE_ROOT` | Yes | None | Root directory for isolated caches |
| `MATHLEDGER_SNAPSHOT_ROOT` | Yes | None | Root directory for snapshots |
| `MATHLEDGER_EXPORT_ROOT` | Yes | None | Root directory for telemetry exports |
| `U2_MASTER_SEED` | Yes | None | Master seed (hex string, from PREREG hash) |
| `U2_RUN_ID` | Yes | None | Unique run identifier (UUID v5) |
| `U2_CYCLE_LIMIT` | No | 1000 | Maximum cycles per run |
| `U2_SNAPSHOT_INTERVAL` | No | 100 | Cycles between snapshots |

**Environment Validation:**

```python
# backend/rfl/env_validator.py (PHASE II)
def validate_u2_environment() -> dict[str, str]:
    """Validate all required U2 environment variables."""
    from backend.rfl.env_mode import require_mode, RFLEnvMode

    require_mode(RFLEnvMode.UPLIFT_EXPERIMENT)

    required = [
        "MATHLEDGER_CACHE_ROOT",
        "MATHLEDGER_SNAPSHOT_ROOT",
        "MATHLEDGER_EXPORT_ROOT",
        "U2_MASTER_SEED",
        "U2_RUN_ID",
    ]

    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        raise EnvironmentError(
            f"Missing required U2 environment variables: {missing}"
        )

    # Validate master seed format (64 hex chars = SHA256)
    seed = os.environ["U2_MASTER_SEED"]
    if not re.match(r"^[0-9a-f]{64}$", seed, re.IGNORECASE):
        raise ValueError(
            f"U2_MASTER_SEED must be 64 hex characters (SHA256), got: {len(seed)} chars"
        )

    # Validate directories exist and are writable
    for var in ["MATHLEDGER_CACHE_ROOT", "MATHLEDGER_SNAPSHOT_ROOT", "MATHLEDGER_EXPORT_ROOT"]:
        path = Path(os.environ[var])
        if not path.exists():
            path.mkdir(parents=True, mode=0o700)
        if not os.access(path, os.W_OK):
            raise PermissionError(f"{var} directory not writable: {path}")

    return {var: os.environ[var] for var in required}
```

### 6.5 Phase II Environment Hardening Summary

| Requirement | Enforcement Point | Validation | NFR |
|-------------|-------------------|------------|-----|
| Mode declaration | `RFL_ENV_MODE` env var | Enforcer at startup | NFR-001 |
| Cache isolation | Per-run directories | Path validation, no symlinks | NFR-002 |
| PRNG determinism | `DeterministicPRNG` class | Checkpoint comparison | NFR-003, NFR-006 |
| Structured logging | `U2LogEntry` schema | Schema validation | NFR-004 |
| Snapshot integrity | Content hashing | Hash verification on restore | NFR-005 |
| No banned randomness | Static analysis + guards | Runtime detection | NFR-007 |
| Export completeness | Required cycle events | Event audit at run end | — |

**Pre-Run Checklist (Phase II U2):**

- [ ] `RFL_ENV_MODE=uplift_experiment` set
- [ ] `MATHLEDGER_CACHE_ROOT` configured and writable
- [ ] `MATHLEDGER_SNAPSHOT_ROOT` configured and writable
- [ ] `MATHLEDGER_EXPORT_ROOT` configured and writable
- [ ] `U2_MASTER_SEED` set to SHA256 of `PREREG_UPLIFT_U2.yaml`
- [ ] `U2_RUN_ID` set to unique UUID v5
- [ ] All PRNG components use `DeterministicPRNG`
- [ ] Logging schema validation enabled
- [ ] Snapshot triggers configured
- [ ] No banned randomness sources in execution path
- [ ] Preregistration file copied to run export directory

---

## Appendix B: Limitations

**What the current enforcer does NOT check (Phase II future work):**
- Password entropy beyond banned list (no pattern detection, no dictionary check)
- Sequential character detection (`abc123`, `qwerty`)
- Mixed character class enforcement (uppercase + lowercase + digits + symbols)
- Minimum unique characters for passwords (only checked for API keys)
- Docker binding verification (localhost only)
- Resource limit configuration
- TLS/SSL certificate validation
