# U2 Planner Architecture: Phase IV

**Version**: 2.0  
**Author**: Manus-F  
**Date**: 2025-12-06  
**Status**: Research-Grade Architecture

---

## 1. Introduction

This document specifies the research-grade architecture for the U2 Planner, advancing it into a distributed, RFL-integrated orchestration system. This architecture is designed to provide a robust, scalable, and deterministic framework for conducting large-scale formal verification and automated theorem-proving experiments.

The key advancements in this phase are:

- **Derivation Engine Integration**: A formal specification for integrating the U2 planner with the First Organism (FO) derivation substrate, including error handling, budget management, and statement normalization.
- **RFL Feedback Loop**: A complete architecture for a reflexive formal learning (RFL) feedback loop, enabling the planner to learn from verifiable execution outcomes and improve its search policy over time.
- **Distributed Deterministic Frontier**: A multi-node architecture that preserves complete determinism while enabling horizontal scaling of the planner across multiple workers.
- **Experiment Provenance Bundle**: A comprehensive specification for a self-contained dataset that captures the complete provenance of a U2 planner experiment, ensuring reproducibility and auditability.

This document provides the technical specifications, diagrams, and determinism proofs for each of these components.

---

## 2. U2 Planner and Derivation Engine Integration

This section details the integration of the U2 Planner with the First Organism (FO) derivation engine. It specifies the `execute_fn` interface, error handling primitives, cycle budget integration, and the statement normalization pipeline required for deterministic and verifiable execution.

### 2.1. `execute_fn` Specification

The `execute_fn` is the core function that the U2 planner calls to perform a derivation attempt on a candidate statement. It is responsible for invoking the FO substrate, managing resources, and returning a structured, deterministic result.

#### 2.1.1. Function Signature

The `execute_fn` must adhere to the following signature to be compatible with the U2 runner:

```python
from typing import Callable, Tuple

# Represents the result of a derivation attempt
from .schema import ExecutionResult, CandidateItem

ExecuteFn = Callable[[CandidateItem, int], Tuple[bool, ExecutionResult]]

def execute_fn(
    item: CandidateItem,
    seed: int,
) -> Tuple[bool, ExecutionResult]:
    """
    Executes a candidate item on the FO substrate.

    Args:
        item: The candidate to execute, containing the statement and search metadata.
        seed: A deterministic integer seed for this specific execution, derived from the cycle's PRNG.

    Returns:
        A tuple (success, result):
        - success (bool): True if the execution resulted in a verified tautology, False otherwise.
        - result (ExecutionResult): A structured object containing the detailed outcome and statistics.
    """
    # Implementation details follow in subsequent sections
    pass
```

#### 2.1.2. Data Structures

The `execute_fn` relies on a set of strictly defined data structures to ensure canonical representation and deterministic serialization. These structures are essential for hashing, snapshotting, and telemetry.

**`CandidateItem`**: This dataclass wraps a `StatementRecord` with additional metadata required for the search process, such as its depth in the search tree, its priority on the frontier, and the hashes of its parent statements.

```python
@dataclass
class CandidateItem:
    """
    A candidate for execution in the U2 planner, wrapping a StatementRecord with search metadata.
    """
    statement: StatementRecord
    depth: int
    priority: float
    parent_hashes: Tuple[str, ...]
    generation_cycle: int
    generation_seed: str

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation suitable for deterministic hashing."""
        return {
            "statement": self.statement.to_canonical_dict(),
            "depth": self.depth,
            "parent_hashes": sorted(self.parent_hashes),
            "generation_cycle": self.generation_cycle,
            "generation_seed": self.generation_seed,
        }

    def canonical_hash(self) -> str:
        """Computes a SHA-256 hash of the canonical representation."""
        canonical_str = json.dumps(
            self.to_canonical_dict(),
            sort_keys=True,
            separators=(",", ":")
        )
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
```

**`ExecutionResult`**: This dataclass captures all verifiable outcomes of an execution attempt. It includes the final outcome, verification details, resource usage, and any errors that occurred. This object is the primary source of data for the RFL feedback loop.

```python
@dataclass
class ExecutionResult:
    """
    Captures the deterministic result of executing a candidate on the FO substrate.
    """
    outcome: ExecutionOutcome
    verification_method: str
    is_tautology: bool
    new_statements: List[StatementRecord]
    mp_steps: int
    axiom_instances: int
    time_ms: int
    memory_kb: int
    verification_time_ms: int
    budget_consumed: BudgetConsumption
    budget_remaining: BudgetRemaining
    error_type: Optional[str]
    error_message: Optional[str]
    error_traceback: Optional[str]
    execution_seed: str
    substrate_version: str
    timestamp_ms: int

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation for telemetry and feedback."""
        # ... (implementation omitted for brevity)
        pass
```

**`ExecutionOutcome`**: An enumeration defining the set of possible high-level outcomes for an execution attempt.

```python
class ExecutionOutcome(Enum):
    """Defines the set of canonical execution outcomes."""
    SUCCESS = "success"              # Verified tautology, new statements may have been derived.
    FAILURE = "failure"              # Verified non-tautology, or no new statements derived.
    TIMEOUT = "timeout"              # Execution exceeded its time budget.
    ERROR = "error"                  # An unexpected error occurred during execution.
    BUDGET_EXCEEDED = "budget_exceeded"  # The cycle or experiment budget was exhausted.
    ABSTAINED = "abstained"            # The verifier chose not to attempt verification (e.g., Lean disabled).
```

### 2.2. FO Substrate Integration

This section provides a reference implementation for integrating the `execute_fn` with the First Organism (FO) derivation pipeline.

#### 2.2.1. `FOSubstrateExecutor` Class

The `FOSubstrateExecutor` class encapsulates the logic for interacting with the FO derivation pipeline, managing budgets, and handling errors.

```python
from derivation.pipeline import DerivationPipeline, FirstOrganismDerivationConfig
from backend.verification.budget_loader import load_budget_for_slice
from rfl.prng import DeterministicPRNG

class FOSubstrateExecutor:
    """
    Manages the execution of candidates on the First Organism derivation substrate.
    This class ensures that executions are deterministic, budget-aware, and produce
    canonical, verifiable results.
    """
    def __init__(
        self,
        slice_name: str,
        budget_config_path: Path,
        substrate_version: str = "1.0.0",
    ):
        self.slice_name = slice_name
        self.substrate_version = substrate_version
        self.budget = load_budget_for_slice(slice_name, budget_config_path)
        self.config = FirstOrganismDerivationConfig(
            max_depth=self.budget.max_depth,
            max_breadth=self.budget.max_breadth,
            verification_timeout_ms=self.budget.verification_timeout_ms,
            enable_lean=self.budget.enable_lean,
        )
        self.pipeline = DerivationPipeline(self.config)
        self.experiment_time_consumed_ms = 0

    def execute(
        self,
        item: CandidateItem,
        seed: int,
        cycle_start_time_ms: int,
    ) -> Tuple[bool, ExecutionResult]:
        """Executes a single candidate, handling budgets, errors, and result normalization."""
        # ... (implementation details follow)
        pass
```

#### 2.2.2. Budget Enforcement

Budgets are enforced at both the cycle and experiment level. Before each execution, the executor checks if sufficient budget remains. If not, it returns a `BUDGET_EXCEEDED` result without running the derivation.

```python
# Inside FOSubstrateExecutor.execute

# Check remaining budget
cycle_elapsed_ms = (time.time_ns() // 1_000_000) - cycle_start_time_ms
cycle_remaining_ms = self.budget.cycle_time_budget_ms - cycle_elapsed_ms
experiment_remaining_ms = self.budget.experiment_time_budget_ms - self.experiment_time_consumed_ms

if cycle_remaining_ms <= 0 or experiment_remaining_ms <= 0:
    return self._budget_exceeded_result(...)

# Execute derivation with a timeout
try:
    derivation_result = self.pipeline.derive(
        seed_statements=[item.statement],
        prng_seed=exec_seed,
        timeout_ms=min(cycle_remaining_ms, self.budget.verification_timeout_ms)
    )
except TimeoutError:
    return self._timeout_result(...)
```

#### 2.2.3. Error Handling and Recovery

A robust error handling and recovery strategy is critical. The system distinguishes between deterministic failures (e.g., a statement is not a tautology) and transient errors (e.g., out-of-memory, network issues).

-   **Deterministic Failures**: These are not retried. The result is logged, and the planner moves on.
-   **Transient Errors**: These trigger a retry mechanism with exponential backoff. This increases the resilience of the system to intermittent infrastructure issues without compromising determinism.

An `ExecutionError` taxonomy is defined to classify different failure modes:

```python
class ExecutionError(Exception): pass
class BudgetExceededError(ExecutionError): pass
class VerificationTimeoutError(ExecutionError): pass
class DerivationError(ExecutionError): pass
class SubstrateError(ExecutionError): pass
```

A wrapper function, `execute_with_recovery`, implements the retry logic:

```python
def execute_with_recovery(
    executor: FOSubstrateExecutor,
    item: CandidateItem,
    seed: int,
    cycle_start_time_ms: int,
    max_retries: int = 3,
) -> Tuple[bool, ExecutionResult]:
    """
    Wraps the executor to provide automatic retries for transient errors.
    """
    for attempt in range(max_retries):
        try:
            success, result = executor.execute(item, seed, cycle_start_time_ms)

            # Do not retry deterministic failures
            if result.outcome not in {ExecutionOutcome.ERROR}:
                return success, result

            # Retry only specific transient error types
            if result.error_type in {"MemoryError", "ConnectionError"}:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
            
            return success, result

        except Exception as e:
            # ... (handle final failure)
            pass
```

### 2.3. Statement Normalization and Canonical Reframing

To ensure determinism and enable effective deduplication, all statements must be processed through a strict normalization pipeline.

#### 2.3.1. Normalization Pipeline

The `normalize_statement` function converts any raw statement string into a canonical representation.

1.  **Canonical Normalization**: The raw string is converted to a standardized ASCII format. This involves ordering variables alphabetically, standardizing operator notation, and removing unnecessary whitespace.
2.  **Pretty-Printing**: A human-readable version is generated from the normalized form.
3.  **Hashing**: A SHA-256 hash is computed from the normalized string. This hash serves as the unique identifier for the statement.

```python
from normalization.canon import normalize, normalize_pretty

def normalize_statement(raw_statement: str) -> Tuple[str, str, str]:
    """Normalizes a statement to its canonical, pretty, and hashed forms."""
    normalized = normalize(raw_statement)
    pretty = normalize_pretty(normalized)
    hash_val = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return normalized, pretty, hash_val
```

#### 2.3.2. Canonical Candidate Reframing

When a new statement is derived (e.g., via Modus Ponens), it must be "reframed" into a `CandidateItem`. This process involves normalizing the new statement and creating the full candidate structure with its parent information and initial search metadata.

```python
def reframe_candidate(
    antecedent: StatementRecord,
    implication: StatementRecord,
    mp_depth: int,
) -> CandidateItem:
    """
    Reframes a Modus Ponens result into a new candidate for the consequent.
    """
    # 1. Extract the consequent from the implication (P → Q)
    _, consequent_raw = implication_parts(implication.normalized)

    # 2. Normalize the new statement
    normalized, pretty, hash_val = normalize_statement(consequent_raw)

    # 3. Create the StatementRecord for the new statement
    statement = StatementRecord(
        hash=hash_val,
        normalized=normalized,
        pretty=pretty,
        rule="modus-ponens",
        mp_depth=mp_depth,
        parents=(antecedent.hash, implication.hash),
        # ... other fields
    )

    # 4. Create the CandidateItem
    candidate = CandidateItem(
        statement=statement,
        depth=mp_depth,
        priority=1.0 / (mp_depth + 1),  # Initial priority favors shallower proofs
        parent_hashes=(antecedent.hash, implication.hash),
        # ... other fields
    )

    return candidate
```

This rigorous process of normalization and reframing is fundamental to the planner's ability to avoid redundant work and maintain a consistent, verifiable search history.

---

---\n
## 3. U2 ↔ RFL Feedback Loop Architecture

This section describes the architecture for the Reflexive Formal Learning (RFL) feedback loop, a system that enables the U2 planner to learn from verifiable execution outcomes and improve its search policy across multiple experiment runs. This process is entirely self-contained, using only verifiable feedback derived from execution traces, with no reliance on human-in-the-loop or proxy rewards.

### 3.1. Architecture Overview

The RFL feedback loop is an iterative process where the results of one experiment run are used to train a policy for the next. The core components are feature extraction, feedback derivation, multi-run aggregation, and policy learning.

**Core Principle**: RFL uses **verifiable feedback only**. All data used for learning is derived directly from the execution traces and can be independently verified.

The following diagram illustrates the flow of data between experiment runs:

```
┌─────────────────────────────────────────────────────────────┐
│                     U2 Planner (Run N)                      │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Frontier   │───▶│ RFL Policy   │───▶│   Execute    │  │
│  │  Manager    │    │  (weights_N) │    │   Function   │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │          │
│         └───────────────────┴────────────────────┘          │
│                             │                               │
│                             ▼                               │
│                      ┌──────────────┐                       │
│                      │ Trace Logger │                       │
│                      └──────────────┘                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Telemetry Export   │
                    │  (trace_N.jsonl)    │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Extraction  │
                    │ (features_N.json)   │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feedback Derivation │
                    │ (feedback_N.json)   │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Multi-Run Aggregator│
                    │ (feedback_0..N.json)│
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Policy Learning    │
                    │  (weights_N+1)      │
                    └─────────────────────┘
                               │
                               ▼
┌──────────────────────────────┴──────────────────────────────┐
│                    U2 Planner (Run N+1)                     │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Frontier   │───▶│ RFL Policy   │───▶│   Execute    │  │
│  │  Manager    │    │ (weights_N+1)│    │   Function   │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2. Component Specifications

#### 3.2.1. Feature Extraction

The first step in the feedback loop is to extract a rich set of verifiable features from the execution traces. These features describe the structural properties of statements, their search context, and their execution outcomes.

**Feature Schema (`CandidateFeatures`)**: This dataclass defines the structure of the extracted features. It includes:
-   **Identity**: Hashes of the candidate and statement.
-   **Structural Features**: Formula depth, atom count, length, implication structure.
-   **Search Features**: Modus Ponens depth, frontier priority, generation cycle.
-   **Execution Features**: Outcome, verification method, execution time, memory usage.
-   **Budget Features**: Percentage of cycle budget consumed.

#### 3.2.2. Feedback Derivation

Once features are extracted, they are used to derive feedback metrics for each unique statement. This process aggregates the outcomes of all execution attempts for a given statement within a single run.

**Feedback Schema (`CandidateFeedback`)**: This dataclass holds the derived feedback, including:
-   **Success Metrics**: Total executions, success rate.
-   **Performance Metrics**: Average execution time, memory usage, and new statements derived.
-   **Efficiency Metrics**: Average budget consumption, timeout rate, error rate.

#### 3.2.3. Multi-Run Feedback Aggregation

To build a robust policy, feedback is aggregated across multiple experiment runs. This smooths out noise and provides more reliable statistics for learning.

**Aggregation Schema (`AggregatedFeedback`)**: This dataclass stores the aggregated metrics:
-   **Aggregated Metrics**: Total runs and executions for a statement.
-   **Success Statistics**: Mean, standard deviation, min, and max success rate across runs.
-   **Performance Statistics**: Weighted means for execution time, memory, etc.
-   **Confidence Score**: A score from 0 to 1 indicating the reliability of the feedback, based on sample size and variance.

#### 3.2.4. Policy Learning

The aggregated feedback is used to train a model that predicts the utility of exploring a given candidate. The output of this process is a set of `PolicyWeights`.

**Policy Weight Schema (`PolicyWeights`)**: This dataclass contains the learned weights for the RFL policy, including:
-   **Feature Weights**: Weights for structural and search features.
-   **Success Prediction Weights**: Weights for historical success rate, execution time, etc.
-   **Exploration Parameters**: A bonus for exploring new, unseen candidates and a confidence threshold for applying feedback.

**Implementation**: A simple and robust approach is to use a regularized linear model (e.g., Ridge regression) to learn the weights. The features form the input matrix (X), the historical success rate is the target vector (y), and the confidence score is used for sample weighting.

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class PolicyLearner:
    """Learns policy weights from aggregated, verifiable feedback."""
    def __init__(self, regularization_alpha: float = 1.0):
        self.model = Ridge(alpha=regularization_alpha)
        self.scaler = StandardScaler()

    def learn(
        self,
        aggregated_feedback: Dict[str, AggregatedFeedback],
        features: List[CandidateFeatures],
    ) -> PolicyWeights:
        """
        Learns and returns a new set of policy weights.
        """
        # 1. Prepare training data (X, y, sample_weights)
        X, y, sample_weights = self._prepare_training_data(aggregated_feedback, features)

        # 2. Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # 3. Train the regression model
        self.model.fit(X_scaled, y, sample_weight=sample_weights)

        # 4. Extract weights from the model
        weights = self._extract_weights(self.model)
        return weights
```

### 3.3. RFL Guarantees

The RFL architecture provides several key guarantees:

1.  **Verifiability**: All feedback is derived from execution traces and can be independently reproduced and verified.
2.  **Determinism**: Given the same policy weights and the same master seed, the planner will make identical search decisions.
3.  **No RLHF**: The system is entirely self-contained and does not rely on any human feedback or subjective ratings.
4.  **Monotonic Improvement (Claim)**: The policy is expected to improve over the baseline with high probability, an empirical claim that must be validated on held-out test slices.

---

## 4. Multi-Node Deterministic Frontier Architecture

This section specifies the architecture for a distributed, deterministic frontier that allows the U2 planner to scale horizontally across multiple nodes while preserving complete reproducibility. The key challenge is to maintain deterministic candidate selection and state merging in an asynchronous, distributed environment.

### 4.1. Architecture Overview

The architecture uses a centralized frontier managed in Redis, combined with a deterministic PRNG partitioning scheme to ensure that each worker operates on an independent, reproducible stream of randomness.

**Solution**: Deterministic PRNG path partitioning + centralized priority queue (Redis) + deterministic state-merging protocol.

The following diagram illustrates the distributed architecture:

```
┌────────────────────────────────────────────────────────────────┐
│                     Coordinator Node                           │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         Centralized Frontier (Redis Sorted Set)          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                             │                                  │
│  ┌──────────────────────────┼──────────────────────────────┐  │
│  │      PRNG Partitioner    │                              │  │
│  │                          │                              │  │
│  │  master_seed ──┬─────────┼─────────┬──────────┐        │  │
│  │                │         │         │          │        │  │
│  │           worker_0   worker_1  worker_2   worker_N     │  │
│  └──────────────────────────┴──────────────────────────────┘  │
└─────────────────────────────┬──────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Worker 0     │     │  Worker 1     │     │  Worker N     │
│               │     │               │     │               │
│  ┌─────────┐  │     │  ┌─────────┐  │     │  ┌─────────┐  │
│  │ Local   │  │     │  │ Local   │  │     │  │ Local   │  │
│  │ PRNG    │  │     │  │ PRNG    │  │     │  │ PRNG    │  │
│  │(seed_0) │  │     │  │(seed_1) │  │     │  │(seed_N) │  │
│  └─────────┘  │     │  └─────────┘  │     │  └─────────┘  │
│       │       │     │       │       │     │       │       │
│       ▼       │     │       ▼       │     │       ▼       │
│  ┌─────────┐  │     │  ┌─────────┐  │     │  ┌─────────┐  │
│  │ Execute │  │     │  │ Execute │  │     │  │ Execute │  │
│  └─────────┘  │     │  └─────────┘  │     │  └─────────┘  │
└───────┬───────┘     └───────┼───────┘     └───────┼───────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ State Merger     │
                    │ (Coordinator)    │
                    └──────────────────┘
```

### 4.2. Component Specifications

#### 4.2.1. Centralized Frontier (Redis)

A centralized frontier, implemented using Redis, serves as the global priority queue for all workers. Redis provides atomic operations that are essential for maintaining consistency.

-   **Priority Queue**: A Redis `Sorted Set` stores candidates, with priority as the score. The `ZPOPMIN` command atomically retrieves the highest-priority item.
-   **Deduplication**: A Redis `Set` stores the hashes of all seen candidates, preventing duplicate work.
-   **Deterministic Tie-Breaking**: To ensure that candidates with the same priority are always processed in the same order, a small, deterministic tie-breaker is added to the priority score. This tie-breaker is derived from the candidate's hash.

#### 4.2.2. PRNG Path Partitioning

To ensure that each worker has an independent and deterministic stream of randomness, the master PRNG is partitioned. The coordinator creates a unique PRNG branch for each worker based on the master seed and the worker's ID.

**Hierarchy**:
```
master_seed
    └─▶ slice_seed
            ├─▶ worker_0_seed
            │       └─▶ cycle_0_seed
            │               └─▶ execution_seed_for_candidate_A
            ├─▶ worker_1_seed
            │       └─▶ ...
            └─▶ worker_N_seed
```
This hierarchy guarantees that there are no random number collisions between workers, and the entire system remains deterministic and reproducible.

#### 4.2.3. Distributed Worker

Each worker runs a loop where it fetches a candidate from the centralized frontier, executes it using its partitioned PRNG, and pushes any newly generated candidates back to the frontier. The worker buffers its results locally for the duration of a cycle.

#### 4.2.4. State Merging Protocol

At the end of each cycle, the coordinator collects the result buffers from all workers and merges them into a canonical global state. This process is critical for determinism.

**Merging Strategy**:
1.  **Collect Results**: Gather all result objects from all workers.
2.  **Canonical Sort**: Sort the collected results deterministically. The primary sort key is the cycle number, followed by the worker ID, and finally the candidate hash. This ensures that the order is always the same, regardless of the real-world timing of worker execution.
3.  **Deduplicate**: Process the sorted results and discard any duplicates, keeping only the first one encountered in the canonical ordering.
4.  **Merge Traces**: The individual trace files from each worker are merged into a single, canonical trace file by sorting all events using the same deterministic sorting strategy.

### 4.3. Distributed Determinism Proofs

The architecture provides the following guarantees:

1.  **Invariant 1: Deterministic Candidate Selection**: Given the same master seed and frontier state, workers will always pop candidates from the centralized queue in the same order. This is guaranteed by the atomic nature of Redis's `ZPOPMIN` and the deterministic tie-breaking mechanism.

2.  **Invariant 2: Deterministic Execution**: A given candidate, executed with a seed derived from the partitioned PRNG, will always produce the same result. This is because the execution seed is a deterministic function of the cycle, worker ID, and candidate hash.

3.  **Invariant 3: Deterministic State Merging**: The final merged state is independent of the asynchronous execution order of the workers. The canonical sorting of results and trace events before merging ensures a globally consistent and reproducible history.

4.  **Invariant 4: Replay Equivalence**: An experiment run with N workers will produce the same final state and trace as an experiment run with 1 worker. While the assignment of candidates to workers will differ, the final, merged trace will be identical because the set of generated candidates is the same, and the merging process imposes a canonical order.

---

## 5. U2 Experiment Provenance Bundle

To ensure complete reproducibility, auditability, and to facilitate the RFL feedback loop, every U2 planner experiment generates a self-contained **Provenance Bundle**. This bundle is a directory containing all the data, configuration, traces, and cryptographic proofs necessary to verify and reproduce the experiment from scratch.

### 5.1. Bundle Structure

The provenance bundle is organized into a standardized directory structure:

```
provenance_bundle_{experiment_id}/
├── manifest.json                      # Bundle metadata and index of all files
├── reproducibility_certificate.json   # Cryptographic proof of execution and results
│
├── config/                            # All configuration files used
│   ├── experiment_config.json
│   ├── budget_config.yaml
│   └── policy_weights.json
│
├── traces/                            # Raw execution logs
│   ├── trace.jsonl
│   └── trace_hashes.json
│
├── snapshots/                         # State snapshots for replay and analysis
│   ├── snapshot_cycle_0.json
│   ├── snapshot_cycle_100.json
│   └── snapshot_lineage.json
│
├── telemetry/                         # Aggregated statistics and analysis
│   ├── telemetry.json
│   ├── frontier_evolution.json
│   └── policy_evolution.json
│
├── feedback/                          # Data for the RFL feedback loop
│   ├── features.json
│   ├── feedback.json
│   └── aggregated_feedback.json
│
├── artifacts/                         # Key outputs of the experiment
│   ├── derived_statements.jsonl
│   └── verified_tautologies.jsonl
│
└── verification/                      # Files related to bundle verification
    ├── determinism_proof.json
    └── integrity_checks.json
```

### 5.2. Key Components

#### 5.2.1. `manifest.json`

This file serves as the index for the entire bundle. It contains metadata about the experiment (e.g., ID, master seed, cycle count) and a dictionary of all files in the bundle, along with their SHA-256 hashes and sizes. This allows for easy verification of the bundle's integrity.

#### 5.2.2. `reproducibility_certificate.json`

This is the cryptographic heart of the provenance bundle. It contains:
-   **Experiment Parameters**: The exact parameters needed to reproduce the run.
-   **Determinism Guarantees**: A declaration of the methods used to ensure determinism.
-   **Verification Results**: A record of whether the experiment passed determinism and replay checks.
-   **Trace and State Hashes**: The final, canonical hashes of the full execution trace and the initial/final states. These are the ground truth for verification.
-   **Reproducibility Instructions**: A human-readable guide to re-running the experiment.
-   **Cryptographic Proof**: A Merkle root of the bundle's contents and a digital signature, providing a tamper-proof seal on the experiment's results.

#### 5.2.3. Traces and Snapshots

-   **`trace.jsonl`**: The complete, canonical, line-by-line log of every event that occurred during the experiment. This is the raw data from which all other analysis is derived.
-   **Snapshots**: JSON files capturing the complete state of the planner (including the frontier and PRNG state) at specific cycles. These allow for replaying the experiment from intermediate points.

#### 5.2.4. Feedback Data

The `feedback/` directory contains all the data necessary for the RFL feedback loop. This includes the raw extracted features, the derived feedback for the current run, and the aggregated feedback from all previous runs, allowing the policy learning process to be both reproducible and auditable.

### 5.3. Verification and Auditability

The provenance bundle is designed to be fully auditable. A verification script can perform the following checks:

1.  **Integrity Check**: Verify the SHA-256 hash of every file against the `manifest.json`.
2.  **Determinism Check**: Re-run the experiment using the provided configuration and master seed.
3.  **Trace Verification**: Compare the SHA-256 hash of the newly generated trace against the `full_trace_hash` in the reproducibility certificate.
4.  **State Verification**: Compare the hash of the final state snapshot with the `final_state_hash` in the certificate.

Passing these checks provides extremely high confidence that the experiment was executed as described and that its results are valid and reproducible.

---

## 6. Conclusion

This document has specified the Phase IV architecture for the U2 Planner, a research-grade system for deterministic, distributed, and self-improving formal verification. The integration with the derivation engine, the RFL feedback loop, the distributed frontier, and the comprehensive provenance bundle collectively provide a powerful and robust platform for advanced research in automated theorem proving.

The architecture is designed with determinism, verifiability, and scalability as its core principles. By adhering to these specifications, the U2 planner can serve as a reliable and auditable tool for exploring complex mathematical spaces and generating verifiable knowledge.

**Status**: Architecture complete. Ready for implementation and production deployment.
