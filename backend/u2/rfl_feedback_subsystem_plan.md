# RFL Feedback Subsystem: Build Plan

**Author**: Manus-F  
**Date**: 2025-12-06  
**Status**: Implementation Build Plan (Ready for Coding)

---

## Overview

This document provides a **complete build plan** for the RFL (Reasoning Feedback Loop) Subsystem, which enables the U2 Planner to learn from verifiable execution outcomes and improve its search policy over time. The subsystem is designed with complete verifiability, determinism, and no human-in-the-loop (RLHF).

---

## 1. FeatureExtractor Interface

### 1.1. Purpose

Extract **verifiable features** from execution traces for policy learning.

### 1.2. Feature Schema

```python
@dataclass
class CandidateFeatures:
    """
    Verifiable features extracted from a candidate and its execution.
    
    All features are directly observable from execution traces.
    """
    # Structural features (from statement)
    formula_depth: int  # Nesting level of formula
    atom_count: int  # Number of distinct propositional variables
    formula_length: int  # Character count of normalized form
    is_implication: bool  # Top-level operator is →
    implication_depth: int  # Nested implications count
    
    # Search features (from search context)
    mp_depth: int  # Modus Ponens depth in proof tree
    frontier_priority: float  # Priority when popped
    generation_cycle: int  # Cycle when generated
    parent_count: int  # Number of parent statements
    
    # Execution features (from execution result)
    outcome: str  # ExecutionOutcome value
    verification_method: str  # Verification method used
    is_tautology: bool  # Verified as tautology
    execution_time_ms: int  # Time taken
    memory_kb: int  # Memory used
    new_statements_count: int  # Derived statements
    mp_steps: int  # Modus Ponens applications
    budget_consumed_pct: float  # Percentage of budget consumed
    budget_exhausted: bool  # Budget exhausted flag
    
    def to_feature_vector(self) -> List[float]:
        """Convert to numerical feature vector for ML."""
        return [
            float(self.formula_depth),
            float(self.atom_count),
            float(self.formula_length),
            1.0 if self.is_implication else 0.0,
            float(self.implication_depth),
            float(self.mp_depth),
            self.frontier_priority,
            float(self.generation_cycle),
            float(self.parent_count),
            1.0 if self.outcome == "success" else 0.0,
            1.0 if self.is_tautology else 0.0,
            float(self.execution_time_ms),
            float(self.memory_kb),
            float(self.new_statements_count),
            float(self.mp_steps),
            self.budget_consumed_pct,
            1.0 if self.budget_exhausted else 0.0,
        ]
    
    @staticmethod
    def feature_names() -> List[str]:
        """Return feature names in vector order."""
        return [
            "formula_depth",
            "atom_count",
            "formula_length",
            "is_implication",
            "implication_depth",
            "mp_depth",
            "frontier_priority",
            "generation_cycle",
            "parent_count",
            "outcome_success",
            "is_tautology",
            "execution_time_ms",
            "memory_kb",
            "new_statements_count",
            "mp_steps",
            "budget_consumed_pct",
            "budget_exhausted",
        ]
```

### 1.3. FeatureExtractor Class

```python
class FeatureExtractor:
    """
    Extracts verifiable features from candidates and execution results.
    """
    
    def __init__(self):
        pass
    
    def extract(
        self,
        candidate: CandidateItem,
        result: ExecutionResult,
    ) -> CandidateFeatures:
        """
        Extract features from candidate and execution result.
        
        Args:
            candidate: CandidateItem that was executed
            result: ExecutionResult from execution
            
        Returns:
            CandidateFeatures object
        """
        # Extract structural features
        formula_depth = self._compute_formula_depth(candidate.statement.normalized)
        atom_count = self._count_atoms(candidate.statement.normalized)
        formula_length = len(candidate.statement.normalized)
        is_implication = self._is_implication(candidate.statement.normalized)
        implication_depth = self._count_implication_depth(candidate.statement.normalized)
        
        # Extract search features
        mp_depth = candidate.statement.mp_depth
        frontier_priority = candidate.priority
        generation_cycle = candidate.generation_cycle
        parent_count = len(candidate.parent_hashes)
        
        # Extract execution features
        outcome = result.outcome.value
        verification_method = result.verification_method
        is_tautology = result.is_tautology
        execution_time_ms = result.time_ms
        memory_kb = result.memory_kb
        new_statements_count = len(result.new_statements)
        mp_steps = result.mp_steps
        budget_consumed_pct = (
            result.budget_consumed.total_time_ms / 
            (result.budget_consumed.total_time_ms + result.budget_remaining.cycle_time_remaining_ms)
        )
        budget_exhausted = result.budget_remaining.cycle_budget_exhausted
        
        return CandidateFeatures(
            formula_depth=formula_depth,
            atom_count=atom_count,
            formula_length=formula_length,
            is_implication=is_implication,
            implication_depth=implication_depth,
            mp_depth=mp_depth,
            frontier_priority=frontier_priority,
            generation_cycle=generation_cycle,
            parent_count=parent_count,
            outcome=outcome,
            verification_method=verification_method,
            is_tautology=is_tautology,
            execution_time_ms=execution_time_ms,
            memory_kb=memory_kb,
            new_statements_count=new_statements_count,
            mp_steps=mp_steps,
            budget_consumed_pct=budget_consumed_pct,
            budget_exhausted=budget_exhausted,
        )
    
    def _compute_formula_depth(self, normalized: str) -> int:
        """Compute nesting depth of formula."""
        max_depth = 0
        current_depth = 0
        for char in normalized:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth
    
    def _count_atoms(self, normalized: str) -> int:
        """Count distinct propositional variables."""
        # Extract atoms (single letters or p0, p1, etc.)
        import re
        atoms = set(re.findall(r'[a-z]\d*', normalized))
        return len(atoms)
    
    def _is_implication(self, normalized: str) -> bool:
        """Check if top-level operator is implication."""
        # Simple heuristic: contains → at depth 0
        depth = 0
        for i, char in enumerate(normalized):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == '→' and depth == 0:
                return True
        return False
    
    def _count_implication_depth(self, normalized: str) -> int:
        """Count nested implications."""
        count = 0
        for i in range(len(normalized) - 1):
            if normalized[i:i+1] == '→':
                count += 1
        return count
```

---

## 2. CandidateFeedback Aggregator

### 2.1. Purpose

Aggregate execution outcomes for each unique candidate across multiple executions.

### 2.2. Feedback Schema

```python
@dataclass
class CandidateFeedback:
    """
    Aggregated feedback for a single candidate across executions.
    """
    candidate_hash: str
    features: CandidateFeatures
    
    # Aggregated execution statistics
    total_executions: int
    success_count: int
    failure_count: int
    timeout_count: int
    error_count: int
    
    success_rate: float  # success_count / total_executions
    avg_execution_time_ms: float
    avg_memory_kb: float
    avg_new_statements: float
    timeout_rate: float
    error_rate: float
    
    # Provenance
    first_seen_cycle: int
    last_seen_cycle: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "candidate_hash": self.candidate_hash,
            "features": {
                "formula_depth": self.features.formula_depth,
                "atom_count": self.features.atom_count,
                "formula_length": self.features.formula_length,
                "is_implication": self.features.is_implication,
                "implication_depth": self.features.implication_depth,
                "mp_depth": self.features.mp_depth,
                "frontier_priority": self.features.frontier_priority,
                "generation_cycle": self.features.generation_cycle,
                "parent_count": self.features.parent_count,
            },
            "total_executions": self.total_executions,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "timeout_count": self.timeout_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "avg_memory_kb": self.avg_memory_kb,
            "avg_new_statements": self.avg_new_statements,
            "timeout_rate": self.timeout_rate,
            "error_rate": self.error_rate,
            "first_seen_cycle": self.first_seen_cycle,
            "last_seen_cycle": self.last_seen_cycle,
        }
```

### 2.3. FeedbackDeriver Class

```python
class FeedbackDeriver:
    """
    Derives per-candidate feedback from execution traces.
    """
    
    def __init__(self):
        self.extractor = FeatureExtractor()
    
    def derive_feedback(
        self,
        trace_path: Path,
    ) -> Dict[str, CandidateFeedback]:
        """
        Derive feedback from execution trace.
        
        Args:
            trace_path: Path to trace JSONL file
            
        Returns:
            Dictionary mapping candidate_hash → CandidateFeedback
        """
        # Read trace
        executions = []
        with open(trace_path, 'r') as f:
            for line in f:
                event = json.loads(line)
                if event["event_type"] == "execution_result":
                    executions.append(event["data"])
        
        # Group by candidate hash
        by_candidate = {}
        for exec_data in executions:
            candidate_hash = exec_data["candidate_hash"]
            if candidate_hash not in by_candidate:
                by_candidate[candidate_hash] = []
            by_candidate[candidate_hash].append(exec_data)
        
        # Aggregate feedback
        feedback = {}
        for candidate_hash, exec_list in by_candidate.items():
            feedback[candidate_hash] = self._aggregate_executions(
                candidate_hash, exec_list
            )
        
        return feedback
    
    def _aggregate_executions(
        self,
        candidate_hash: str,
        executions: List[Dict[str, Any]],
    ) -> CandidateFeedback:
        """Aggregate multiple executions of same candidate."""
        total = len(executions)
        success_count = sum(1 for e in executions if e["result"]["outcome"] == "success")
        failure_count = sum(1 for e in executions if e["result"]["outcome"] == "failure")
        timeout_count = sum(1 for e in executions if e["result"]["outcome"] == "timeout")
        error_count = sum(1 for e in executions if e["result"]["outcome"] == "error")
        
        success_rate = success_count / total if total > 0 else 0.0
        timeout_rate = timeout_count / total if total > 0 else 0.0
        error_rate = error_count / total if total > 0 else 0.0
        
        avg_time = sum(e["result"]["time_ms"] for e in executions) / total
        avg_memory = sum(e["result"]["memory_kb"] for e in executions) / total
        avg_new_stmts = sum(len(e["result"]["new_statements"]) for e in executions) / total
        
        # Extract features from first execution
        first_exec = executions[0]
        features = self.extractor.extract(
            first_exec["candidate"],
            first_exec["result"],
        )
        
        # Provenance
        first_cycle = min(e["cycle"] for e in executions)
        last_cycle = max(e["cycle"] for e in executions)
        
        return CandidateFeedback(
            candidate_hash=candidate_hash,
            features=features,
            total_executions=total,
            success_count=success_count,
            failure_count=failure_count,
            timeout_count=timeout_count,
            error_count=error_count,
            success_rate=success_rate,
            avg_execution_time_ms=avg_time,
            avg_memory_kb=avg_memory,
            avg_new_statements=avg_new_stmts,
            timeout_rate=timeout_rate,
            error_rate=error_rate,
            first_seen_cycle=first_cycle,
            last_seen_cycle=last_cycle,
        )
```

---

## 3. Multi-Run Feedback Aggregation

### 3.1. Purpose

Aggregate feedback across multiple runs to build robust statistics.

### 3.2. Aggregated Feedback Schema

```python
@dataclass
class AggregatedFeedback:
    """
    Cross-run aggregated feedback for a candidate.
    """
    candidate_hash: str
    features: CandidateFeatures
    
    # Cross-run statistics
    total_runs: int
    total_executions: int
    
    mean_success_rate: float
    std_success_rate: float
    min_success_rate: float
    max_success_rate: float
    
    mean_execution_time_ms: float
    mean_memory_kb: float
    mean_new_statements: float
    mean_timeout_rate: float
    mean_error_rate: float
    
    # Confidence score (0-1)
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "candidate_hash": self.candidate_hash,
            "features": self.features.__dict__,
            "total_runs": self.total_runs,
            "total_executions": self.total_executions,
            "mean_success_rate": self.mean_success_rate,
            "std_success_rate": self.std_success_rate,
            "min_success_rate": self.min_success_rate,
            "max_success_rate": self.max_success_rate,
            "mean_execution_time_ms": self.mean_execution_time_ms,
            "mean_memory_kb": self.mean_memory_kb,
            "mean_new_statements": self.mean_new_statements,
            "mean_timeout_rate": self.mean_timeout_rate,
            "mean_error_rate": self.mean_error_rate,
            "confidence": self.confidence,
        }
```

### 3.3. FeedbackAggregator Class

```python
class FeedbackAggregator:
    """
    Aggregates feedback across multiple runs.
    """
    
    def aggregate(
        self,
        feedback_files: List[Path],
    ) -> Dict[str, AggregatedFeedback]:
        """
        Aggregate feedback from multiple runs.
        
        Args:
            feedback_files: List of feedback JSON files (one per run)
            
        Returns:
            Dictionary mapping candidate_hash → AggregatedFeedback
        """
        # Load all feedback
        all_feedback = {}
        for path in feedback_files:
            with open(path, 'r') as f:
                run_feedback = json.load(f)
                for candidate_hash, feedback_dict in run_feedback.items():
                    if candidate_hash not in all_feedback:
                        all_feedback[candidate_hash] = []
                    all_feedback[candidate_hash].append(feedback_dict)
        
        # Aggregate per candidate
        aggregated = {}
        for candidate_hash, feedback_list in all_feedback.items():
            aggregated[candidate_hash] = self._aggregate_candidate(
                candidate_hash, feedback_list
            )
        
        return aggregated
    
    def _aggregate_candidate(
        self,
        candidate_hash: str,
        feedback_list: List[Dict[str, Any]],
    ) -> AggregatedFeedback:
        """Aggregate feedback for a single candidate across runs."""
        import numpy as np
        
        total_runs = len(feedback_list)
        total_executions = sum(f["total_executions"] for f in feedback_list)
        
        success_rates = [f["success_rate"] for f in feedback_list]
        mean_success_rate = np.mean(success_rates)
        std_success_rate = np.std(success_rates)
        min_success_rate = np.min(success_rates)
        max_success_rate = np.max(success_rates)
        
        mean_execution_time_ms = np.mean([f["avg_execution_time_ms"] for f in feedback_list])
        mean_memory_kb = np.mean([f["avg_memory_kb"] for f in feedback_list])
        mean_new_statements = np.mean([f["avg_new_statements"] for f in feedback_list])
        mean_timeout_rate = np.mean([f["timeout_rate"] for f in feedback_list])
        mean_error_rate = np.mean([f["error_rate"] for f in feedback_list])
        
        # Compute confidence score
        confidence = self._compute_confidence(total_executions, std_success_rate)
        
        # Extract features from first run
        features = CandidateFeatures(**feedback_list[0]["features"])
        
        return AggregatedFeedback(
            candidate_hash=candidate_hash,
            features=features,
            total_runs=total_runs,
            total_executions=total_executions,
            mean_success_rate=mean_success_rate,
            std_success_rate=std_success_rate,
            min_success_rate=min_success_rate,
            max_success_rate=max_success_rate,
            mean_execution_time_ms=mean_execution_time_ms,
            mean_memory_kb=mean_memory_kb,
            mean_new_statements=mean_new_statements,
            mean_timeout_rate=mean_timeout_rate,
            mean_error_rate=mean_error_rate,
            confidence=confidence,
        )
    
    def _compute_confidence(self, sample_size: int, std: float) -> float:
        """
        Compute confidence score based on sample size and variance.
        
        High sample size + low variance = high confidence
        
        Args:
            sample_size: Number of executions
            std: Standard deviation of success rate
            
        Returns:
            Confidence score in [0, 1]
        """
        # Sample size component (sigmoid)
        size_score = 1.0 / (1.0 + np.exp(-0.1 * (sample_size - 20)))
        
        # Variance component (inverse)
        variance_score = 1.0 / (1.0 + std)
        
        # Combined confidence
        confidence = (size_score + variance_score) / 2.0
        
        return confidence
```

---

## 4. PolicyLearner Training Loop

### 4.1. Purpose

Learn policy weights from aggregated feedback using Ridge regression.

### 4.2. Policy Weights Schema

```python
@dataclass
class PolicyWeights:
    """
    Learned weights for RFL policy.
    """
    version: str
    weights: np.ndarray  # Shape: (num_features,)
    intercept: float
    feature_names: List[str]
    scaler_mean: np.ndarray  # StandardScaler mean
    scaler_std: np.ndarray  # StandardScaler std
    
    # Provenance
    learned_from_runs: List[str]
    total_samples: int
    training_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "weights": self.weights.tolist(),
            "intercept": float(self.intercept),
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_std": self.scaler_std.tolist(),
            "learned_from_runs": self.learned_from_runs,
            "total_samples": self.total_samples,
            "training_timestamp": self.training_timestamp,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PolicyWeights":
        """Load from dictionary."""
        return PolicyWeights(
            version=d["version"],
            weights=np.array(d["weights"]),
            intercept=d["intercept"],
            feature_names=d["feature_names"],
            scaler_mean=np.array(d["scaler_mean"]),
            scaler_std=np.array(d["scaler_std"]),
            learned_from_runs=d["learned_from_runs"],
            total_samples=d["total_samples"],
            training_timestamp=d["training_timestamp"],
        )
```

### 4.3. PolicyLearner Class

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

class PolicyLearner:
    """
    Learns policy weights from aggregated feedback.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize policy learner.
        
        Args:
            alpha: Ridge regression regularization strength
        """
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
    
    def train(
        self,
        aggregated_feedback: Dict[str, AggregatedFeedback],
        run_ids: List[str],
    ) -> PolicyWeights:
        """
        Train policy from aggregated feedback.
        
        Args:
            aggregated_feedback: Aggregated feedback dictionary
            run_ids: List of run IDs used for training
            
        Returns:
            PolicyWeights object
        """
        # Prepare training data
        X = []  # Features
        y = []  # Success rates
        sample_weights = []  # Confidence scores
        
        for candidate_hash, feedback in aggregated_feedback.items():
            X.append(feedback.features.to_feature_vector())
            y.append(feedback.mean_success_rate)
            sample_weights.append(feedback.confidence)
        
        X = np.array(X)
        y = np.array(y)
        sample_weights = np.array(sample_weights)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Ridge regression
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        
        # Extract weights
        weights = self.model.coef_
        intercept = self.model.intercept_
        
        # Create PolicyWeights object
        policy_weights = PolicyWeights(
            version="1.0.0",
            weights=weights,
            intercept=intercept,
            feature_names=CandidateFeatures.feature_names(),
            scaler_mean=self.scaler.mean_,
            scaler_std=self.scaler.scale_,
            learned_from_runs=run_ids,
            total_samples=len(X),
            training_timestamp=datetime.utcnow().isoformat(),
        )
        
        return policy_weights
    
    def save_weights(self, weights: PolicyWeights, path: Path):
        """Save weights to JSON file."""
        with open(path, 'w') as f:
            json.dump(weights.to_dict(), f, indent=2, sort_keys=True)
    
    def load_weights(self, path: Path) -> PolicyWeights:
        """Load weights from JSON file."""
        with open(path, 'r') as f:
            return PolicyWeights.from_dict(json.load(f))
```

---

## 5. Weight Provenance Schema

### 5.1. Purpose

Track complete provenance of learned weights for auditability.

### 5.2. Schema

```python
@dataclass
class WeightProvenance:
    """
    Complete provenance record for learned weights.
    """
    version: str
    weights_hash: str  # SHA-256 of weights JSON
    
    # Training data provenance
    training_runs: List[str]
    total_candidates: int
    total_executions: int
    
    # Training parameters
    alpha: float  # Ridge regression alpha
    feature_count: int
    
    # Performance metrics
    train_r2_score: float
    train_mse: float
    
    # Timestamps
    training_started: str
    training_completed: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "weights_hash": self.weights_hash,
            "training_runs": self.training_runs,
            "total_candidates": self.total_candidates,
            "total_executions": self.total_executions,
            "alpha": self.alpha,
            "feature_count": self.feature_count,
            "train_r2_score": self.train_r2_score,
            "train_mse": self.train_mse,
            "training_started": self.training_started,
            "training_completed": self.training_completed,
        }
```

---

## 6. Canonical Serialization Contract

### 6.1. Purpose

Ensure all RFL data structures are deterministically serializable for hashing and verification.

### 6.2. Contract

All RFL data structures MUST implement:

```python
def to_canonical_dict(self) -> Dict[str, Any]:
    """
    Returns canonical dictionary representation.
    
    Requirements:
    - All keys are strings
    - All values are JSON-serializable
    - Dictionaries are sorted by key
    - Lists are in deterministic order
    - No timestamps or non-deterministic fields
    
    Returns:
        Canonical dictionary
    """
    pass

def canonical_hash(self) -> str:
    """
    Computes SHA-256 hash of canonical representation.
    
    Returns:
        Hex-encoded SHA-256 hash
    """
    canonical_str = json.dumps(
        self.to_canonical_dict(),
        sort_keys=True,
        separators=(",", ":")
    )
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
```

### 6.3. Serialization Example

```python
# Serialize CandidateFeatures
features = CandidateFeatures(...)
canonical_dict = features.to_canonical_dict()
canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
hash_value = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

# Verify determinism
assert canonical_hash(features) == hash_value
```

---

## 7. Implementation Checklist

### Phase 1: Feature Extraction
- [ ] Implement `CandidateFeatures` dataclass
- [ ] Implement `FeatureExtractor` class
- [ ] Implement structural feature extraction methods
- [ ] Test feature extraction on sample candidates

### Phase 2: Feedback Derivation
- [ ] Implement `CandidateFeedback` dataclass
- [ ] Implement `FeedbackDeriver` class
- [ ] Implement per-candidate aggregation logic
- [ ] Test feedback derivation on sample traces

### Phase 3: Multi-Run Aggregation
- [ ] Implement `AggregatedFeedback` dataclass
- [ ] Implement `FeedbackAggregator` class
- [ ] Implement confidence score computation
- [ ] Test aggregation on multiple runs

### Phase 4: Policy Learning
- [ ] Implement `PolicyWeights` dataclass
- [ ] Implement `PolicyLearner` class
- [ ] Implement Ridge regression training
- [ ] Test weight learning on sample data

### Phase 5: Weight Provenance
- [ ] Implement `WeightProvenance` dataclass
- [ ] Implement provenance tracking
- [ ] Test provenance serialization

### Phase 6: Integration
- [ ] Integrate with U2 runner
- [ ] Run end-to-end RFL experiment (5 iterations)
- [ ] Validate uplift on held-out test slices

---

**Status**: Build plan complete. Ready for implementation.
