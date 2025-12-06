# PHASE II — NOT USED IN PHASE I
#
# U2 Pipeline Module
# ===================
# Provides modular components for running Phase II U2 uplift experiments:
# - curriculum_loader_v2: Loads Phase II curriculum configuration
# - feature_extraction + scoring: Feature vectors and policy scoring for candidates
# - slice_success_metrics: Integration with the existing metrics module
# - manifest_generator: Generates unified manifests for paired runs
# - attestation_bindings: Dual-root attestation integration
#
# Absolute Safeguards:
# - Do NOT reinterpret Phase I logs as uplift evidence.
# - All Phase II artifacts must be clearly labeled "PHASE II — NOT USED IN PHASE I".
# - All code must remain deterministic.
# - RFL uses verifiable feedback only (no RLHF, no preferences, no proxy rewards).
# - Zero interpretation of uplift — only raw data and placeholders.

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

from experiments.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)


# ============================================================================
# Constants
# ============================================================================

PHASE_II_LABEL = "PHASE II — NOT USED IN PHASE I"
DEFAULT_CONFIG_PATH = Path("config/curriculum_uplift_phase2.yaml")


# ============================================================================
# curriculum_loader_v2
# ============================================================================

@dataclass(frozen=True)
class SliceConfig:
    """
    Configuration for a single experiment slice.
    
    This is a Phase II-specific slice configuration, distinct from the
    curriculum.gates.CurriculumSlice used in Phase I.
    """
    name: str
    description: str
    items: List[str]
    prereg_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "items": list(self.items),
            "prereg_hash": self.prereg_hash,
            "metadata": dict(self.metadata),
        }


@dataclass
class CurriculumConfigV2:
    """
    Phase II curriculum configuration.
    
    Loads slice definitions from curriculum_uplift_phase2.yaml.
    """
    version: str
    slices: Dict[str, SliceConfig]
    config_hash: str
    
    @classmethod
    def load(cls, config_path: Path = DEFAULT_CONFIG_PATH) -> "CurriculumConfigV2":
        """Load curriculum configuration from a YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Curriculum config not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        
        # Compute config hash for traceability
        config_str = json.dumps(raw_config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()
        
        version = str(raw_config.get("version", "2.0"))
        slices_raw = raw_config.get("slices", {})
        
        slices: Dict[str, SliceConfig] = {}
        for slice_name, slice_data in slices_raw.items():
            slices[slice_name] = SliceConfig(
                name=slice_name,
                description=slice_data.get("description", ""),
                items=list(slice_data.get("items", [])),
                prereg_hash=slice_data.get("prereg_hash", ""),
                metadata={k: v for k, v in slice_data.items() 
                          if k not in ("description", "items", "prereg_hash")},
            )
        
        return cls(version=version, slices=slices, config_hash=config_hash)
    
    def get_slice(self, slice_name: str) -> SliceConfig:
        """Get a specific slice configuration by name."""
        if slice_name not in self.slices:
            raise KeyError(f"Slice '{slice_name}' not found in curriculum config")
        return self.slices[slice_name]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "slices": {name: s.to_dict() for name, s in self.slices.items()},
            "config_hash": self.config_hash,
        }


def load_curriculum_v2(config_path: Path = DEFAULT_CONFIG_PATH) -> CurriculumConfigV2:
    """
    Load Phase II curriculum configuration.
    
    This is the main entry point for curriculum loading in U2 experiments.
    """
    return CurriculumConfigV2.load(config_path)


# ============================================================================
# Feature Extraction + Scoring
# ============================================================================

@dataclass
class CandidateFeatures:
    """
    Feature vector for a candidate item.
    
    Features are used by the policy to score and order candidates.
    """
    item: str
    item_hash: str
    length: int
    complexity_estimate: float
    success_history_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item,
            "item_hash": self.item_hash,
            "length": self.length,
            "complexity_estimate": self.complexity_estimate,
            "success_history_score": self.success_history_score,
        }


def compute_item_hash(item: str) -> str:
    """Compute a deterministic hash for an item."""
    return hashlib.sha256(item.encode("utf-8")).hexdigest()[:16]


def extract_features(
    item: str,
    success_history: Optional[Dict[str, float]] = None,
) -> CandidateFeatures:
    """
    Extract features from a candidate item.
    
    Args:
        item: The candidate item string
        success_history: Optional dictionary mapping item hashes to success scores
        
    Returns:
        CandidateFeatures with extracted feature values
    """
    item_hash = compute_item_hash(item)
    length = len(item)
    
    # Complexity estimate: simple heuristic based on operators and nesting
    complexity = 0.0
    for char in item:
        if char in "+-*/^":
            complexity += 1.0
        elif char in "()[]{}":
            complexity += 0.5
    complexity = complexity / max(1, length)  # Normalize by length
    
    # Success history score (from previous cycles)
    success_score = 0.0
    if success_history and item_hash in success_history:
        success_score = success_history[item_hash]
    
    return CandidateFeatures(
        item=item,
        item_hash=item_hash,
        length=length,
        complexity_estimate=complexity,
        success_history_score=success_score,
    )


@dataclass
class PolicyWeights:
    """
    Policy weights for candidate scoring.
    
    These weights determine how features are combined to produce a score.
    """
    length_weight: float = 0.0
    complexity_weight: float = 0.0
    success_history_weight: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "length_weight": self.length_weight,
            "complexity_weight": self.complexity_weight,
            "success_history_weight": self.success_history_weight,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PolicyWeights":
        return cls(
            length_weight=data.get("length_weight", 0.0),
            complexity_weight=data.get("complexity_weight", 0.0),
            success_history_weight=data.get("success_history_weight", 0.0),
        )


def score_candidate(features: CandidateFeatures, weights: PolicyWeights) -> float:
    """
    Compute a score for a candidate based on its features and policy weights.
    
    Higher scores indicate higher priority in the ordering.
    """
    return (
        weights.length_weight * features.length +
        weights.complexity_weight * features.complexity_estimate +
        weights.success_history_weight * features.success_history_score
    )


def extract_and_score_candidates(
    items: List[str],
    weights: PolicyWeights,
    success_history: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, CandidateFeatures, float]]:
    """
    Extract features and compute scores for all candidates.
    
    Args:
        items: List of candidate items
        weights: Policy weights for scoring
        success_history: Optional success history dictionary
        
    Returns:
        List of (item, features, score) tuples, sorted by score descending
    """
    results = []
    for item in items:
        features = extract_features(item, success_history)
        score = score_candidate(features, weights)
        results.append((item, features, score))
    
    # Sort by score descending (higher is better)
    results.sort(key=lambda x: x[2], reverse=True)
    return results


# ============================================================================
# Slice Success Metrics Integration
# ============================================================================

@dataclass
class SuccessMetricConfig:
    """Configuration for a success metric evaluation."""
    metric_type: str  # "goal_hit", "sparse", "chain", "multi_goal"
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type,
            "parameters": dict(self.parameters),
        }


@dataclass
class SuccessMetricResult:
    """Result of a success metric evaluation."""
    success: bool
    metric_value: float
    config: SuccessMetricConfig
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "metric_value": self.metric_value,
            "config": self.config.to_dict(),
            "details": self.details,
        }


def evaluate_success_metric(
    config: SuccessMetricConfig,
    verified_statements: Optional[List[Dict[str, Any]]] = None,
    verified_count: int = 0,
    attempted_count: int = 0,
    verified_hashes: Optional[Set[str]] = None,
    dependency_graph: Optional[Dict[str, List[str]]] = None,
) -> SuccessMetricResult:
    """
    Evaluate a success metric based on the configuration.
    
    This function dispatches to the appropriate metric implementation
    from slice_success_metrics.py.
    
    Args:
        config: Success metric configuration
        verified_statements: List of verified statement dicts with 'hash' keys
        verified_count: Number of verified statements (for sparse metric)
        attempted_count: Number of attempted statements (for sparse metric)
        verified_hashes: Set of verified hashes (for multi_goal metric)
        dependency_graph: Dependency graph (for chain metric)
        
    Returns:
        SuccessMetricResult with evaluation outcome
    """
    metric_type = config.metric_type
    params = config.parameters
    details: Dict[str, Any] = {"input_params": dict(params)}
    
    if metric_type == "goal_hit":
        target_hashes = set(params.get("target_hashes", []))
        min_total_verified = params.get("min_total_verified", 1)
        statements = verified_statements or []
        
        success, value = compute_goal_hit(statements, target_hashes, min_total_verified)
        details["target_hashes_count"] = len(target_hashes)
        details["verified_statements_count"] = len(statements)
        
    elif metric_type == "sparse":
        min_verified = params.get("min_verified", 1)
        success, value = compute_sparse_success(verified_count, attempted_count, min_verified)
        details["verified_count"] = verified_count
        details["attempted_count"] = attempted_count
        
    elif metric_type == "chain":
        chain_target_hash = params.get("chain_target_hash", "")
        min_chain_length = params.get("min_chain_length", 1)
        statements = verified_statements or []
        graph = dependency_graph or {}
        
        success, value = compute_chain_success(
            statements, graph, chain_target_hash, min_chain_length
        )
        details["chain_target_hash"] = chain_target_hash
        details["graph_nodes_count"] = len(graph)
        
    elif metric_type == "multi_goal":
        required_goal_hashes = set(params.get("required_goal_hashes", []))
        verified = verified_hashes or set()
        
        success, value = compute_multi_goal_success(verified, required_goal_hashes)
        details["required_goal_hashes_count"] = len(required_goal_hashes)
        details["verified_hashes_count"] = len(verified)
        
    else:
        # Unknown metric type - return failure
        success = False
        value = 0.0
        details["error"] = f"Unknown metric type: {metric_type}"
    
    return SuccessMetricResult(
        success=success,
        metric_value=value,
        config=config,
        details=details,
    )


# ============================================================================
# Attestation Bindings
# ============================================================================

def compute_attestation_hash(data: Any) -> str:
    """
    Compute a deterministic attestation hash for any data.
    
    Uses canonical JSON serialization for reproducibility.
    """
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CycleAttestation:
    """
    Attestation record for a single experiment cycle.
    
    This provides a cryptographic binding of the cycle inputs and outputs.
    """
    cycle_index: int
    slice_name: str
    mode: str
    seed: int
    item: str
    item_hash: str
    result: str
    success: bool
    attestation_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "seed": self.seed,
            "item": self.item,
            "item_hash": self.item_hash,
            "result": self.result,
            "success": self.success,
            "attestation_hash": self.attestation_hash,
        }


def create_cycle_attestation(
    cycle_index: int,
    slice_name: str,
    mode: str,
    seed: int,
    item: str,
    result: str,
    success: bool,
) -> CycleAttestation:
    """
    Create an attestation record for a cycle.
    
    The attestation hash binds all inputs and outputs together.
    """
    item_hash = compute_item_hash(item)
    
    attestation_data = {
        "cycle_index": cycle_index,
        "slice_name": slice_name,
        "mode": mode,
        "seed": seed,
        "item": item,
        "item_hash": item_hash,
        "result": result,
        "success": success,
    }
    attestation_hash = compute_attestation_hash(attestation_data)
    
    return CycleAttestation(
        cycle_index=cycle_index,
        slice_name=slice_name,
        mode=mode,
        seed=seed,
        item=item,
        item_hash=item_hash,
        result=result,
        success=success,
        attestation_hash=attestation_hash,
    )


# ============================================================================
# Manifest Generator
# ============================================================================

@dataclass
class DebugArtifact:
    """Debug artifact for a single cycle."""
    cycle_index: int
    candidate_ordering_trace: List[Dict[str, Any]]
    feature_vectors: List[Dict[str, Any]]
    policy_weights: Dict[str, float]
    success_metric_evaluation: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "candidate_ordering_trace": self.candidate_ordering_trace,
            "feature_vectors": self.feature_vectors,
            "policy_weights": self.policy_weights,
            "success_metric_evaluation": self.success_metric_evaluation,
        }


@dataclass
class PairedRunManifest:
    """
    Unified manifest for a paired baseline/RFL experiment run.
    
    Contains all metadata needed to reproduce and analyze the experiment.
    """
    label: str
    experiment_id: str
    slice_name: str
    cycles: int
    initial_seed: int
    slice_config_hash: str
    prereg_hash: str
    baseline_log: str
    rfl_log: str
    baseline_ht_hash: str
    rfl_ht_hash: str
    delta_p_placeholder: Optional[float]  # Placeholder - NO INTERPRETATION
    created_at: str
    deterministic_seed_schedule: List[int]
    debug_artifacts_baseline: Optional[str] = None
    debug_artifacts_rfl: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "cycles": self.cycles,
            "initial_seed": self.initial_seed,
            "slice_config_hash": self.slice_config_hash,
            "prereg_hash": self.prereg_hash,
            "baseline_log": self.baseline_log,
            "rfl_log": self.rfl_log,
            "baseline_ht_hash": self.baseline_ht_hash,
            "rfl_ht_hash": self.rfl_ht_hash,
            "delta_p_placeholder": self.delta_p_placeholder,
            "created_at": self.created_at,
            "deterministic_seed_schedule": self.deterministic_seed_schedule,
            "debug_artifacts_baseline": self.debug_artifacts_baseline,
            "debug_artifacts_rfl": self.debug_artifacts_rfl,
        }


def compute_slice_config_hash(slice_config: SliceConfig) -> str:
    """Compute a hash of the slice configuration."""
    config_dict = slice_config.to_dict()
    return compute_attestation_hash(config_dict)


def compute_ht_series_hash(ht_series: List[Dict[str, Any]]) -> str:
    """Compute a hash of the telemetry series."""
    return compute_attestation_hash(ht_series)


def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """Generate a deterministic list of seeds for each cycle."""
    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]


def create_paired_manifest(
    experiment_id: str,
    slice_config: SliceConfig,
    cycles: int,
    initial_seed: int,
    baseline_log_path: str,
    rfl_log_path: str,
    baseline_ht_series: List[Dict[str, Any]],
    rfl_ht_series: List[Dict[str, Any]],
    debug_artifacts_baseline_path: Optional[str] = None,
    debug_artifacts_rfl_path: Optional[str] = None,
) -> PairedRunManifest:
    """
    Create a unified manifest for a paired baseline/RFL experiment.
    
    Note: delta_p_placeholder is always None. We do NOT compute or interpret
    uplift in the manifest generator. This is intentional to maintain
    zero interpretation of results.
    """
    slice_config_hash = compute_slice_config_hash(slice_config)
    seed_schedule = generate_seed_schedule(initial_seed, cycles)
    baseline_ht_hash = compute_ht_series_hash(baseline_ht_series)
    rfl_ht_hash = compute_ht_series_hash(rfl_ht_series)
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    return PairedRunManifest(
        label=PHASE_II_LABEL,
        experiment_id=experiment_id,
        slice_name=slice_config.name,
        cycles=cycles,
        initial_seed=initial_seed,
        slice_config_hash=slice_config_hash,
        prereg_hash=slice_config.prereg_hash,
        baseline_log=baseline_log_path,
        rfl_log=rfl_log_path,
        baseline_ht_hash=baseline_ht_hash,
        rfl_ht_hash=rfl_ht_hash,
        delta_p_placeholder=None,  # NO INTERPRETATION
        created_at=created_at,
        deterministic_seed_schedule=seed_schedule,
        debug_artifacts_baseline=debug_artifacts_baseline_path,
        debug_artifacts_rfl=debug_artifacts_rfl_path,
    )


def save_manifest(manifest: PairedRunManifest, output_path: Path) -> None:
    """Save a manifest to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=True)


def save_debug_artifacts(artifacts: List[DebugArtifact], output_path: Path) -> None:
    """Save debug artifacts to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for artifact in artifacts:
            f.write(json.dumps(artifact.to_dict()) + "\n")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "PHASE_II_LABEL",
    "DEFAULT_CONFIG_PATH",
    # Curriculum Loader V2
    "SliceConfig",
    "CurriculumConfigV2",
    "load_curriculum_v2",
    # Feature Extraction + Scoring
    "CandidateFeatures",
    "PolicyWeights",
    "compute_item_hash",
    "extract_features",
    "score_candidate",
    "extract_and_score_candidates",
    # Success Metrics
    "SuccessMetricConfig",
    "SuccessMetricResult",
    "evaluate_success_metric",
    # Attestation Bindings
    "CycleAttestation",
    "compute_attestation_hash",
    "create_cycle_attestation",
    # Manifest Generator
    "DebugArtifact",
    "PairedRunManifest",
    "compute_slice_config_hash",
    "compute_ht_series_hash",
    "generate_seed_schedule",
    "create_paired_manifest",
    "save_manifest",
    "save_debug_artifacts",
]
