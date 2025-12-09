#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Universe Compiler
------------------

Compiles a NoiseSpec into a fully realized SyntheticUniverse that can
generate deterministic synthetic logs.

The compiler:
    1. Validates the spec for consistency
    2. Resolves all dimensions (classes, items, cycles)
    3. Pre-computes rare event triggers
    4. Creates a ready-to-generate universe object

Must NOT generate uplift interpretations.
All outputs labeled "PHASE II — SYNTHETIC".
Entire system is deterministic.

==============================================================================
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.synthetic_uplift.noise_models import (
    SAFETY_LABEL,
    CorrelationEngine,
    DriftModulator,
    NoiseConfig,
    NoiseEngine,
)
from experiments.synthetic_uplift.noise_schema import (
    ItemSpec,
    NoiseSpec,
    ProbabilityMatrix,
    RareEventChannel,
    RareEventEngine,
    SpecValidationError,
    VarianceConfig,
    validate_spec,
)


# ==============================================================================
# COMPILATION ERRORS
# ==============================================================================

class CompilationError(Exception):
    """Raised when universe compilation fails."""
    pass


# ==============================================================================
# SYNTHETIC UNIVERSE
# ==============================================================================

@dataclass
class UniverseStatistics:
    """Runtime statistics for a generated universe."""
    total_cycles: int = 0
    success_count: int = 0
    rare_event_triggers: Dict[str, int] = field(default_factory=dict)
    per_class_success: Dict[str, int] = field(default_factory=dict)
    per_class_total: Dict[str, int] = field(default_factory=dict)
    probability_range: Tuple[float, float] = (1.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_cycles": self.total_cycles,
            "success_count": self.success_count,
            "success_rate": self.success_count / self.total_cycles if self.total_cycles > 0 else 0.0,
            "rare_event_triggers": self.rare_event_triggers,
            "per_class_success": self.per_class_success,
            "per_class_total": self.per_class_total,
            "probability_range": list(self.probability_range),
        }


@dataclass
class SyntheticUniverse:
    """
    A compiled synthetic universe ready for log generation.
    
    Contains all pre-computed data structures needed for deterministic
    generation of synthetic uplift logs.
    """
    # Source spec
    spec: NoiseSpec
    spec_hash: str
    
    # Resolved items
    items: List[ItemSpec]
    item_ids: List[str]
    item_class_map: Dict[str, str]
    
    # Engines
    drift_modulator: DriftModulator
    correlation_engine: CorrelationEngine
    rare_event_engine: RareEventEngine
    variance_config: VarianceConfig
    
    # Pre-computed
    seed_schedule: List[int]
    
    # Compilation metadata
    compiled_at: str
    
    def __post_init__(self):
        """Initialize tracking state."""
        self._policy_scores: Dict[str, float] = {}
    
    def get_probability(self, mode: str, item_class: str) -> float:
        """Get base probability for a mode and class."""
        return self.spec.probabilities.get_probability(mode, item_class)
    
    def generate_outcome(
        self,
        item_id: str,
        mode: str,
        cycle: int,
        cycle_seed: int,
    ) -> Dict[str, Any]:
        """
        Generate a complete outcome for an item at a cycle.
        
        Applies all noise models in order:
            1. Base probability lookup
            2. Drift modulation
            3. Variance noise
            4. Rare event effects
            5. Correlation
        
        Returns:
            Dict with all outcome details
        """
        item_class = self.item_class_map.get(item_id, "class_a")
        
        # 1. Base probability
        base_prob = self.get_probability(mode, item_class)
        
        # 2. Drift modulation
        drifted_prob = self.drift_modulator.modulate(
            base_prob, cycle, self.spec.num_cycles
        )
        
        # 3. Variance noise
        var_prob = self._apply_variance(drifted_prob, cycle, item_id, cycle_seed)
        
        # 4. Rare event effects
        event_prob, active_events = self.rare_event_engine.apply_effects(
            var_prob, cycle, item_class
        )
        
        # 5. Generate independent outcome
        combined_seed = cycle_seed ^ hash(item_id)
        local_rng = random.Random(combined_seed)
        roll = local_rng.random()
        independent_success = roll < event_prob
        
        # 6. Apply correlation
        final_success = self.correlation_engine.apply_correlation(
            independent_success,
            item_class,
            cycle,
            cycle_seed,
            item_id,
        )
        
        return {
            "success": final_success,
            "outcome": "VERIFIED" if final_success else "ABSTAIN",
            "mock_result": {
                "synthetic": True,
                "base_probability": base_prob,
                "drifted_probability": drifted_prob,
                "variance_probability": var_prob,
                "event_probability": event_prob,
                "roll": roll,
                "class": item_class,
                "independent_outcome": independent_success,
                "correlated_outcome": final_success,
                "active_rare_events": active_events,
            }
        }
    
    def _apply_variance(
        self,
        prob: float,
        cycle: int,
        item_id: str,
        cycle_seed: int,
    ) -> float:
        """Apply variance noise to a probability."""
        if self.variance_config.per_cycle_sigma <= 0 and self.variance_config.per_item_sigma <= 0:
            return prob
        
        # Per-cycle noise
        cycle_noise = 0.0
        if self.variance_config.per_cycle_sigma > 0:
            cycle_rng = random.Random(cycle_seed ^ 0xABCDEF01)
            cycle_noise = cycle_rng.gauss(0, self.variance_config.per_cycle_sigma)
        
        # Per-item noise
        item_noise = 0.0
        if self.variance_config.per_item_sigma > 0:
            item_rng = random.Random(hash(item_id) ^ cycle_seed ^ 0xFEDCBA98)
            item_noise = item_rng.gauss(0, self.variance_config.per_item_sigma)
        
        # Heteroscedastic scaling
        if self.variance_config.heteroscedastic:
            scale = math.sqrt(prob * (1 - prob))
            cycle_noise *= scale
            item_noise *= scale
        
        # Apply and clamp
        modified = prob + cycle_noise + item_noise
        return max(
            self.variance_config.min_prob,
            min(self.variance_config.max_prob, modified)
        )
    
    def select_item(
        self,
        mode: str,
        cycle_seed: int,
    ) -> str:
        """Select an item for a cycle based on mode."""
        rng = random.Random(cycle_seed)
        
        if mode == "baseline":
            return rng.choice(self.item_ids)
        else:
            # RFL mode: use policy scores
            if self._policy_scores:
                scored = sorted(
                    [(item, self._policy_scores.get(item, 0.5)) for item in self.item_ids],
                    key=lambda x: x[1],
                    reverse=True,
                )
                return scored[0][0]
            else:
                return rng.choice(self.item_ids)
    
    def update_policy(self, item_id: str, success: bool):
        """Update RFL policy scores."""
        if item_id not in self._policy_scores:
            self._policy_scores[item_id] = 0.5
        
        if success:
            self._policy_scores[item_id] = min(self._policy_scores[item_id] * 1.1, 0.99)
        else:
            self._policy_scores[item_id] = max(self._policy_scores[item_id] * 0.9, 0.01)
    
    def reset_policy(self):
        """Reset policy scores for a new run."""
        self._policy_scores.clear()
        self.correlation_engine.clear_cache()
    
    def generate_logs(
        self,
        mode: str,
        out_dir: Path,
        verbose: bool = True,
    ) -> Tuple[Path, Path, UniverseStatistics]:
        """
        Generate complete JSONL logs for this universe.
        
        Returns:
            (results_path, manifest_path, statistics)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        self.reset_policy()
        
        results_path = out_dir / f"{self.spec.name}_{mode}.jsonl"
        manifest_path = out_dir / f"{self.spec.name}_{mode}_manifest.json"
        
        stats = UniverseStatistics()
        telemetry_series: List[Dict[str, Any]] = []
        
        min_prob = 1.0
        max_prob = 0.0
        
        if verbose:
            print(f"{'='*60}")
            print(f"{SAFETY_LABEL}")
            print(f"{'='*60}")
            print(f"Generating: {self.spec.name} / {mode}")
            print(f"  Cycles: {self.spec.num_cycles}")
            print(f"  Items:  {len(self.item_ids)}")
            print()
        
        with open(results_path, "w", encoding="utf-8") as f:
            for cycle in range(self.spec.num_cycles):
                cycle_seed = self.seed_schedule[cycle]
                
                # Select item
                chosen_item = self.select_item(mode, cycle_seed)
                item_class = self.item_class_map[chosen_item]
                
                # Generate outcome
                outcome = self.generate_outcome(chosen_item, mode, cycle, cycle_seed)
                success = outcome["success"]
                
                # Track statistics
                stats.total_cycles += 1
                if success:
                    stats.success_count += 1
                
                stats.per_class_total[item_class] = stats.per_class_total.get(item_class, 0) + 1
                if success:
                    stats.per_class_success[item_class] = stats.per_class_success.get(item_class, 0) + 1
                
                # Track rare events
                for event_name in outcome["mock_result"].get("active_rare_events", []):
                    stats.rare_event_triggers[event_name] = stats.rare_event_triggers.get(event_name, 0) + 1
                
                # Track probability range
                event_prob = outcome["mock_result"]["event_probability"]
                min_prob = min(min_prob, event_prob)
                max_prob = max(max_prob, event_prob)
                
                # Update policy for RFL
                if mode == "rfl":
                    self.update_policy(chosen_item, success)
                
                # Build record
                record = {
                    "cycle": cycle,
                    "universe": self.spec.name,
                    "slice": self.spec.name,
                    "mode": mode,
                    "seed": cycle_seed,
                    "item": chosen_item,
                    "result": json.dumps(outcome["mock_result"]),
                    "success": success,
                    "outcome": outcome["outcome"],
                    "proof_found": success,
                    "abstention": not success,
                    "effective_probability": event_prob,
                    "active_rare_events": outcome["mock_result"].get("active_rare_events", []),
                    "label": SAFETY_LABEL,
                    "synthetic": True,
                    "universe_version": self.spec.version,
                }
                
                telemetry_series.append(record)
                f.write(json.dumps(record) + "\n")
                
                if verbose and ((cycle + 1) % 100 == 0 or cycle == self.spec.num_cycles - 1):
                    rate = stats.success_count / stats.total_cycles * 100
                    print(f"  Cycle {cycle + 1}/{self.spec.num_cycles}: success={rate:.1f}%")
        
        stats.probability_range = (min_prob, max_prob)
        
        # Generate manifest
        manifest = {
            "label": SAFETY_LABEL,
            "synthetic": True,
            "universe_name": self.spec.name,
            "universe_description": self.spec.description,
            "universe_version": self.spec.version,
            "spec_hash": self.spec_hash,
            "mode": mode,
            "cycles": self.spec.num_cycles,
            "master_seed": self.spec.seed,
            "probability_matrix": self.spec.probabilities.to_dict(),
            "drift_config": self.spec.drift.to_dict(),
            "correlation_config": self.spec.correlation.to_dict(),
            "variance_config": self.spec.variance.to_dict(),
            "rare_events": [e.to_dict() for e in self.spec.rare_events],
            "telemetry_hash": hashlib.sha256(
                json.dumps(telemetry_series, sort_keys=True).encode()
            ).hexdigest(),
            "statistics": stats.to_dict(),
            "outputs": {
                "results": str(results_path),
                "manifest": str(manifest_path),
            },
            "compiled_at": self.compiled_at,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        
        if verbose:
            print()
            print(f"{'='*60}")
            print("Generation Complete")
            print(f"{'='*60}")
            print(f"  Results:  {results_path}")
            print(f"  Manifest: {manifest_path}")
            print(f"  Success:  {stats.success_count}/{stats.total_cycles} ({stats.to_dict()['success_rate']*100:.1f}%)")
            print(f"  Prob range: [{min_prob:.3f}, {max_prob:.3f}]")
            if stats.rare_event_triggers:
                print(f"  Rare events: {stats.rare_event_triggers}")
            print()
            print(f"⚠️  {SAFETY_LABEL}")
        
        return results_path, manifest_path, stats


# ==============================================================================
# COMPILER
# ==============================================================================

def compile_universe(spec: NoiseSpec) -> SyntheticUniverse:
    """
    Compile a NoiseSpec into a ready-to-generate SyntheticUniverse.
    
    Validates the spec and builds all necessary data structures.
    
    Args:
        spec: The noise specification to compile
    
    Returns:
        A compiled SyntheticUniverse
    
    Raises:
        CompilationError: If validation fails
    """
    # 1. Validate
    errors = validate_spec(spec)
    if errors:
        raise CompilationError(f"Spec validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    # 2. Resolve items
    items = spec.get_items()
    item_ids = [item.id for item in items]
    item_class_map = {item.id: item.item_class for item in items}
    
    # 3. Build noise config for engines
    noise_config = NoiseConfig(
        drift=spec.drift,
        correlation=spec.correlation,
    )
    
    # 4. Create engines
    drift_modulator = DriftModulator(spec.drift)
    correlation_engine = CorrelationEngine(spec.correlation, spec.seed)
    rare_event_engine = RareEventEngine(spec.rare_events, spec.seed)
    
    # 5. Pre-compute seed schedule
    rng = random.Random(spec.seed)
    seed_schedule = [rng.randint(0, 2**32 - 1) for _ in range(spec.num_cycles)]
    
    # 6. Compute spec hash
    spec_hash = spec.compute_hash()
    
    # 7. Build universe
    universe = SyntheticUniverse(
        spec=spec,
        spec_hash=spec_hash,
        items=items,
        item_ids=item_ids,
        item_class_map=item_class_map,
        drift_modulator=drift_modulator,
        correlation_engine=correlation_engine,
        rare_event_engine=rare_event_engine,
        variance_config=spec.variance,
        seed_schedule=seed_schedule,
        compiled_at=datetime.now(timezone.utc).isoformat(),
    )
    
    return universe


def compile_and_generate(
    spec: NoiseSpec,
    mode: str,
    out_dir: Path,
    verbose: bool = True,
) -> Tuple[Path, Path, UniverseStatistics]:
    """
    Convenience function to compile and generate in one step.
    """
    universe = compile_universe(spec)
    return universe.generate_logs(mode, out_dir, verbose)


# ==============================================================================
# COMPARISON UTILITIES
# ==============================================================================

def compare_universes(
    universe1: SyntheticUniverse,
    universe2: SyntheticUniverse,
) -> Dict[str, Any]:
    """
    Compare two universes and return a diff summary.
    
    Does NOT interpret uplift - only structural comparison.
    """
    s1, s2 = universe1.spec, universe2.spec
    
    diff = {
        "label": SAFETY_LABEL,
        "comparison_type": "structural_only",
        "universe1": s1.name,
        "universe2": s2.name,
        "differences": {},
    }
    
    # Compare probabilities
    prob_diff = {}
    for mode in ["baseline", "rfl"]:
        p1 = getattr(s1.probabilities, mode)
        p2 = getattr(s2.probabilities, mode)
        all_classes = set(p1.keys()) | set(p2.keys())
        for cls in all_classes:
            v1 = p1.get(cls, 0.5)
            v2 = p2.get(cls, 0.5)
            if v1 != v2:
                prob_diff[f"{mode}/{cls}"] = {"u1": v1, "u2": v2}
    if prob_diff:
        diff["differences"]["probabilities"] = prob_diff
    
    # Compare drift
    if s1.drift.to_dict() != s2.drift.to_dict():
        diff["differences"]["drift"] = {
            "u1": s1.drift.to_dict(),
            "u2": s2.drift.to_dict(),
        }
    
    # Compare correlation
    if s1.correlation.to_dict() != s2.correlation.to_dict():
        diff["differences"]["correlation"] = {
            "u1": s1.correlation.to_dict(),
            "u2": s2.correlation.to_dict(),
        }
    
    # Compare rare events
    if len(s1.rare_events) != len(s2.rare_events):
        diff["differences"]["rare_event_count"] = {
            "u1": len(s1.rare_events),
            "u2": len(s2.rare_events),
        }
    
    # Compare dimensions
    if s1.num_cycles != s2.num_cycles:
        diff["differences"]["num_cycles"] = {"u1": s1.num_cycles, "u2": s2.num_cycles}
    
    if len(s1.classes) != len(s2.classes) or set(s1.classes) != set(s2.classes):
        diff["differences"]["classes"] = {"u1": s1.classes, "u2": s2.classes}
    
    return diff

