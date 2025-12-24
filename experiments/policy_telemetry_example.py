"""
Example integration of RFL Phase II policy telemetry into experiment runs.

This demonstrates how to:
1. Load telemetry config from config/rfl_policy_phase2.yaml
2. Collect periodic policy snapshots with optional feature telemetry
3. Write telemetry to JSONL for post-experiment analysis

Usage:
    from experiments.policy_telemetry_example import PolicyTelemetryCollector
    
    collector = PolicyTelemetryCollector.from_config("config/rfl_policy_phase2.yaml")
    
    # During experiment loop:
    for step in range(num_steps):
        # ... policy update ...
        if collector.should_snapshot(step):
            collector.capture_snapshot(policy_state, step)
    
    # At end:
    collector.save_to_file("artifacts/policy_snapshots.jsonl")
"""

import json
import yaml
from pathlib import Path
from typing import List, Optional

from rfl.policy import PolicyState, summarize_policy_state, PolicyStateSnapshot


class PolicyTelemetryCollector:
    """
    Collects policy telemetry snapshots during experiment runs.
    
    Attributes:
        enabled: Whether feature telemetry is enabled
        top_k: Number of top features to track
        snapshot_interval: How often to capture snapshots
        snapshots: List of captured snapshots
    """
    
    def __init__(
        self,
        enabled: bool = False,
        top_k: int = 5,
        snapshot_interval: int = 10
    ):
        """
        Initialize telemetry collector.
        
        Args:
            enabled: Enable per-feature telemetry
            top_k: Number of top features to track
            snapshot_interval: Capture snapshot every N steps
        """
        self.enabled = enabled
        self.top_k = top_k
        self.snapshot_interval = snapshot_interval
        self.snapshots: List[PolicyStateSnapshot] = []
    
    @classmethod
    def from_config(cls, config_path: str) -> "PolicyTelemetryCollector":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config/rfl_policy_phase2.yaml
            
        Returns:
            PolicyTelemetryCollector instance
        """
        config_file = Path(config_path)
        if not config_file.exists():
            # Use defaults if config not found
            return cls()
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        feature_telemetry = config.get("feature_telemetry", {})
        
        return cls(
            enabled=feature_telemetry.get("enabled", False),
            top_k=feature_telemetry.get("top_k", 5),
            snapshot_interval=config.get("snapshot_interval", 10)
        )
    
    def should_snapshot(self, step: int) -> bool:
        """
        Check if we should capture a snapshot at this step.
        
        Args:
            step: Current update step
            
        Returns:
            True if snapshot should be captured
        """
        return step > 0 and step % self.snapshot_interval == 0
    
    def capture_snapshot(
        self,
        policy_state: PolicyState,
        step: Optional[int] = None
    ) -> PolicyStateSnapshot:
        """
        Capture a policy snapshot.
        
        Args:
            policy_state: Current policy state
            step: Override step number (default: use policy_state.step)
            
        Returns:
            Captured snapshot
        """
        snapshot = summarize_policy_state(
            policy_state,
            include_feature_telemetry=self.enabled,
            top_k=self.top_k
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save all snapshots to JSONL file.
        
        Args:
            filepath: Path to output file
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for snapshot in self.snapshots:
                f.write(json.dumps(snapshot.to_dict()) + '\n')
    
    def clear(self) -> None:
        """Clear all collected snapshots."""
        self.snapshots.clear()


def example_usage():
    """Example of using PolicyTelemetryCollector in an experiment."""
    from rfl.policy import init_cold_start, PolicyUpdater, extract_features, compute_reward
    
    # Load config
    collector = PolicyTelemetryCollector.from_config("config/rfl_policy_phase2.yaml")
    
    # Initialize policy
    feature_names = ["formula_length", "num_atoms", "num_connectives", "bias"]
    policy_state = init_cold_start(feature_names, seed=42)
    updater = PolicyUpdater(learning_rate=0.01, seed=42)
    
    print(f"Feature telemetry enabled: {collector.enabled}")
    print(f"Snapshot interval: {collector.snapshot_interval}")
    
    # Simulate experiment loop
    formulas = ["p -> q", "p & q", "(p | q) -> r", "~p -> q"]
    
    for i, formula in enumerate(formulas):
        step = i + 1
        
        # Extract features
        features_vec = extract_features(formula)
        features = features_vec.features
        
        # Simulate outcome
        success = i % 2 == 0  # Every other formula succeeds
        reward_signal = compute_reward(success=success, abstained=False)
        
        # Update policy
        policy_state = updater.update(policy_state, features, reward_signal.reward)
        
        # Capture snapshot if needed
        if collector.should_snapshot(step):
            snapshot = collector.capture_snapshot(policy_state, step)
            print(f"Step {step}: Captured snapshot (L2 norm: {snapshot.l2_norm:.4f})")
            
            if snapshot.feature_telemetry:
                print(f"  Top positive: {snapshot.feature_telemetry.top_k_positive[:2]}")
                print(f"  Sparsity: {snapshot.feature_telemetry.sparsity:.2f}")
    
    # Save to file
    collector.save_to_file("artifacts/example_policy_snapshots.jsonl")
    print(f"\nSaved {len(collector.snapshots)} snapshots to artifacts/example_policy_snapshots.jsonl")


if __name__ == "__main__":
    example_usage()
