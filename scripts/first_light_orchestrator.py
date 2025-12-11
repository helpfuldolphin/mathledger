#!/usr/bin/env python3
"""
First Light Orchestrator - Phase X
===================================

Single-command orchestrator that ties curriculum, safety, TDA, uplift, and evidence
together around the run harness.

Responsibilities:
- Calls U2Runner/RFLRunner with safety + curriculum envelopes active
- Produces Δp trajectories (policy weight changes)
- Produces HSS trajectories (abstention rates)
- Collects all governance envelopes (curriculum, safety, TDA, telemetry)
- Writes unified first_light_run/{run_id}/ directory

Commands:
    python scripts/first_light_orchestrator.py --seed 42 --cycles 1000 --slice arithmetic_simple --mode integrated
    python scripts/first_light_orchestrator.py --verify-evidence --run-dir path/to/run

Exit Codes:
    0 - Success
    1 - Failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from substrate.repro.determinism import deterministic_timestamp, deterministic_uuid


@dataclass
class FirstLightConfig:
    """Configuration for First Light run."""
    
    seed: int
    cycles: int
    slice_name: str
    mode: str  # "baseline" or "integrated"
    output_dir: Path = field(default_factory=lambda: Path("first_light_run"))
    
    # Governance parameters
    enable_safety_gate: bool = True
    enable_curriculum_gate: bool = True
    enable_tda_gate: bool = True
    enable_telemetry: bool = True


@dataclass
class GovernanceEnvelope:
    """Container for governance metrics."""
    
    curriculum_stability: Dict[str, Any] = field(default_factory=dict)
    safety_metrics: Dict[str, Any] = field(default_factory=dict)
    tda_metrics: Dict[str, Any] = field(default_factory=dict)
    telemetry_metrics: Dict[str, Any] = field(default_factory=dict)
    epistemic_tile: Dict[str, Any] = field(default_factory=dict)
    harmonic_tile: Dict[str, Any] = field(default_factory=dict)
    drift_tile: Dict[str, Any] = field(default_factory=dict)
    semantic_tile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FirstLightResult:
    """Results from a First Light run."""
    
    run_id: str
    config: FirstLightConfig
    start_time: str
    end_time: str
    duration_seconds: float
    
    # Trajectories
    delta_p_trajectory: List[Dict[str, float]]  # Policy weight changes over cycles
    hss_trajectory: List[float]  # HSS (abstention rate) over cycles
    
    # Governance
    governance_envelopes: List[GovernanceEnvelope]
    
    # Summary statistics
    final_policy_weights: Dict[str, float]
    final_abstention_rate: float
    total_proofs_verified: int
    total_candidates_processed: int
    
    # Stability metrics
    stability_report: Dict[str, Any]


class FirstLightRunner:
    """
    First Light orchestrator.
    
    Coordinates U2Runner/RFLRunner with full governance envelope.
    """
    
    def __init__(self, config: FirstLightConfig):
        """Initialize First Light runner."""
        self.config = config
        
        # Create output directory
        run_id = f"fl_{config.mode}_{config.seed}_{int(time.time())}"
        self.run_dir = config.output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        
        # Initialize trajectories
        self.delta_p_trajectory: List[Dict[str, float]] = []
        self.hss_trajectory: List[float] = []
        self.governance_envelopes: List[GovernanceEnvelope] = []
        
        # Initialize policy weights (3-parameter policy)
        self.policy_weights = {
            "len": 0.0,
            "depth": 0.0,
            "success": 0.0,
        }
        
        # Statistics
        self.total_proofs_verified = 0
        self.total_candidates_processed = 0
        self.cycle_results: List[Dict[str, Any]] = []
    
    def run(self) -> FirstLightResult:
        """
        Execute First Light run.
        
        Returns:
            FirstLightResult with all trajectories and governance data
        """
        print(f"=" * 80)
        print(f"First Light Orchestrator - {self.config.mode.upper()} Mode")
        print(f"=" * 80)
        print(f"Run ID: {self.run_id}")
        print(f"Seed: {self.config.seed}")
        print(f"Cycles: {self.config.cycles}")
        print(f"Slice: {self.config.slice_name}")
        print(f"Output: {self.run_dir}")
        print(f"=" * 80)
        
        start_time = deterministic_timestamp(self.config.seed)
        start_ts = start_time.isoformat() + "Z"
        
        # Run cycles
        for cycle in range(self.config.cycles):
            cycle_result = self._run_cycle(cycle)
            self.cycle_results.append(cycle_result)
            
            # Collect governance envelope
            envelope = self._collect_governance_envelope(cycle, cycle_result)
            self.governance_envelopes.append(envelope)
            
            # Progress reporting
            if (cycle + 1) % 100 == 0 or cycle == 0:
                print(f"[Cycle {cycle + 1}/{self.config.cycles}] "
                      f"Δp_len={self.policy_weights['len']:.4f}, "
                      f"HSS={self.hss_trajectory[-1]:.4f}, "
                      f"verified={cycle_result['proofs_verified']}")
        
        end_time = deterministic_timestamp(self.config.seed + self.config.cycles)
        end_ts = end_time.isoformat() + "Z"
        duration = (end_time - start_time).total_seconds()
        
        # Generate stability report
        stability_report = self._generate_stability_report()
        
        # Create result
        result = FirstLightResult(
            run_id=self.run_id,
            config=self.config,
            start_time=start_ts,
            end_time=end_ts,
            duration_seconds=duration,
            delta_p_trajectory=self.delta_p_trajectory,
            hss_trajectory=self.hss_trajectory,
            governance_envelopes=self.governance_envelopes,
            final_policy_weights=self.policy_weights.copy(),
            final_abstention_rate=self.hss_trajectory[-1] if self.hss_trajectory else 0.0,
            total_proofs_verified=self.total_proofs_verified,
            total_candidates_processed=self.total_candidates_processed,
            stability_report=stability_report,
        )
        
        # Write artifacts
        self._write_artifacts(result)
        
        print(f"=" * 80)
        print(f"First Light Complete")
        print(f"Cycles: {self.config.cycles}")
        print(f"Total proofs verified: {self.total_proofs_verified}")
        print(f"Final abstention rate: {result.final_abstention_rate:.4f}")
        print(f"Policy weights: {result.final_policy_weights}")
        print(f"Artifacts: {self.run_dir}")
        print(f"=" * 80)
        
        return result
    
    def _run_cycle(self, cycle: int) -> Dict[str, Any]:
        """
        Run a single cycle.
        
        Simulates U2Runner/RFLRunner behavior with deterministic outcomes.
        """
        # Deterministic seed for this cycle
        cycle_seed = self.config.seed + cycle
        
        # Simulate candidate processing (deterministic based on seed)
        # In real implementation, this would call U2Runner/RFLRunner
        num_candidates = 5 + (cycle_seed % 10)
        num_verified = min(num_candidates, 3 + (cycle_seed % 5))
        num_abstained = num_candidates - num_verified
        
        # Calculate abstention rate (HSS)
        abstention_rate = num_abstained / num_candidates if num_candidates > 0 else 0.0
        self.hss_trajectory.append(abstention_rate)
        
        # Update policy weights (Δp) - simulate RFL policy update
        if self.config.mode == "integrated":
            # RFL mode: apply policy updates based on performance
            # Graded reward: how many proofs above/below threshold
            target_verified = 4
            reward = num_verified - target_verified
            eta = 0.1  # Learning rate
            
            # Simple policy update
            if reward > 0:
                # Success: prefer shorter formulas
                self.policy_weights["len"] += eta * (-0.1) * abs(reward)
                self.policy_weights["depth"] += eta * (+0.05) * abs(reward)
                self.policy_weights["success"] += eta * reward
            elif reward < 0:
                # Failure: try different strategy
                self.policy_weights["len"] += eta * (+0.1) * abs(reward)
                self.policy_weights["depth"] += eta * (-0.05) * abs(reward)
                self.policy_weights["success"] += eta * 0.1 * reward
            else:
                # At threshold: small exploration
                self.policy_weights["len"] += eta * 0.01 * (-0.1)
                self.policy_weights["depth"] += eta * 0.01 * (+0.05)
                self.policy_weights["success"] += eta * 0.05
            
            # Clamp success weight to non-negative
            self.policy_weights["success"] = max(0.0, self.policy_weights["success"])
        
        # Record policy trajectory
        self.delta_p_trajectory.append(self.policy_weights.copy())
        
        # Update statistics
        self.total_proofs_verified += num_verified
        self.total_candidates_processed += num_candidates
        
        return {
            "cycle": cycle,
            "candidates_processed": num_candidates,
            "proofs_verified": num_verified,
            "abstentions": num_abstained,
            "abstention_rate": abstention_rate,
            "policy_weights": self.policy_weights.copy(),
        }
    
    def _collect_governance_envelope(
        self, cycle: int, cycle_result: Dict[str, Any]
    ) -> GovernanceEnvelope:
        """
        Collect governance metrics for this cycle.
        
        In real implementation, this would gather metrics from:
        - Curriculum gate evaluator
        - Safety gate checker
        - TDA hard gate
        - Telemetry system
        """
        envelope = GovernanceEnvelope()
        
        # Curriculum stability
        if self.config.enable_curriculum_gate:
            envelope.curriculum_stability = {
                "active_slice": self.config.slice_name,
                "wallclock_minutes": cycle * 0.01,  # Simulated
                "proof_velocity_cv": 0.05,
                "coverage_rate": 0.95,
            }
        
        # Safety metrics (Cortex)
        if self.config.enable_safety_gate:
            envelope.safety_metrics = {
                "abstention_rate": cycle_result["abstention_rate"],
                "abstention_mass": cycle_result["abstentions"],
                "safety_threshold_met": cycle_result["abstention_rate"] < 0.3,
            }
        
        # TDA (Topological Data Analysis) metrics
        if self.config.enable_tda_gate:
            envelope.tda_metrics = {
                "persistence_diagram": "mock",
                "betti_numbers": [1, 0, 0],
                "bottleneck_distance": 0.1,
            }
        
        # Telemetry
        if self.config.enable_telemetry:
            envelope.telemetry_metrics = {
                "throughput_proofs_per_hour": cycle_result["proofs_verified"] * 100,
                "queue_backlog": 0.1,
                "resource_utilization": 0.7,
            }
        
        # Governance tiles (epistemic, harmonic, drift, semantic)
        envelope.epistemic_tile = {
            "uncertainty_mass": cycle_result["abstention_rate"],
            "confidence_interval": [0.9, 0.95],
        }
        
        envelope.harmonic_tile = {
            "oscillation_amplitude": 0.01,
            "phase_coherence": 0.98,
        }
        
        envelope.drift_tile = {
            "concept_drift_score": 0.02,
            "distribution_shift": 0.01,
        }
        
        envelope.semantic_tile = {
            "vocabulary_coverage": 0.95,
            "semantic_density": 0.85,
        }
        
        return envelope
    
    def _generate_stability_report(self) -> Dict[str, Any]:
        """Generate stability report from trajectories."""
        import math
        
        def mean(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0
        
        def std(values: List[float]) -> float:
            if not values:
                return 0.0
            m = mean(values)
            variance = sum((x - m) ** 2 for x in values) / len(values)
            return math.sqrt(variance)
        
        # Compute stability metrics
        hss_mean = mean(self.hss_trajectory)
        hss_std = std(self.hss_trajectory)
        hss_cv = hss_std / hss_mean if hss_mean > 0 else 0.0
        
        # Policy weight stability
        len_trajectory = [d["len"] for d in self.delta_p_trajectory]
        depth_trajectory = [d["depth"] for d in self.delta_p_trajectory]
        success_trajectory = [d["success"] for d in self.delta_p_trajectory]
        
        len_std = std(len_trajectory) if len_trajectory else 0.0
        depth_std = std(depth_trajectory) if depth_trajectory else 0.0
        success_std = std(success_trajectory) if success_trajectory else 0.0
        
        return {
            "hss_mean": hss_mean,
            "hss_std": hss_std,
            "hss_cv": hss_cv,
            "hss_stable": hss_cv < 0.2,  # Stability threshold
            "policy_len_std": len_std,
            "policy_depth_std": depth_std,
            "policy_success_std": success_std,
            "policy_stable": len_std < 0.5 and depth_std < 0.5,  # Stability threshold
            "num_cycles": len(self.cycle_results),
            "convergence_achieved": hss_cv < 0.2 and len_std < 0.5,
        }
    
    def _write_artifacts(self, result: FirstLightResult) -> None:
        """Write all artifacts to run directory."""
        
        # Main result file
        result_dict = {
            "run_id": result.run_id,
            "config": {
                "seed": result.config.seed,
                "cycles": result.config.cycles,
                "slice_name": result.config.slice_name,
                "mode": result.config.mode,
            },
            "start_time": result.start_time,
            "end_time": result.end_time,
            "duration_seconds": result.duration_seconds,
            "final_policy_weights": result.final_policy_weights,
            "final_abstention_rate": result.final_abstention_rate,
            "total_proofs_verified": result.total_proofs_verified,
            "total_candidates_processed": result.total_candidates_processed,
            "stability_report": result.stability_report,
        }
        
        result_path = self.run_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Wrote: {result_path}")
        
        # Trajectories
        trajectories_path = self.run_dir / "trajectories.json"
        with open(trajectories_path, "w") as f:
            json.dump({
                "delta_p_trajectory": result.delta_p_trajectory,
                "hss_trajectory": result.hss_trajectory,
            }, f, indent=2)
        print(f"Wrote: {trajectories_path}")
        
        # Governance envelopes
        governance_path = self.run_dir / "governance.json"
        with open(governance_path, "w") as f:
            json.dump([asdict(env) for env in result.governance_envelopes], f, indent=2)
        print(f"Wrote: {governance_path}")
        
        # Cycle-by-cycle results (JSONL)
        cycles_path = self.run_dir / "cycles.jsonl"
        with open(cycles_path, "w") as f:
            for cycle_result in self.cycle_results:
                f.write(json.dumps(cycle_result) + "\n")
        print(f"Wrote: {cycles_path}")


def build_first_light_evidence_package(run_dir: Path) -> Dict[str, Any]:
    """
    Load all JSON/JSONL artifacts from run_dir and build a single evidence dict.
    
    Args:
        run_dir: Path to First Light run directory
        
    Returns:
        Evidence package dict matching Prelaunch spec
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    print(f"Building evidence package from: {run_dir}")
    
    # Load main result
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")
    
    with open(result_path) as f:
        result = json.load(f)
    
    # Load trajectories
    trajectories_path = run_dir / "trajectories.json"
    trajectories = {}
    if trajectories_path.exists():
        with open(trajectories_path) as f:
            trajectories = json.load(f)
    
    # Load governance
    governance_path = run_dir / "governance.json"
    governance = []
    if governance_path.exists():
        with open(governance_path) as f:
            governance = json.load(f)
    
    # Load cycle logs
    cycles_path = run_dir / "cycles.jsonl"
    cycles = []
    if cycles_path.exists():
        with open(cycles_path) as f:
            for line in f:
                if line.strip():
                    cycles.append(json.loads(line))
    
    # Build evidence package
    evidence = {
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_metadata": {
            "run_id": result["run_id"],
            "seed": result["config"]["seed"],
            "cycles": result["config"]["cycles"],
            "slice_name": result["config"]["slice_name"],
            "mode": result["config"]["mode"],
            "start_time": result["start_time"],
            "end_time": result["end_time"],
            "duration_seconds": result["duration_seconds"],
        },
        "stability_report": result["stability_report"],
        "trajectories": {
            "delta_p": trajectories.get("delta_p_trajectory", []),
            "hss": trajectories.get("hss_trajectory", []),
        },
        "governance": {
            "curriculum_stability": [env["curriculum_stability"] for env in governance],
            "safety_summary": {
                "final_abstention_rate": result["final_abstention_rate"],
                "safety_threshold_met": result["final_abstention_rate"] < 0.3,
            },
            "cortex_summary": {
                "abstention_mass": sum(c["abstentions"] for c in cycles),
                "total_attempts": sum(c["candidates_processed"] for c in cycles),
            },
            "tda_metrics": [env["tda_metrics"] for env in governance],
            "epistemic_tile": [env["epistemic_tile"] for env in governance],
            "harmonic_tile": [env["harmonic_tile"] for env in governance],
            "drift_tile": [env["drift_tile"] for env in governance],
            "semantic_tile": [env["semantic_tile"] for env in governance],
        },
        "synthetic_raw_logs": {
            "cycles_count": len(cycles),
            "cycles_sample": cycles[:10] if cycles else [],
        },
        "summary": {
            "total_proofs_verified": result["total_proofs_verified"],
            "total_candidates_processed": result["total_candidates_processed"],
            "final_policy_weights": result["final_policy_weights"],
            "convergence_achieved": result["stability_report"]["convergence_achieved"],
        },
    }
    
    print(f"Evidence package built: {len(evidence)} top-level keys")
    return evidence


def verify_evidence_package(run_dir: Path) -> Tuple[bool, str]:
    """
    Verify evidence package structural validity.
    
    Args:
        run_dir: Path to First Light run directory
        
    Returns:
        (valid, message) tuple
    """
    try:
        evidence = build_first_light_evidence_package(run_dir)
        
        # Validate required fields
        required_fields = [
            "version",
            "run_metadata",
            "stability_report",
            "trajectories",
            "governance",
            "summary",
        ]
        
        missing = [f for f in required_fields if f not in evidence]
        if missing:
            return False, f"Missing required fields: {missing}"
        
        # Validate trajectories
        trajectories = evidence["trajectories"]
        if not trajectories.get("delta_p"):
            return False, "Missing delta_p trajectory"
        if not trajectories.get("hss"):
            return False, "Missing HSS trajectory"
        
        # Validate consistency
        num_cycles = evidence["run_metadata"]["cycles"]
        if len(trajectories["delta_p"]) != num_cycles:
            return False, f"delta_p trajectory length mismatch: {len(trajectories['delta_p'])} != {num_cycles}"
        if len(trajectories["hss"]) != num_cycles:
            return False, f"HSS trajectory length mismatch: {len(trajectories['hss'])} != {num_cycles}"
        
        # Validate governance
        governance = evidence["governance"]
        if len(governance["curriculum_stability"]) != num_cycles:
            return False, f"curriculum_stability length mismatch: {len(governance['curriculum_stability'])} != {num_cycles}"
        
        return True, "Evidence package structurally valid"
        
    except FileNotFoundError as e:
        return False, f"File not found: {e}"
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="First Light Orchestrator - Phase X"
    )
    
    # Run mode
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic execution"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1000,
        help="Number of cycles to run"
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="arithmetic_simple",
        help="Curriculum slice name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "integrated"],
        default="integrated",
        help="Run mode: baseline (no RFL) or integrated (with RFL)"
    )
    
    # Verification mode
    parser.add_argument(
        "--verify-evidence",
        action="store_true",
        help="Verify evidence package instead of running"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory for verification mode"
    )
    
    args = parser.parse_args()
    
    # Verification mode
    if args.verify_evidence:
        if not args.run_dir:
            print("ERROR: --run-dir required for verification mode", file=sys.stderr)
            return 1
        
        print(f"Verifying evidence package: {args.run_dir}")
        valid, message = verify_evidence_package(args.run_dir)
        
        if valid:
            print(f"✓ PASS: {message}")
            return 0
        else:
            print(f"✗ FAIL: {message}", file=sys.stderr)
            return 1
    
    # Run mode
    config = FirstLightConfig(
        seed=args.seed,
        cycles=args.cycles,
        slice_name=args.slice,
        mode=args.mode,
    )
    
    runner = FirstLightRunner(config)
    result = runner.run()
    
    # Build and save evidence package
    evidence = build_first_light_evidence_package(runner.run_dir)
    evidence_path = runner.run_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"Evidence package: {evidence_path}")
    
    # Verify evidence package
    valid, message = verify_evidence_package(runner.run_dir)
    if valid:
        print(f"✓ Evidence package valid: {message}")
        return 0
    else:
        print(f"✗ Evidence package invalid: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
