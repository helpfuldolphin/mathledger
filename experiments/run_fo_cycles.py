"""
First Organism Cycle Runner
===========================

A standalone harness for executing back-to-back First Organism cycles.
Supports 'baseline' (RFL OFF) and 'rfl' (RFL ON) modes with hermetic determinism.

Usage:
    # Default slice (first-organism-slice):
    uv run python experiments/run_fo_cycles.py --mode=baseline --cycles=1000 --out=results/fo_baseline.jsonl
    uv run python experiments/run_fo_cycles.py --mode=rfl --cycles=1000 --out=results/fo_rfl.jsonl

    # Wide Slice (slice_medium) for RFL uplift experiments:
    uv run python experiments/run_fo_cycles.py \
      --mode=baseline \
      --cycles=1000 \
      --slice-name=slice_medium \
      --system=pl \
      --out=results/fo_baseline_wide.jsonl

    uv run python experiments/run_fo_cycles.py \
      --mode=rfl \
      --cycles=1000 \
      --slice-name=slice_medium \
      --system=pl \
      --out=results/fo_rfl_wide.jsonl
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Canonical Imports ---
from substrate.repro.determinism import (
    deterministic_run_id,
    deterministic_unix_timestamp,
    deterministic_hash,
    deterministic_seed_from_content,
)
from ledger.ui_events import (
    capture_ui_event,
    ui_event_store,
    consume_ui_artifacts,
)
from ledger.blocking import seal_block_with_dual_roots
from curriculum.gates import (
    GateEvaluator,
    NormalizedMetrics,
    make_first_organism_slice,
    load,
    CurriculumSlice,
)
from derivation.pipeline import (
    run_slice_for_test,
    make_first_organism_derivation_slice,
    make_first_organism_seed_statements,
)
from rfl.config import RFLConfig, CurriculumSlice as RFLCurriculumSlice
from rfl.runner import RFLRunner
from substrate.bridge.context import AttestedRunContext

# --- Constants ---
MDAP_EPOCH_SEED = 0x4D444150
NAMESPACE = "first-organism-cycle-runner"


class CycleRunner:
    def __init__(self, mode: str, output_path: Path, slice_name: Optional[str] = None, system: str = "pl"):
        self.mode = mode
        self.output_path = output_path
        self.results = []
        self.slice_name = slice_name
        self.system = system
        
        # Load curriculum slice
        self.slice_cfg = self._load_slice()
        
        # Initialize RFL runner once if in RFL mode to maintain policy state
        self.rfl_runner: Optional[RFLRunner] = None
        if self.mode == 'rfl':
            self._init_rfl_runner()
    
    def _load_slice(self) -> CurriculumSlice:
        """
        Load curriculum slice by name from curriculum.yaml, or use default.
        
        For Wide Slice runs (RFL uplift experiments), pass --slice-name=slice_medium
        to use the medium difficulty slice from curriculum.yaml (atoms=5, depth_max=7).
        """
        if self.slice_name:
            # Load from curriculum.yaml
            system_cfg = load(self.system)
            for slice_obj in system_cfg.slices:
                if slice_obj.name == self.slice_name:
                    return slice_obj
            raise ValueError(f"Slice '{self.slice_name}' not found in system '{self.system}'")
        else:
            # Default to first-organism-slice (hardcoded for backward compatibility)
            return make_first_organism_slice()

    def _init_rfl_runner(self):
        """Initialize RFL runner with mocked DB access."""
        # Extract params from loaded slice
        params = self.slice_cfg.params
        derive_steps = params.get("derive_steps", 1)
        max_breadth = params.get("breadth_max", params.get("max_breadth", 1))
        max_total = params.get("total_max", params.get("max_total", 1))
        depth_max = params.get("depth_max", params.get("max_depth", 1))
        
        config = RFLConfig(
            experiment_id="fo-cycle-experiment",
            num_runs=10000, # Large enough for experiment
            random_seed=MDAP_EPOCH_SEED,
            system_id=1,
            derive_steps=derive_steps,
            max_breadth=max_breadth,
            max_total=max_total,
            depth_max=depth_max,
            bootstrap_replicates=1000,
            coverage_threshold=0.01,
            uplift_threshold=0.0,
            dual_attestation=False, # We handle attestation externally
            curriculum=[
                RFLCurriculumSlice(
          
          
                    name=self.slice_cfg.name,
                    start_run=1,
                    end_run=10000,
                    derive_steps=derive_steps,
                    max_breadth=max_breadth,
           
                    max_total=max_total,
                    depth_max=depth_max,
                )
            ],
        )
        # Mock DB loading
        with patch("rfl.runner.load_baseline_from_db", return_value=[]):
            self.rfl_runner = RFLRunner(config)

    def _setup_cycle(self, cycle_index: int):
        """Reset per-cycle state."""
        ui_event_store.clear()
        # Determine cycle-specific seed from (cycle_index, fixed_epoch_seed) for determinism
        cycle_seed = MDAP_EPOCH_SEED + cycle_index
        return cycle_seed

    def run_cycle(self, cycle_index: int) -> Dict[str, Any]:
        cycle_seed = self._setup_cycle(cycle_index)
        seed_content = f"cycle-{cycle_index}-{cycle_seed}"

        # --- Phase 1: UI Event (U_t) ---
        event_id = deterministic_run_id("ui-event", NAMESPACE, seed_content)
        timestamp = deterministic_unix_timestamp(deterministic_seed_from_content(seed_content, NAMESPACE))
        
        # Deterministically vary the target hash slightly or keep constant?
        # For now, constant to match the test baseline, but unique event ID.
        statement_hash = deterministic_hash(f"p -> p {cycle_index}") 

        ui_event = {
            "event_id": event_id,
            "event_type": "select_statement",
            "actor": "cycle-runner",
            "statement_hash": statement_hash,
            "action": "toggle_abstain",
            "meta": {"origin": "cycle-runner", "cycle": cycle_index},
            "timestamp": timestamp,
        }
        capture_ui_event(ui_event)
        
        # --- Phase 2: Curriculum Gate ---
        # We use a passing metrics set
        slice_cfg = self.slice_cfg
        metrics_raw = {
            "metrics": {
                "coverage": {"ci_lower": 0.95, "sample_size": 24},
                "proofs": {"abstention_rate": 12.0, "attempt_mass": 3200},
                "curriculum": {
                    "active_slice": {"wallclock_minutes": 45.0, "proof_velocity_cv": 0.05}
                },
                "throughput": {
                    "proofs_per_hour": 240.0,
                    "coefficient_of_variation": 0.04,
                    "window_minutes": 60,
                },
                "queue": {"backlog_fraction": 0.12},
            },
            "provenance": {"attestation_hash": f"cycle-{cycle_index}"},
        }
        normalized = NormalizedMetrics.from_raw(metrics_raw)
        gate_statuses = GateEvaluator(normalized, slice_cfg).evaluate()
        gates_passed = all(s.passed for s in gate_statuses)

        # --- Phase 3: Derivation ---
        # Use the loaded curriculum slice directly for derivation
        # If using a named slice from curriculum.yaml, use it; otherwise use default
        if self.slice_name:
            # Use the curriculum slice directly (it's already a CurriculumSlice)
            derivation_slice = slice_cfg
        else:
            # Default first-organism derivation slice
            derivation_slice = make_first_organism_derivation_slice()
        
        # Vary seed statements based on cycle_index to ensure diversity
        # This makes each cycle produce different candidates while maintaining determinism
        # CRITICAL: Include both non-tautologies AND tautologies so some statements verify
        atom_pool = ['p', 'q', 'r', 's', 't']
        cycle_atom_idx = cycle_index % len(atom_pool)
        next_atom_idx = (cycle_index + 1) % len(atom_pool)
        atom1 = atom_pool[cycle_atom_idx]
        atom2 = atom_pool[next_atom_idx]
        
        # Create cycle-specific seeds with mix of non-tautologies and tautologies
        from normalization.canon import normalize, normalize_pretty
        from derivation.derive_utils import sha256_statement
        from derivation.pipeline import StatementRecord, _canonical_pretty
        cycle_seeds = []
        
        # Seed 1: Atom (non-tautology) - varies by cycle
        expr1 = atom1
        normalized1 = normalize(expr1)
        if normalized1:
            cycle_seeds.append(
                StatementRecord(
                    normalized=normalized1,
                    hash=sha256_statement(normalized1),
                    pretty=_canonical_pretty(normalized1),
                    rule="seed:atom",
                    is_axiom=False,
                    mp_depth=0,
                    parents=(),
                    verification_method="seed",
                )
            )
        
        # Seed 2: Contingent implication (non-tautology) - varies by cycle
        # This will produce non-tautologies via MP
        expr2 = f"({atom1}->({atom2}))"
        normalized2 = normalize(expr2)
        if normalized2:
            cycle_seeds.append(
                StatementRecord(
                    normalized=normalized2,
                    hash=sha256_statement(normalized2),
                    pretty=_canonical_pretty(normalized2),
                    rule="seed:implication",
                    is_axiom=False,
                    mp_depth=0,
                    parents=(),
                    verification_method="seed",
                )
            )
        
        # Seed 3: Tautology (p->p) - ensures some statements verify
        # This is a tautology itself and can also produce more via MP
        expr3 = f"({atom1}->{atom1})"
        normalized3 = normalize(expr3)
        if normalized3:
            cycle_seeds.append(
                StatementRecord(
                    normalized=normalized3,
                    hash=sha256_statement(normalized3),
                    pretty=_canonical_pretty(normalized3),
                    rule="seed:tautology",
                    is_axiom=False,
                    mp_depth=0,
                    parents=(),
                    verification_method="seed",
                )
            )
        
        # Seed 4: K combinator pattern (tautology) - x->(y->x)
        # This produces tautologies and enables more via MP
        expr4 = f"({atom1}->({atom2}->{atom1}))"
        normalized4 = normalize(expr4)
        if normalized4:
            cycle_seeds.append(
                StatementRecord(
                    normalized=normalized4,
                    hash=sha256_statement(normalized4),
                    pretty=_canonical_pretty(normalized4),
                    rule="seed:k_combinator",
                    is_axiom=False,
                    mp_depth=0,
                    parents=(),
                    verification_method="seed",
                )
            )
        
        # Run derivation with higher limit to get diverse candidates
        derivation_limit = max(10, derivation_slice.params.get('total_max', 10))
        # Use cycle-specific seeds if available, otherwise fall back to default
        seeds_to_use = cycle_seeds if cycle_seeds else list(make_first_organism_seed_statements())
        
        # Get policy weights and success history from RFL runner if in RFL mode
        policy_weights = None
        success_count = None
        stop_after_verified = None
        max_candidates = None
        
        # For uplift experiments: limit candidates to make ordering matter
        # Baseline uses random (seeded) order, RFL uses policy-based order
        # Phase I: Currently set tight (2) for negative control; Phase II will tune this
        if self.slice_cfg.name == "slice_uplift_proto":
            max_candidates = 2  # Tight budget - ordering determines what gets explored
        
        if self.mode == 'rfl' and self.rfl_runner:
            policy_weights = self.rfl_runner.policy_weights.copy()
            success_count = self.rfl_runner.success_count.copy()  # Pass success history
        
        derivation_result = run_slice_for_test(
            derivation_slice,
            existing=seeds_to_use,
            limit=derivation_limit,
            policy_weights=policy_weights,
            success_count=success_count,
            stop_after_verified=stop_after_verified,
            max_candidates=max_candidates,
            mode=self.mode,
            cycle_seed=cycle_index,  # Use cycle index as seed for deterministic randomization
        )
        
        # Collect all candidates (abstained first, then verified)
        abstained_list = list(derivation_result.abstained_candidates) if derivation_result.abstained_candidates else []
        verified_list = list(derivation_result.statements) if derivation_result.statements else []
        all_candidates = abstained_list + verified_list
        num_abstained = len(abstained_list)
        
        # Use cycle_index to deterministically select candidate
        # This ensures diversity across cycles while maintaining reproducibility
        candidate = None
        is_abstention = False
        status = "error"
        verification_method = "none"
        
        if all_candidates:
            candidate_index = cycle_index % len(all_candidates)
            candidate = all_candidates[candidate_index]
            
            # Determine if selected candidate is verified or abstained
            # Abstained candidates come first, then verified
            if candidate_index < num_abstained:
                # Candidate is from abstained list
                is_abstention = True
                status = "abstain"
            else:
                # Candidate is from verified list
                is_abstention = False
                status = "verified"
            
            verification_method = candidate.verification_method if candidate else "none"
        
        candidate_hash = candidate.hash if candidate else "none"

        # --- Phase 4: Attestation (H_t) ---
        proof_payload = {
            "statement": candidate.pretty if candidate else "nil",
            "statement_hash": candidate_hash,
            "status": status,
            "prover": "lean-interface",
            "verification_method": verification_method,
            "reason": "cycle-test",
        }
        
        # Seal block (computes R_t, U_t, H_t)
        # Note: consume_ui_artifacts() is called inside if ui_events is None
        block = seal_block_with_dual_roots("pl", [proof_payload])
        
        h_t = block["composite_attestation_root"]
        r_t = block["reasoning_merkle_root"]
        u_t = block["ui_merkle_root"]

        # --- Phase 5: RFL Metabolism ---
        rfl_stats = {"executed": False}
        if self.mode == 'rfl' and self.rfl_runner and candidate:
            # Use actual derivation results for abstention metrics
            # For debug/uplift slices: abstention = (verified == 0)
            num_verified = derivation_result.n_verified if derivation_result else 0
            num_abstained = derivation_result.n_abstained if derivation_result else 0
            total_candidates = derivation_result.n_candidates if derivation_result else 1
            
            # Abstention rate: fraction that did NOT find proofs
            # If verified > 0, abstention_rate should be lower
            abstention_rate = 1.0 - (num_verified / max(total_candidates, 1.0)) if total_candidates > 0 else 1.0
            abstention_mass = float(num_abstained)
            attempt_mass = float(total_candidates)
            
            attestation_context = AttestedRunContext(
                slice_id=slice_cfg.name,
                statement_hash=candidate_hash,
                proof_status="success" if num_verified > 0 else "failure",
                block_id=cycle_index + 1,
                composite_root=h_t,
                reasoning_root=r_t,
                ui_root=u_t,
                abstention_metrics={"rate": abstention_rate, "mass": abstention_mass},
                policy_id=f"policy-{cycle_index}",
                metadata={
                    "attempt_mass": attempt_mass,
                    "verified_count": num_verified,
                    "abstention_count": num_abstained,
                    "abstention_breakdown": {"derivation_abstain": num_abstained},
                    "first_organism_abstentions": int(abstention_mass),
                },
            )
            
            # We patch inside the loop too just in case
            with patch("rfl.runner.load_baseline_from_db", return_value=[]):
                 result = self.rfl_runner.run_with_attestation(attestation_context)
            
            ledger_entry = self.rfl_runner.policy_ledger[-1]
            rfl_stats = {
                "executed": True,
                "policy_update": result.policy_update_applied,
                "update_count": self.rfl_runner.policy_update_count,  # Total updates applied so far
                "policy_ledger_length": len(self.rfl_runner.policy_ledger),  # Number of cycles processed
                "symbolic_descent": ledger_entry.symbolic_descent,
                "policy_reward": ledger_entry.policy_reward,
                "abstention_histogram": self.rfl_runner.abstention_histogram.copy(),
                "abstention_rate_before": ledger_entry.abstention_fraction,  # For metrics cross-reference
                "abstention_rate_after": ledger_entry.abstention_fraction,
            }
            
            # Optional: Trigger batched metrics export every N cycles (loose coupling)
            # This allows RFL metrics to be written to JSONL for later analysis
            if hasattr(self.rfl_runner, 'metrics_logger') and self.rfl_runner.metrics_logger:
                # Metrics logger will have already logged via run_with_attestation
                # This is just for visibility in FO cycle logs
                pass

        # --- Result Assembly ---
        # Canonical JSON format for Dyno Chart analysis
        # Ensure stable key ordering and no trailing commas
        # Uplift-specific fields: proof_found, success (explicit boolean)
        
        # For debug/uplift slices: use derivation-based success metric
        # Success = did we find ANY verified proofs in this cycle?
        # This is more appropriate for uplift experiments than candidate-selection-based logic
        num_verified = derivation_result.n_verified if derivation_result else 0
        is_debug_slice = self.slice_cfg.name in ("slice_debug_uplift", "slice_easy_fo", "slice_uplift_proto")
        
        if is_debug_slice:
            # Derivation-based success: did we prove anything?
            # For slice_uplift_proto, require at least 7 verified proofs
            # (with medium slice params, avg is ~6.8, so ~80% baseline success with target=7)
            if self.slice_cfg.name == "slice_uplift_proto":
                proof_found = (num_verified >= 7)
            else:
                proof_found = (num_verified > 0)
            success = proof_found
            # Override status/abstention based on derivation results, not candidate selection
            if proof_found:
                status = "success"
                is_abstention = False
            else:
                status = "abstain"
                is_abstention = True
        else:
            # Original logic: based on selected candidate
            proof_found = (status == "verified")
            success = proof_found  # Alias for clarity in uplift analysis
        
        result = {
            "cycle": cycle_index,
            "slice_name": self.slice_cfg.name,
            "status": status,
            "method": verification_method,
            "abstention": is_abstention,
            "mode": self.mode,
            "roots": {
                "h_t": h_t,
                "r_t": r_t,
                "u_t": u_t,
            },
            "derivation": {
                "candidates": derivation_result.n_candidates,
                "abstained": derivation_result.n_abstained,
                "verified": derivation_result.n_verified,
                "candidate_hash": candidate_hash,
                "candidate_text": candidate.pretty if candidate else "none",
                "total_candidates_available": len(all_candidates),
            },
            "rfl": rfl_stats,
            "gates_passed": gates_passed,
            # Uplift analysis fields
            "proof_found": proof_found,
            "success": success,
            # Policy weights for debugging (only in RFL mode)
            "policy_weights": self.rfl_runner.policy_weights.copy() if self.mode == 'rfl' and self.rfl_runner else None,
        }
        return result

    def run(self, cycles: int):
        print(f"Starting {cycles} cycles in mode='{self.mode}' with slice='{self.slice_cfg.name}'...")
        with open(self.output_path, 'w') as f:
            for i in range(cycles):
                result = self.run_cycle(i)
                # Use canonical JSON (no trailing commas, stable key ordering)
                f.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
                if (i + 1) % 100 == 0:
                    sys.stdout.write(f"\rProgress: {i + 1}/{cycles}")
                    sys.stdout.flush()
        print(f"\nDone. Results written to {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run First Organism cycles.")
    parser.add_argument("--mode", choices=["baseline", "rfl"], default="baseline", help="Execution mode")
    parser.add_argument("--cycles", type=int, default=10, help="Number of cycles to run")
    parser.add_argument("--out", type=str, default="results.jsonl", help="Output JSONL file path")
    parser.add_argument("--slice-name", type=str, default=None, help="Curriculum slice name (default: first-organism-slice)")
    parser.add_argument("--system", type=str, default="pl", help="System slug (default: pl)")
    
    args = parser.parse_args()
    
    runner = CycleRunner(args.mode, Path(args.out), slice_name=args.slice_name, system=args.system)
    runner.run(args.cycles)


if __name__ == "__main__":
    main()
