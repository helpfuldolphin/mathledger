"""
RFL 40-Run Orchestrator

Executes 40 derivation experiments and verifies reflexive metabolism:
- Coverage ≥ 92% (bootstrap CI lower bound)
- Uplift > 1.0 (bootstrap CI lower bound)
"""

import hashlib
import json
import logging
import os
import redis
import time
import numpy as np
from typing import List, Dict, Any, Optional, Sequence
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from collections import Counter
import warnings

from backend.safety import evaluate_hard_gate_decision, check_gate_decision
from backend.tda import TDAMode

LATENCY_BUCKET_THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
ABSTENTION_BUCKET_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)

from .config import RFLConfig, CurriculumSlice
from .experiment import RFLExperiment, ExperimentResult
from .coverage import CoverageTracker, CoverageMetrics, load_baseline_from_db
from substrate.repro.determinism import deterministic_timestamp
from .bootstrap_stats import (
    compute_coverage_ci,
    compute_uplift_ci,
    verify_metabolism,
    BootstrapResult,
    bootstrap_percentile
)
from .audit import RFLAuditLog, SymbolicDescentGradient, StepIdComputation
from .experiment_logging import RFLExperimentLogger
from .provenance import ManifestBuilder

# ---------------- Logger ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RFLRunner")


@dataclass
class RunLedgerEntry:
    """Structured curriculum ledger entry for a single RFL run."""

    run_id: str
    slice_name: str  # Resolved curriculum slice name
    status: str
    coverage_rate: float
    novelty_rate: float
    throughput: float
    success_rate: float
    abstention_fraction: float
    policy_reward: float
    symbolic_descent: float
    budget_spent: int
    derive_steps: int
    max_breadth: int
    max_total: int
    abstention_breakdown: Dict[str, int] = field(default_factory=dict)
    # Attestation-specific fields (optional, populated by run_with_attestation)
    attestation_slice_id: Optional[str] = None  # Original slice_id from attestation
    composite_root: Optional[str] = None  # H_t for traceability


from substrate.bridge.context import AttestedRunContext


@dataclass
class RflResult:
    """Outcome emitted by run_with_attestation."""

    policy_update_applied: bool
    source_root: str
    abstention_mass_delta: float
    step_id: str
    ledger_entry: Optional[RunLedgerEntry] = None


class RFLRunner:
    """Orchestrator for 40-run RFL experiment suite."""

    def __init__(self, config: RFLConfig):
        """
        Initialize RFL runner.

        Args:
            config: RFL experiment configuration
        """
        self.config = config
        config.validate()

        # Create artifacts directory
        Path(config.artifacts_dir).mkdir(parents=True, exist_ok=True)

        # Initialize provenance system
        self.manifest_builder = ManifestBuilder()

        # Initialize components
        self.experiment = RFLExperiment(
            db_url=config.database_url,
            system_id=config.system_id
        )

        # Load baseline statements for coverage tracking
        baseline_hashes = load_baseline_from_db(
            config.database_url,
            config.system_id
        )
        logger.info(f"[INIT] Loaded {len(baseline_hashes)} baseline statements from database")

        self.coverage_tracker = CoverageTracker(baseline_statements=baseline_hashes)

        # Results storage
        self.run_results: List[ExperimentResult] = []
        self.coverage_ci: Optional[BootstrapResult] = None
        self.uplift_ci: Optional[BootstrapResult] = None
        self.metabolism_passed: bool = False
        self.metabolism_message: str = ""
        self.policy_ledger: List[RunLedgerEntry] = []
        self.abstention_histogram: Counter[str] = Counter()
        self.dual_attestation_records: Dict[str, Any] = {
            "enabled": self.config.dual_attestation,
            "checks": {}
        }
        self.dual_attestation_records.setdefault("attestations", [])
        self._previous_coverage_rate: Optional[float] = None
        self._throughput_reference: float = 1.0
        self._has_throughput_reference: bool = False
        self.abstention_fraction: float = 0.0
        self.first_organism_runs_total: int = 0
        self.policy_update_count: int = 0  # Track total number of policy updates applied
        
        # Policy weights for candidate scoring (simple 3-parameter policy)
        # These weights control how candidates are ordered during search
        self.policy_weights: Dict[str, float] = {
            "len": 0.0,      # Weight for candidate text length
            "depth": 0.0,    # Weight for candidate AST depth
            "success": 0.0,  # Weight for success history (how often this hash appeared in successful cycles)
        }
        
        # Success history tracking: per candidate hash, track success correlation
        # This allows the policy to learn which formulas tend to lead to successful cycles
        self.success_count: Dict[str, int] = {}  # candidate_hash -> number of successful cycles
        self.attempt_count: Dict[str, int] = {}   # candidate_hash -> number of cycles where it appeared

        # Audit log for RFL Law compliance (determinism verification)
        self.audit_log = RFLAuditLog(seed=self.config.random_seed)

        # Experiment Logger (Schema v1)
        self.experiment_logger = RFLExperimentLogger(config)

        # Metrics Logger for Wide Slice (JSONL format)
        # Only enable if experiment_id suggests Wide Slice usage
        self.metrics_logger: Optional[Any] = None
        if "wide_slice" in config.experiment_id.lower():
            from .metrics_logger import RFLMetricsLogger
            metrics_path = Path("results") / "rfl_wide_slice_runs.jsonl"
            self.metrics_logger = RFLMetricsLogger(str(metrics_path))
            logger.info(f"[INIT] Metrics logger enabled: {metrics_path}")

        # Telemetry
        self._redis_client = None
        # Skip Redis if explicitly disabled or for debug experiments
        if os.getenv("FIRST_ORGANISM_DISABLE_REDIS", "").lower() in ("1", "true", "yes"):
            logger.info("[INFO] Redis telemetry disabled via FIRST_ORGANISM_DISABLE_REDIS")
        else:
            try:
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self._redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test connection immediately to catch failures early
                if self._redis_client:
                    self._redis_client.ping()
            except Exception as e:
                logger.warning(f"[WARN] Telemetry: Redis not available for metrics: {e}")
                self._redis_client = None

    def _increment_metric(self, key: str, amount: float = 1.0):
        if self._redis_client:
            try:
                if isinstance(amount, float):
                    self._redis_client.incrbyfloat(key, amount)
                else:
                    self._redis_client.incr(key, int(amount))
            except Exception:
                pass

    @staticmethod
    def _bucket_label(value: float, thresholds: Sequence[float]) -> str:
        for threshold in thresholds:
            if value <= threshold:
                return str(threshold)
        return "+Inf"

    def _record_bucket(self, prefix: str, value: float, thresholds: Sequence[float]) -> None:
        label = self._bucket_label(value, thresholds)
        self._increment_metric(f"{prefix}:{label}")

    def _set_metric(self, key: str, value: Any):
        if self._redis_client:
            try:
                self._redis_client.set(key, str(value))
            except Exception:
                pass

    def _push_metric_history(self, key: str, value: str, max_len: int = 20):
        if self._redis_client:
            try:
                pipe = self._redis_client.pipeline()
                pipe.lpush(key, value)
                pipe.ltrim(key, 0, max_len - 1)
                pipe.execute()
            except Exception:
                pass

    def run_all(self) -> Dict[str, Any]:
        """
        Execute all experiments and compute metabolism metrics.

        Returns:
            Dictionary with complete results and verification status
        """
        logger.info("======================================================================")
        logger.info(f"RFL Experiment Suite: {self.config.experiment_id}")
        logger.info("======================================================================")
        logger.info("Configuration:")
        logger.info(f"  Runs: {self.config.num_runs}")
        logger.info(f"  Steps per run: {self.config.derive_steps}")
        logger.info(f"  Max breadth: {self.config.max_breadth}")
        logger.info(f"  Max total: {self.config.max_total}")
        logger.info(f"  Coverage threshold: {self.config.coverage_threshold}")
        logger.info(f"  Uplift threshold: {self.config.uplift_threshold}")
        logger.info(f"  Bootstrap replicates: {self.config.bootstrap_replicates}")
        curriculum_summary = ", ".join(
            f"{slice_cfg.name}[{slice_cfg.start_run}-{slice_cfg.end_run}]"
            for slice_cfg in self.config.curriculum
        )
        logger.info(f"  Curriculum: {curriculum_summary}")
        logger.info("======================================================================")

        # Phase 1: Execute experiments
        logger.info(f"Phase 1: Executing {self.config.num_runs} derivation experiments...")
        self._execute_experiments()

        # Phase 2: Compute coverage metrics
        logger.info("Phase 2: Computing coverage metrics...")
        self._compute_coverage_metrics()

        # Phase 3: Compute uplift metrics
        logger.info("Phase 3: Computing uplift metrics...")
        self._compute_uplift_metrics()

        # Phase 4: Verify metabolism
        logger.info("Phase 4: Verifying reflexive metabolism...")
        self._verify_metabolism()

        # Phase 5: Export results
        logger.info("Phase 5: Exporting results...")
        results = self._export_results()

        logger.info("======================================================================")
        logger.info("RFL Experiment Suite Complete")
        logger.info("======================================================================")
        logger.info(self.metabolism_message)
        logger.info("======================================================================")

        return results

    def _execute_experiments(self) -> None:
        """Execute all derivation experiments sequentially."""
        np.random.seed(self.config.random_seed)

        for i in range(self.config.num_runs):
            run_id = f"rfl_{self.config.experiment_id}_run_{i+1:02d}"
            slice_cfg = self.config.resolve_slice(i + 1)

            logger.info(
                f"[{i+1}/{self.config.num_runs}] Executing {run_id} "
                f"(slice='{slice_cfg.name}', steps={slice_cfg.derive_steps}, breadth={slice_cfg.max_breadth})..."
            )

            policy_context = {
                "slice": slice_cfg.name,
                "slice_index": i,
                "slice_bounds": [slice_cfg.start_run, slice_cfg.end_run],
                "derive_steps": slice_cfg.derive_steps,
                "max_breadth": slice_cfg.max_breadth,
                "max_total": slice_cfg.max_total,
                "depth_max": slice_cfg.depth_max
            }

            result = self.experiment.run(
                run_id=run_id,
                derive_steps=slice_cfg.derive_steps,
                max_breadth=slice_cfg.max_breadth,
                max_total=slice_cfg.max_total,
                depth_max=slice_cfg.depth_max,
                policy_context=policy_context,
                seed=self.config.random_seed + i
            )

            # --- Provenance: Generate Manifest ---
            effective_seed = self.config.random_seed + i
            manifest = self.manifest_builder.build(
                run_index=i + 1,
                config=self.config,
                result=result,
                effective_seed=effective_seed
            )
            
            if self.config.experiment_id in ["fo_1000_baseline", "fo_1000_rfl"]:
                 manifest_dir = Path(f"artifacts/experiments/rfl/{self.config.experiment_id}")
                 if self.config.num_runs > 1:
                     manifest_path = manifest_dir / f"manifest_run_{i+1:02d}.json"
                 else:
                     manifest_path = manifest_dir / "manifest.json"
            else:
                 manifest_path = Path(f"docs/evidence/manifests/RFL_RUN_{i+1:02d}.json")

            self.manifest_builder.save(manifest, manifest_path)
            # -------------------------------------

            self.run_results.append(result)
            self._merge_abstention_breakdown(result.abstention_breakdown)

            # Update coverage tracker
            if result.status == "success":
                coverage_metrics = self.coverage_tracker.record_run(
                    statement_hashes=result.statement_hashes,
                    run_id=run_id,
                    target_space_size=slice_cfg.max_total
                )
                reward = self._compute_reward_signal(result, coverage_metrics)
                symbolic_descent = self._update_symbolic_descent(coverage_metrics.coverage_rate)
                policy_context["reward"] = reward
                policy_context["symbolic_descent"] = symbolic_descent
                result.policy_context.update(policy_context)
                self._record_policy_entry(
                    result,
                    slice_cfg,
                    coverage_metrics=coverage_metrics,
                    reward=reward,
                    symbolic_descent=symbolic_descent
                )

                logger.info(
                    f"  Result: {result.successful_proofs} proofs, "
                    f"coverage={coverage_metrics.coverage_rate:.4f}, "
                    f"novelty={coverage_metrics.novelty_rate:.4f}, "
                    f"reward={reward:.4f}"
                )
            else:
                result.policy_context.update(policy_context)
                self._record_policy_entry(result, slice_cfg)
                logger.info(f"  Result: {result.status} - {result.error_message}")

        # Summary
        successful_runs = sum(1 for r in self.run_results if r.status == "success")
        logger.info(f"Execution complete: {successful_runs}/{self.config.num_runs} successful runs")

    def _merge_abstention_breakdown(self, breakdown: Dict[str, int]) -> None:
        """Aggregate abstention buckets across runs."""
        if not breakdown:
            return
        for key, value in breakdown.items():
            self.abstention_histogram[key] += int(value)

    def _compute_reward_signal(self, result: ExperimentResult, coverage_metrics: CoverageMetrics) -> float:
        """
        Compute deterministic reward shaping signal for policy adaptation.

        Components:
            - success rate (40%)
            - novelty rate (30%)
            - throughput ratio (30%), normalized to the first non-zero throughput
            - abstention penalty (1 - abstention_rate)
        """
        if not self._has_throughput_reference and result.throughput_proofs_per_hour > 0:
            self._throughput_reference = result.throughput_proofs_per_hour
            self._has_throughput_reference = True

        throughput_reference = self._throughput_reference if self._has_throughput_reference else 1.0
        throughput_ratio = (
            min(1.5, result.throughput_proofs_per_hour / throughput_reference)
            if throughput_reference > 0
            else 0.0
        )

        success_component = result.success_rate
        novelty_component = coverage_metrics.novelty_rate
        throughput_component = min(1.0, throughput_ratio)
        abstention_penalty = 1.0 - min(1.0, result.abstention_rate)

        composite = (
            0.4 * success_component +
            0.3 * novelty_component +
            0.3 * throughput_component
        )
        return max(0.0, composite * abstention_penalty)

    def _update_symbolic_descent(self, coverage_rate: Optional[float]) -> float:
        """Track symbolic descent delta between consecutive successful runs."""
        if coverage_rate is None:
            return 0.0

        if self._previous_coverage_rate is None:
            self._previous_coverage_rate = coverage_rate
            return coverage_rate

        delta = coverage_rate - self._previous_coverage_rate
        self._previous_coverage_rate = coverage_rate
        return delta

    def _record_policy_entry(
        self,
        result: ExperimentResult,
        slice_cfg: CurriculumSlice,
        coverage_metrics: Optional[CoverageMetrics] = None,
        reward: float = 0.0,
        symbolic_descent: float = 0.0
    ) -> None:
        """Append a ledger entry capturing policy dynamics for the run."""
        coverage_rate = coverage_metrics.coverage_rate if coverage_metrics else 0.0
        novelty_rate = coverage_metrics.novelty_rate if coverage_metrics else 0.0
        throughput = result.throughput_proofs_per_hour if result.status == "success" else 0.0

        entry = RunLedgerEntry(
            run_id=result.run_id,
            slice_name=slice_cfg.name,
            status=result.status,
            coverage_rate=float(coverage_rate),
            novelty_rate=float(novelty_rate),
            throughput=float(throughput),
            success_rate=float(result.success_rate),
            abstention_fraction=float(result.abstention_rate),
            policy_reward=float(reward),
            symbolic_descent=float(symbolic_descent),
            budget_spent=int(slice_cfg.max_total),
            derive_steps=int(slice_cfg.derive_steps),
            max_breadth=int(slice_cfg.max_breadth),
            max_total=int(slice_cfg.max_total),
            abstention_breakdown=dict(result.abstention_breakdown)
        )

        self.policy_ledger.append(entry)

    def run_with_attestation(self, attestation: AttestedRunContext) -> RflResult:
        """
        Consume a sealed dual-attestation payload (Hₜ, Rₜ, Uₜ) so the verification ladder (proof/abstain)
        materializes symbolic descent as described in the whitepaper.
        
        NOTE: FO hermetic and Wide Slice experiments do NOT depend on a live Lean kernel;
        they exercise the 'lean-disabled' abstention mode for deterministic behavior.
        The abstention_metrics in the attestation context reflect deterministic abstention
        decisions from the derivation pipeline (see derivation.verification.LeanFallback).
        """
        self.first_organism_runs_total += 1
        self._increment_metric("rfl_first_organism_runs_total")
        # Telemetry: First Organism Metrics
        self._increment_metric("ml:metrics:first_organism:runs_total")
        self._set_metric("ml:metrics:first_organism:last_ht", attestation.composite_root)

        # Latency: Computed from attestation metadata timestamps (not wall-clock)
        ui_ts = attestation.metadata.get("timestamp") or attestation.metadata.get("event_timestamp")
        ended_ts = attestation.metadata.get("first_organism_ended_at")
        if ui_ts and ended_ts:
            try:
                from datetime import datetime as dt
                start = dt.fromisoformat(ui_ts.replace("Z", "+00:00"))
                end = dt.fromisoformat(ended_ts.replace("Z", "+00:00"))
                latency = (end - start).total_seconds()
                if latency >= 0:
                    self._set_metric("ml:metrics:first_organism:latency_seconds", latency)
                    self._increment_metric("ml:metrics:first_organism:latency_sum", latency)
                    self._increment_metric("ml:metrics:first_organism:latency_count")
                    self._record_bucket(
                        "ml:metrics:first_organism:latency_bucket",
                        latency,
                        LATENCY_BUCKET_THRESHOLDS,
                    )
            except (ValueError, TypeError):
                pass

        duration = float(attestation.metadata.get("first_organism_duration_seconds", 0.0))
        if duration >= 0:
            self._set_metric("ml:metrics:first_organism:duration_seconds", duration)
            self._push_metric_history("ml:metrics:first_organism:duration_history", f"{duration:.6f}")

        last_ts = attestation.metadata.get("first_organism_ended_at")
        if not last_ts:
            last_ts = deterministic_timestamp(0).isoformat() + "Z"
        self._set_metric("ml:metrics:first_organism:last_run_timestamp", last_ts)

        abstentions = int(attestation.metadata.get("first_organism_abstentions", 0))
        if abstentions >= 0:
            self._set_metric("ml:metrics:first_organism:last_abstentions", abstentions)
            self._push_metric_history("ml:metrics:first_organism:abstention_history", str(abstentions))

        abstention_rate = float(attestation.abstention_rate)
        if abstention_rate >= 0:
            self._set_metric("ml:metrics:first_organism:last_abstention_rate", abstention_rate)
            self._record_bucket(
                "ml:metrics:first_organism:abstention_bucket",
                abstention_rate,
                ABSTENTION_BUCKET_THRESHOLDS,
            )

        try:
            self._validate_attestation(attestation)
        except Exception:
            self._increment_metric("ml:metrics:first_organism:runs_failed")
            raise

        self._increment_metric("ml:metrics:first_organism:runs_completed")

        # ═══════════════════════════════════════════════════════════════════
        # CORTEX: TDA Hard Gate Decision (BLOCKING)
        # ═══════════════════════════════════════════════════════════════════
        # Evaluate hard gate decision before proceeding with RFL update.
        # This is the Brain - the central decision point that can block execution.
        
        if self.config.tda_enabled:
            # Build context for TDA evaluation
            gate_context = {
                "abstention_rate": attestation.abstention_rate,
                "coverage_rate": attestation.metadata.get("coverage_rate", 0.0),
                "verified_count": attestation.metadata.get("verified_count", 0),
                "cycle_index": self.first_organism_runs_total,
                "composite_root": attestation.composite_root,
                "slice_id": attestation.slice_id,
            }
            
            # Parse TDA mode from config
            try:
                tda_mode = TDAMode(self.config.tda_mode)
            except ValueError:
                logger.warning(f"Invalid TDA mode '{self.config.tda_mode}', defaulting to BLOCK")
                tda_mode = TDAMode.BLOCK
            
            # Evaluate gate decision (the Cortex)
            gate_envelope = evaluate_hard_gate_decision(gate_context, tda_mode)
            
            # Log the decision
            logger.info(
                f"[CORTEX] TDA Gate Decision: {gate_envelope.decision} "
                f"(mode={gate_envelope.tda_mode}, status={gate_envelope.slo.status.value})"
            )
            logger.info(f"[CORTEX] {gate_envelope.slo.message}")
            
            # Record gate decision in attestation records for audit
            self.dual_attestation_records.setdefault("gate_decisions", []).append({
                "cycle": self.first_organism_runs_total,
                "composite_root": attestation.composite_root,
                "envelope": gate_envelope.to_dict(),
            })
            
            # BLOCKING CHECK: Enforce gate decision
            # If gate says block, execution stops here with RuntimeError
            check_gate_decision(gate_envelope)
        
        # ═══════════════════════════════════════════════════════════════════

        slice_cfg = self._resolve_slice(attestation.slice_id)
        policy_id = attestation.policy_id or "default"

        step_material = (
            f"{self.config.experiment_id}|{slice_cfg.name}|{policy_id}|{attestation.composite_root}"
        )
        step_id = hashlib.sha256(step_material.encode("utf-8")).hexdigest()

        attempt_mass = float(
            attestation.metadata.get("attempt_mass", max(attestation.abstention_mass, 1.0))
        )
        expected_mass = self.config.abstention_tolerance * attempt_mass
        abstention_mass_delta = attestation.abstention_mass - expected_mass
        abstention_rate_delta = attestation.abstention_rate - self.config.abstention_tolerance
        policy_update_applied = (
            abs(abstention_mass_delta) > 1e-9 or abs(abstention_rate_delta) > 1e-9
        )

        reward = max(0.0, 1.0 - max(attestation.abstention_rate, 0.0))
        symbolic_descent = -abstention_rate_delta

        if policy_update_applied:
            self.policy_update_count += 1
            breakdown = attestation.metadata.get("abstention_breakdown", {})
            if isinstance(breakdown, dict):
                for key, value in breakdown.items():
                    self.abstention_histogram[key] += int(value)
            self.abstention_histogram["attestation_mass"] += int(
                round(attestation.abstention_mass)
            )
            self.abstention_histogram["attestation_events"] += 1
            
            # Update policy weights based on verified count (graded reward)
            # Use verified_count as a graded signal instead of binary success
            verified_count = attestation.metadata.get("verified_count", 0)
            target_verified = 7  # Match the success threshold for slice_uplift_proto
            
            # Track success history for the candidate hash from this cycle
            candidate_hash = attestation.statement_hash
            cycle_success = (verified_count >= target_verified)
            
            # Update success/attempt counts for this candidate hash
            self.attempt_count[candidate_hash] = self.attempt_count.get(candidate_hash, 0) + 1
            if cycle_success:
                self.success_count[candidate_hash] = self.success_count.get(candidate_hash, 0) + 1
            
            # Graded reward: how far above/below target
            reward = verified_count - target_verified
            
            # Increased learning rate (10x larger) to make policy matter more
            eta = 0.1  # Increased from 0.01 (10x)
            
            # More aggressive update scaling - use reward directly (not normalized)
            # This makes even small differences in verified count matter more
            # Use simple heuristics: prefer shorter formulas and moderate depth
            # Positive reward (verified > target): reinforce current strategy
            # Negative reward (verified < target): try opposite strategy
            if reward > 0:
                # Success: prefer shorter formulas (negative len weight)
                # and moderate depth (slight positive depth weight)
                # Scale by reward magnitude (more proofs = bigger update)
                update_magnitude = min(abs(reward) * 0.5, 2.0)  # Cap at 2.0 for stability
                self.policy_weights["len"] += eta * (-0.1) * update_magnitude  # Prefer shorter
                self.policy_weights["depth"] += eta * (+0.05) * update_magnitude  # Slight preference for depth
                # Strongly reinforce success history feature when we succeed
                # Use reward magnitude: more proofs above target = bigger boost
                self.policy_weights["success"] += eta * reward  # Direct scaling by reward
            elif reward < 0:
                # Failure: push away from current bias
                # Even small failures should trigger meaningful updates
                update_magnitude = min(abs(reward) * 0.5, 2.0)  # Cap at 2.0
                self.policy_weights["len"] += eta * (+0.1) * update_magnitude  # Try longer
                self.policy_weights["depth"] += eta * (-0.05) * update_magnitude  # Try different depth
                # Gently decrease success weight (10x smaller penalty than success boost)
                # This allows success weight to grow over time if successes outnumber failures
                self.policy_weights["success"] += eta * 0.1 * reward  # Small penalty (reward is negative)
            # If reward == 0, still do a small update to encourage exploration
            else:
                # Exactly at threshold: small random-walk update to explore
                self.policy_weights["len"] += eta * 0.01 * (-0.1)  # Tiny preference for shorter
                self.policy_weights["depth"] += eta * 0.01 * (+0.05)  # Tiny preference for depth
                # Small positive update for success feature (we hit target, even if barely)
                self.policy_weights["success"] += eta * 0.05  # Small positive boost
            
            # CRITICAL: Clamp success weight to non-negative
            # We never want to penalize successful hashes - only prefer them or ignore them
            self.policy_weights["success"] = max(0.0, self.policy_weights["success"])

        self.abstention_fraction = max(self.abstention_fraction, attestation.abstention_rate)

        ledger_entry = RunLedgerEntry(
            run_id=step_id,
            slice_name=slice_cfg.name,
            status="attestation",
            coverage_rate=float(attestation.metadata.get("coverage_rate", 0.0)),
            novelty_rate=float(attestation.metadata.get("novelty_rate", 0.0)),
            throughput=float(attestation.metadata.get("throughput", 0.0)),
            success_rate=float(attestation.metadata.get("success_rate", 0.0)),
            abstention_fraction=float(attestation.abstention_rate),
            policy_reward=float(reward),
            symbolic_descent=float(symbolic_descent),
            budget_spent=int(round(attempt_mass)),
            derive_steps=int(attestation.metadata.get("derive_steps", slice_cfg.derive_steps)),
            max_breadth=int(attestation.metadata.get("max_breadth", slice_cfg.max_breadth)),
            max_total=int(attestation.metadata.get("max_total", slice_cfg.max_total)),
            abstention_breakdown=dict(attestation.metadata.get("abstention_breakdown", {})),
            attestation_slice_id=attestation.slice_id,
            composite_root=attestation.composite_root,
        )
        self.policy_ledger.append(ledger_entry)

        # Log to JSONL metrics logger if enabled
        if self.metrics_logger:
            previous_rate = (
                self.policy_ledger[-2].abstention_fraction
                if len(self.policy_ledger) > 1
                else None
            )
            self.metrics_logger.log_run(ledger_entry, previous_abstention_rate=previous_rate)

        self.dual_attestation_records.setdefault("attestations", []).append(
            {
                "step_id": step_id,
                "composite_root": attestation.composite_root,
                "reasoning_root": attestation.reasoning_root,
                "ui_root": attestation.ui_root,
                "metadata": attestation.metadata,
                "abstention_rate": attestation.abstention_rate,
                "abstention_mass": attestation.abstention_mass,
            }
        )

        result = RflResult(
            policy_update_applied=policy_update_applied,
            source_root=attestation.composite_root,
            abstention_mass_delta=abstention_mass_delta,
            step_id=step_id,
            ledger_entry=ledger_entry,
        )

        # Record to audit log for RFL Law compliance
        self.audit_log.record_transformation(
            attestation=attestation,
            result=result,
            config=self.config,
            resolved_slice_name=slice_cfg.name,
        )

        # Structured Experiment Logging (Schema v1)
        self.experiment_logger.log_cycle(
            cycle_index=self.first_organism_runs_total,
            mode="rfl",  # Assumes RFL mode when running with attestation
            attestation=attestation,
            result=result,
            metrics_cartographer_data=None  # Can be enhanced to pull from sidecar/redis
        )

        return result

    def _resolve_slice(self, slice_name: Optional[str]) -> CurriculumSlice:
        if slice_name:
            for slice_cfg in self.config.curriculum:
                if slice_cfg.name == slice_name:
                    return slice_cfg
        return self.config.curriculum[0]

    def _validate_attestation(self, attestation: AttestedRunContext) -> None:
        roots = [
            ("composite_root", attestation.composite_root),
            ("reasoning_root", attestation.reasoning_root),
            ("ui_root", attestation.ui_root),
        ]
        for root_name, root_value in roots:
            if len(root_value) != 64:
                raise ValueError(f"{root_name} must be 64 hex chars, got {len(root_value)}")
            int(root_value, 16)
        if attestation.abstention_rate < 0.0:
            raise ValueError("abstention_rate must be non-negative")
        if attestation.abstention_mass < 0.0:
            raise ValueError("abstention_mass must be non-negative")

    def _compute_coverage_metrics(self) -> None:
        """Compute bootstrap CI for coverage rate."""
        successful_runs = [r for r in self.run_results if r.status == "success"]

        if len(successful_runs) < 2:
            warnings.warn("ABSTAIN: Need at least 2 successful runs for coverage CI")
            self.coverage_ci = BootstrapResult(
                point_estimate=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                std_error=np.nan,
                num_replicates=0,
                method="ABSTAIN"
            )
            self.dual_attestation_records["checks"]["coverage"] = {
                "reason": "insufficient_successful_runs",
                "consistent": False,
                "primary": None,
                "secondary": None,
                "resolved_method": "ABSTAIN"
            }
            return

        # Extract coverage rates from tracker
        coverage_rates = np.array([
            m.coverage_rate for m in self.coverage_tracker.run_metrics
        ])

        self.coverage_ci = compute_coverage_ci(
            coverage_rates,
            num_replicates=self.config.bootstrap_replicates,
            confidence_level=self.config.confidence_level,
            random_state=self.config.random_seed
        )

        if self.config.dual_attestation:
            secondary = bootstrap_percentile(
                coverage_rates,
                statistic=np.mean,
                num_replicates=self.config.bootstrap_replicates,
                confidence_level=self.config.confidence_level,
                random_state=self.config.random_seed + 1
            )
            self.coverage_ci = self._dual_attest("coverage", self.coverage_ci, secondary)
        else:
            self.dual_attestation_records["checks"]["coverage"] = {
                "skipped": True,
                "primary": self.coverage_ci.to_dict(),
                "resolved_method": self.coverage_ci.method
            }

        logger.info(f"  Coverage: {self.coverage_ci.point_estimate:.4f} "
              f"[{self.coverage_ci.ci_lower:.4f}, {self.coverage_ci.ci_upper:.4f}] "
              f"({self.coverage_ci.method})")

    def _compute_uplift_metrics(self) -> None:
        """Compute bootstrap CI for uplift ratio."""
        successful_runs = [r for r in self.run_results if r.status == "success"]

        if len(successful_runs) < 2:
            warnings.warn("ABSTAIN: Need at least 2 successful runs for uplift CI")
            self.uplift_ci = BootstrapResult(
                point_estimate=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                std_error=np.nan,
                num_replicates=0,
                method="ABSTAIN"
            )
            self.dual_attestation_records["checks"]["uplift"] = {
                "reason": "insufficient_successful_runs",
                "consistent": False,
                "primary": None,
                "secondary": None,
                "resolved_method": "ABSTAIN"
            }
            return

        # For uplift, we need baseline vs treatment comparison
        # Strategy: First half as baseline, second half as treatment (RFL-influenced)
        n = len(successful_runs)
        baseline_runs = successful_runs[:n//2]
        treatment_runs = successful_runs[n//2:]

        baseline_throughput = np.array([r.throughput_proofs_per_hour for r in baseline_runs])
        treatment_throughput = np.array([r.throughput_proofs_per_hour for r in treatment_runs])

        # Pad to equal length if needed
        if len(baseline_throughput) < len(treatment_throughput):
            baseline_throughput = np.pad(
                baseline_throughput,
                (0, len(treatment_throughput) - len(baseline_throughput)),
                mode='edge'
            )
        elif len(treatment_throughput) < len(baseline_throughput):
            treatment_throughput = np.pad(
                treatment_throughput,
                (0, len(baseline_throughput) - len(treatment_throughput)),
                mode='edge'
            )

        self.uplift_ci = compute_uplift_ci(
            baseline_throughput,
            treatment_throughput,
            num_replicates=self.config.bootstrap_replicates,
            confidence_level=self.config.confidence_level,
            random_state=self.config.random_seed
        )

        if self.uplift_ci.method == "ABSTAIN":
            self.dual_attestation_records["checks"]["uplift"] = {
                "primary": self.uplift_ci.to_dict(),
                "secondary": None,
                "consistent": False,
                "reason": "primary_abstained",
                "resolved_method": "ABSTAIN"
            }
            return

        if self.config.dual_attestation:
            # Ratio per-run, excluding zero baseline entries
            safe_mask = baseline_throughput > 0
            ratio_series = treatment_throughput[safe_mask] / baseline_throughput[safe_mask]
            ratio_series = ratio_series[np.isfinite(ratio_series)]

            if len(ratio_series) >= 2:
                secondary = bootstrap_percentile(
                    ratio_series,
                    statistic=np.mean,
                    num_replicates=self.config.bootstrap_replicates,
                    confidence_level=self.config.confidence_level,
                    random_state=self.config.random_seed + 1
                )
            else:
                secondary = BootstrapResult(
                    point_estimate=np.nan,
                    ci_lower=np.nan,
                    ci_upper=np.nan,
                    std_error=np.nan,
                    num_replicates=0,
                    method="ABSTAIN"
                )

            self.uplift_ci = self._dual_attest("uplift", self.uplift_ci, secondary)
        else:
            self.dual_attestation_records["checks"]["uplift"] = {
                "skipped": True,
                "primary": self.uplift_ci.to_dict(),
                "resolved_method": self.uplift_ci.method
            }

        logger.info(f"  Uplift: {self.uplift_ci.point_estimate:.4f} "
              f"[{self.uplift_ci.ci_lower:.4f}, {self.uplift_ci.ci_upper:.4f}] "
              f"({self.uplift_ci.method})")

    def _dual_attest(
        self,
        metric_name: str,
        primary: BootstrapResult,
        secondary: BootstrapResult
    ) -> BootstrapResult:
        """Apply dual attestation guardrail to a bootstrap metric."""
        record = {
            "primary": primary.to_dict(),
            "secondary": secondary.to_dict() if secondary else None,
            "tolerance": self.config.dual_attestation_tolerance
        }

        consistent = (
            secondary is not None
            and secondary.method != "ABSTAIN"
            and abs(primary.point_estimate - secondary.point_estimate) <= self.config.dual_attestation_tolerance
            and abs(primary.ci_lower - secondary.ci_lower) <= self.config.dual_attestation_tolerance
            and abs(primary.ci_upper - secondary.ci_upper) <= self.config.dual_attestation_tolerance
        )

        if not consistent:
            record["consistent"] = False
            record["reason"] = (
                "secondary_abstained" if secondary.method == "ABSTAIN" else "tolerance_exceeded"
            )
            record["resolved_method"] = "ABSTAIN"
            self.dual_attestation_records["checks"][metric_name] = record
            warnings.warn(f"Dual attestation mismatch for {metric_name}; forcing ABSTAIN")
            return BootstrapResult(
                point_estimate=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                std_error=np.nan,
                num_replicates=0,
                method="ABSTAIN"
            )

        record["consistent"] = True
        record["resolved_method"] = primary.method
        self.dual_attestation_records["checks"][metric_name] = record
        return primary

    def _verify_metabolism(self) -> None:
        """Verify reflexive metabolism acceptance criteria."""
        if self.coverage_ci is None or self.uplift_ci is None:
            self.metabolism_passed = False
            self.metabolism_message = "[ERROR] Missing CI computations"
            return

        self.metabolism_passed, self.metabolism_message = verify_metabolism(
            self.coverage_ci,
            self.uplift_ci,
            coverage_threshold=self.config.coverage_threshold,
            uplift_threshold=self.config.uplift_threshold
        )

        successful_runs = [r for r in self.run_results if r.status == "success"]
        total_statements = sum(r.total_statements for r in successful_runs)
        total_abstentions = sum(r.abstentions for r in successful_runs)
        self.abstention_fraction = (
            total_abstentions / total_statements if total_statements > 0 else 0.0
        )

        if self.abstention_fraction > self.config.abstention_tolerance:
            self.metabolism_passed = False
            self.metabolism_message = (
                f"[FAIL] Abstention fraction {self.abstention_fraction:.4f} "
                f"exceeds tolerance {self.config.abstention_tolerance:.4f}"
            )

    def _summarize_policy_ledger(self) -> Dict[str, Any]:
        """Aggregate curriculum ledger statistics."""
        if not self.policy_ledger:
            return {
                "entries": 0,
                "mean_reward": 0.0,
                "median_reward": 0.0,
                "mean_symbolic_descent": 0.0,
                "mean_abstention_fraction": 0.0,
                "mean_coverage": 0.0,
                "curriculum_counts": {}
            }

        rewards = np.array([entry.policy_reward for entry in self.policy_ledger], dtype=float)
        symbolics = np.array([entry.symbolic_descent for entry in self.policy_ledger], dtype=float)
        abstentions = np.array([entry.abstention_fraction for entry in self.policy_ledger], dtype=float)
        coverage = np.array([entry.coverage_rate for entry in self.policy_ledger], dtype=float)
        slice_counts = Counter(entry.slice_name for entry in self.policy_ledger)

        return {
            "entries": len(self.policy_ledger),
            "mean_reward": float(np.mean(rewards)),
            "median_reward": float(np.median(rewards)),
            "mean_symbolic_descent": float(np.mean(symbolics)),
            "mean_abstention_fraction": float(np.mean(abstentions)),
            "mean_coverage": float(np.mean(coverage)),
            "curriculum_counts": dict(slice_counts)
        }

    def _export_results(self) -> Dict[str, Any]:
        """
        Export complete results to JSON files.

        Returns:
            Dictionary with all results
        """
        results = {
            "experiment_id": self.config.experiment_id,
            "config": self.config.to_dict(),
            "execution_summary": {
                "total_runs": len(self.run_results),
                "successful_runs": sum(1 for r in self.run_results if r.status == "success"),
                "failed_runs": sum(1 for r in self.run_results if r.status == "failed"),
                "aborted_runs": sum(1 for r in self.run_results if r.status == "aborted")
            },
            "runs": [r.to_dict() for r in self.run_results],
            "coverage": {
                "per_run": [m.to_dict() for m in self.coverage_tracker.run_metrics],
                "aggregate": self.coverage_tracker.get_aggregate_metrics(),
                "bootstrap_ci": self.coverage_ci.to_dict() if self.coverage_ci else None
            },
            "uplift": {
                "bootstrap_ci": self.uplift_ci.to_dict() if self.uplift_ci else None
            },
            "abstentions": {
                "histogram": dict(self.abstention_histogram),
                "fraction": float(self.abstention_fraction),
                "tolerance": float(self.config.abstention_tolerance)
            },
            "policy": {
                "ledger": [asdict(entry) for entry in self.policy_ledger],
                "summary": self._summarize_policy_ledger()
            },
            "dual_attestation": self.dual_attestation_records,
            "metabolism_verification": {
                "passed": self.metabolism_passed,
                "message": self.metabolism_message,
                "abstention_fraction": float(self.abstention_fraction),
                "abstention_tolerance": float(self.config.abstention_tolerance),
                "timestamp": deterministic_timestamp(0).isoformat() + "Z"
            }
        }

        # Export main results
        results_path = Path(self.config.artifacts_dir) / self.config.results_file
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=True)
        logger.info(f"  Results exported to {results_path}")

        # Export coverage details
        coverage_path = Path(self.config.artifacts_dir) / self.config.coverage_file
        self.coverage_tracker.export_results(str(coverage_path))
        logger.info(f"  Coverage exported to {coverage_path}")

        # Export audit log for RFL Law compliance verification
        if self.audit_log.entries:
            audit_path = Path(self.config.artifacts_dir) / "rfl_audit.json"
            self.audit_log.export(audit_path)
            logger.info(f"  Audit log exported to {audit_path}")

        # Export JSONL metrics if logger was enabled
        if self.metrics_logger:
            logger.info(f"  JSONL metrics logged to {self.metrics_logger.output_path}")

        # --- Provenance: Generate Suite Manifest ---
        # Identify key artifacts
        log_files = []
        if hasattr(self, "experiment_logger"):
            log_files.append(str(self.experiment_logger.log_file).replace("\\", "/"))
        
        figure_files = []
        # Add coverage file as a data artifact
        figure_files.append(str(coverage_path).replace("\\", "/"))
        # If curves are generated (not yet implemented in this function but config has it)
        curves_path = Path(self.config.artifacts_dir) / self.config.curves_file
        if curves_path.exists():
            figure_files.append(str(curves_path).replace("\\", "/"))

        # Start/End times (approximate for suite)
        start_time = self.run_results[0].start_time if self.run_results else datetime.utcnow().isoformat() + "Z"
        end_time = self.run_results[-1].end_time if self.run_results else datetime.utcnow().isoformat() + "Z"

        suite_manifest = self.manifest_builder.build_suite_manifest(
            config=self.config,
            execution_summary=results["execution_summary"],
            log_paths=log_files,
            figure_paths=figure_files,
            start_time=start_time,
            end_time=end_time
        )

        # Save to artifacts/experiments/rfl/{experiment_id}/manifest.json
        # This matches the structure: artifacts/experiments/rfl/fo_1000_baseline/manifest.json
        # provided artifacts_dir points to artifacts/rfl or artifacts/experiments/rfl/{id}
        
        # The config.artifacts_dir usually points to "artifacts/rfl" by default.
        # To strictly follow the prompt "artifacts/experiments/rfl/{id}/manifest.json":
        manifest_dir = Path("artifacts/experiments/rfl") / self.config.experiment_id
        manifest_path = manifest_dir / "manifest.json"
        
        self.manifest_builder.save(suite_manifest, manifest_path)
        logger.info(f"  Suite Manifest exported to {manifest_path}")
        # -------------------------------------------

        return results


def main():
    """Entry point for RFL runner CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="RFL 40-Run Experiment Suite")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (5 runs)"
    )
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = RFLConfig.from_json(args.config)
    elif args.quick:
        from .config import RFL_QUICK_CONFIG
        config = RFL_QUICK_CONFIG
        logger.info("Running QUICK mode (5 runs)")
    else:
        config = RFLConfig.from_env()

    # Run experiment suite
    runner = RFLRunner(config)
    results = runner.run_all()

    # Exit with appropriate code
    if runner.metabolism_passed:
        last_ht = "unknown"
        if runner.dual_attestation_records.get("attestations"):
             last_ht = runner.dual_attestation_records["attestations"][-1]["composite_root"]
        
        print(f"[PASS] First Organism: UI→RFL closed-loop attested (H_t={last_ht}) — organism alive.")
        return 0
    else:
        print(f"[FAIL] Metabolism verification FAILED. Message: {runner.metabolism_message}")
        return 1


if __name__ == "__main__":
    exit(main())
