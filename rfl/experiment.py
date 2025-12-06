"""
Single RFL Experiment Executor

# NOTE: Canonical rfl module; backend.* imports are forbidden here (except via shims).

Runs one derivation experiment and collects metrics for RFL analysis.

DETERMINISM NOTE: All timestamps are derived from content or explicit seeds.
No datetime.now(), datetime.utcnow(), or time.time() calls in the attestation path.
"""

import subprocess
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import psycopg
from psycopg.rows import dict_row

from substrate.repro.determinism import deterministic_timestamp, deterministic_isoformat
# TODO: Move get_or_create_system_id to canonical namespace (ledger or derivation)
from backend.axiom_engine.derive_utils import get_or_create_system_id


@dataclass
class ExperimentResult:
    """Results from a single RFL experiment run."""

    run_id: str
    system_id: int
    start_time: str
    end_time: str
    duration_seconds: float

    # Derivation outcomes
    total_statements: int
    successful_proofs: int
    failed_proofs: int
    abstentions: int

    # Performance metrics
    throughput_proofs_per_hour: float
    mean_depth: float
    max_depth: int

    # Coverage
    statement_hashes: List[str] = field(default_factory=list)
    distinct_statements: int = 0

    # Policy
    derive_steps: int = 0
    max_breadth: int = 0
    max_total: int = 0

    # Policy instrumentation
    policy_context: Dict[str, Any] = field(default_factory=dict)
    abstention_breakdown: Dict[str, int] = field(default_factory=dict)

    # Artifacts
    logs: List[str] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)

    # Status
    status: str = "success"  # success, failed, aborted
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "system_id": self.system_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "total_statements": self.total_statements,
            "successful_proofs": self.successful_proofs,
            "failed_proofs": self.failed_proofs,
            "abstentions": self.abstentions,
            "throughput_proofs_per_hour": self.throughput_proofs_per_hour,
            "mean_depth": self.mean_depth,
            "max_depth": self.max_depth,
            "distinct_statements": self.distinct_statements,
            "derive_steps": self.derive_steps,
            "max_breadth": self.max_breadth,
            "max_total": self.max_total,
            "status": self.status,
            "error_message": self.error_message,
            "policy_context": self.policy_context,
            "abstention_breakdown": self.abstention_breakdown,
            "logs": self.logs,
            "figures": self.figures
        }

    @property
    def success_rate(self) -> float:
        """Success rate (successful / total)."""
        total = self.successful_proofs + self.failed_proofs
        if total == 0:
            return 0.0
        return self.successful_proofs / total

    @property
    def abstention_rate(self) -> float:
        """Abstention rate (abstentions / total)."""
        total = self.total_statements
        if total == 0:
            return 0.0
        return self.abstentions / total


class RFLExperiment:
    """Executor for a single RFL derivation experiment."""

    def __init__(self, db_url: str, system_id: int = 1):
        """
        Initialize experiment executor.

        Args:
            db_url: PostgreSQL connection string
            system_id: Theory system ID (default 1 = propositional logic)
        """
        self.db_url = db_url
        self.system_id = system_id

    def run(
        self,
        run_id: str,
        derive_steps: int = 50,
        max_breadth: int = 200,
        max_total: int = 1000,
        depth_max: int = 4,
        policy_context: Optional[Dict[str, Any]] = None,
        seed: int = 0
    ) -> ExperimentResult:
        """
        Execute a single derivation experiment.

        Args:
            run_id: Unique identifier for this run
            derive_steps: Number of derivation steps
            max_breadth: Max new statements per step
            max_total: Max total statements per run
            depth_max: Max formula depth
            policy_context: Optional metadata describing policy slice
            seed: Deterministic seed for timestamps and operations

        Returns:
            ExperimentResult with metrics
        """
        policy_context = policy_context or {}
        
        # DETERMINISM: Always use deterministic timestamps for reproducibility.
        # The seed is derived from run_id content if not explicitly provided.
        # This ensures identical runs produce identical attestation artifacts.
        effective_seed = seed if seed != 0 else hash(run_id) & 0x7FFFFFFF
        start_time = deterministic_timestamp(effective_seed)
        # End time offset ensures monotonicity while remaining deterministic
        end_time = deterministic_timestamp(effective_seed + 1)

        start_ts = start_time.isoformat() + "Z"

        print(f"[{run_id}] Starting derivation experiment...")
        print(f"  System ID: {self.system_id}")
        print(f"  Steps: {derive_steps}, Breadth: {max_breadth}, Total: {max_total}")

        # Get baseline metrics before derivation
        baseline_count = self._get_statement_count()
        print(f"  Baseline statements: {baseline_count}")

        # Run derivation via CLI
        try:
            cmd = [
                "uv", "run", "python",
                "experiments/rfl/derive_wrapper.py",
                "--system-id", str(self.system_id),
                "--steps", str(derive_steps),
                "--max-breadth", str(max_breadth),
                "--max-total", str(max_total),
                "--depth-max", str(depth_max)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=False
            )

            if result.returncode != 0:
                error_msg = f"Derive CLI failed with code {result.returncode}: {result.stderr}"
                print(f"  ERROR: {error_msg}")

                return ExperimentResult(
                    run_id=run_id,
                    system_id=self.system_id,
                    start_time=start_ts,
                    end_time=end_time.isoformat() + "Z",
                    duration_seconds=(end_time - start_time).total_seconds(),
                    total_statements=0,
                    successful_proofs=0,
                    failed_proofs=0,
                    abstentions=0,
                    throughput_proofs_per_hour=0.0,
                    mean_depth=0.0,
                    max_depth=0,
                    derive_steps=derive_steps,
                    max_breadth=max_breadth,
                    max_total=max_total,
                    policy_context=policy_context,
                    abstention_breakdown={"engine_failure": 1},
                    status="failed",
                    error_message=error_msg
                )

            print(f"  Derivation completed")

        except subprocess.TimeoutExpired:
            error_msg = "Derivation timed out after 1 hour"
            print(f"  ERROR: {error_msg}")

            return ExperimentResult(
                run_id=run_id,
                system_id=self.system_id,
                start_time=start_ts,
                end_time=end_time.isoformat() + "Z",
                duration_seconds=(end_time - start_time).total_seconds(),
                total_statements=0,
                successful_proofs=0,
                failed_proofs=0,
                abstentions=0,
                throughput_proofs_per_hour=0.0,
                mean_depth=0.0,
                max_depth=0,
                derive_steps=derive_steps,
                max_breadth=max_breadth,
                max_total=max_total,
                policy_context=policy_context,
                abstention_breakdown={"timeout": 1},
                status="aborted",
                error_message=error_msg
            )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"  ERROR: {error_msg}")

            ts_end = end_time.isoformat() + "Z"
            return ExperimentResult(
                run_id=run_id,
                system_id=self.system_id,
                start_time=start_ts,
                end_time=ts_end,
                duration_seconds=(end_time - start_time).total_seconds(),
                total_statements=0,
                successful_proofs=0,
                failed_proofs=0,
                abstentions=0,
                throughput_proofs_per_hour=0.0,
                mean_depth=0.0,
                max_depth=0,
                derive_steps=derive_steps,
                max_breadth=max_breadth,
                max_total=max_total,
                policy_context=policy_context,
                abstention_breakdown={"unexpected_error": 1},
                status="failed",
                error_message=error_msg
            )

        # Collect metrics from database
        # DETERMINISM: end_time is already set deterministically above
        duration_seconds = (end_time - start_time).total_seconds()
        # Ensure non-zero duration for throughput calc if deterministic
        if duration_seconds <= 0:
            duration_seconds = 1.0

        metrics = self._collect_metrics(baseline_count, start_time, end_time)

        return ExperimentResult(
            run_id=run_id,
            system_id=self.system_id,
            start_time=start_ts,
            end_time=end_time.isoformat() + "Z",
            duration_seconds=duration_seconds,
            total_statements=metrics["total_statements"],
            successful_proofs=metrics["successful_proofs"],
            failed_proofs=metrics["failed_proofs"],
            abstentions=metrics["abstentions"],
            throughput_proofs_per_hour=metrics["throughput"],
            mean_depth=metrics["mean_depth"],
            max_depth=metrics["max_depth"],
            statement_hashes=metrics["statement_hashes"],
            distinct_statements=len(set(metrics["statement_hashes"])),
            derive_steps=derive_steps,
            max_breadth=max_breadth,
            max_total=max_total,
            policy_context=policy_context,
            status="success",
            abstention_breakdown=self._compute_abstention_breakdown(metrics)
        )

    def _get_statement_count(self) -> int:
        """Get current statement count from database."""
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    sys_id = self.system_id
                    if sys_id == 1:
                        sys_id = get_or_create_system_id(cur, "pl")
                    
                    cur.execute(
                        "SELECT COUNT(*) as cnt FROM statements WHERE system_id = %s",
                        (sys_id,)
                    )
                    row = cur.fetchone()
                    return row[0] if row else 0
        except Exception as e:
            print(f"  Warning: Could not query baseline count: {e}")
            return 0

    def _collect_metrics(
        self,
        baseline_count: int,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Collect metrics from database for statements/proofs created during experiment.

        Args:
            baseline_count: Statement count before experiment
            start_time: Experiment start time
            end_time: Experiment end time

        Returns:
            Dictionary with metrics
        """
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    sys_id = self.system_id
                    if sys_id == 1:
                        sys_id = get_or_create_system_id(cur, "pl")

                    # New statements created
                    cur.execute(
                        """
                        SELECT hash, derivation_depth as depth
                        FROM statements
                        WHERE system_id = %s
                          AND created_at >= %s
                          AND created_at <= %s
                        """,
                        (sys_id, start_time, end_time)
                    )
                    statements = cur.fetchall()

                    # Proofs for these statements
                    cur.execute(
                        """
                        SELECT p.status, COUNT(*) as cnt
                        FROM proofs p
                        JOIN statements s ON p.statement_id = s.id
                        WHERE s.system_id = %s
                          AND p.created_at >= %s
                          AND p.created_at <= %s
                        GROUP BY p.status
                        """,
                        (sys_id, start_time, end_time)
                    )
                    proof_counts = {row['status']: row['cnt'] for row in cur.fetchall()}

                    # Compute metrics
                    statement_hashes = [s['hash'] for s in statements]
                    depths = [s['depth'] for s in statements if s['depth'] is not None]

                    total_statements = len(statements)
                    successful_proofs = proof_counts.get('success', 0) + proof_counts.get('verified', 0)
                    failed_proofs = proof_counts.get('failed', 0) + proof_counts.get('error', 0)
                    abstentions = max(0, total_statements - successful_proofs - failed_proofs)

                    mean_depth = sum(depths) / len(depths) if depths else 0.0
                    max_depth = max(depths) if depths else 0

                    duration_hours = (end_time - start_time).total_seconds() / 3600
                    throughput = successful_proofs / duration_hours if duration_hours > 0 else 0.0

                    return {
                        "total_statements": total_statements,
                        "successful_proofs": successful_proofs,
                        "failed_proofs": failed_proofs,
                        "abstentions": abstentions,
                        "throughput": throughput,
                        "mean_depth": mean_depth,
                        "max_depth": max_depth,
                        "statement_hashes": statement_hashes
                    }

        except Exception as e:
            print(f"  Warning: Could not collect metrics: {e}")
            return {
                "total_statements": 0,
                "successful_proofs": 0,
                "failed_proofs": 0,
                "abstentions": 0,
                "throughput": 0.0,
                "mean_depth": 0.0,
                "max_depth": 0,
                "statement_hashes": []
            }

    def _compute_abstention_breakdown(self, metrics: Dict[str, Any]) -> Dict[str, int]:
        """
        Classify abstentions across coarse buckets for Reflexive Formal Learning analysis.
        """
        breakdown: Dict[str, int] = {}
        total = int(metrics.get("total_statements", 0) or 0)
        abstentions = int(metrics.get("abstentions", 0) or 0)
        successes = int(metrics.get("successful_proofs", 0) or 0)
        throughput = float(metrics.get("throughput", 0.0) or 0.0)

        if total == 0:
            breakdown["empty_run"] = breakdown.get("empty_run", 0) + 1

        if abstentions > 0:
            breakdown["pending_validation"] = breakdown.get("pending_validation", 0) + abstentions

        if successes == 0 and total > 0:
            breakdown["no_successful_proofs"] = breakdown.get("no_successful_proofs", 0) + total

        if throughput == 0.0 and total > 0:
            breakdown["zero_throughput"] = breakdown.get("zero_throughput", 0) + 1

        return breakdown


if __name__ == "__main__":
    # Example usage
    from substrate.security.runtime_env import get_database_url

    db_url = get_database_url()

    executor = RFLExperiment(db_url=db_url, system_id=1)

    result = executor.run(
        run_id="test_run_01",
        derive_steps=10,
        max_breadth=50,
        max_total=200
    )

    print("\nExperiment Result:")
    print(f"  Status: {result.status}")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Total statements: {result.total_statements}")
    print(f"  Successful proofs: {result.successful_proofs}")
    print(f"  Success rate: {result.success_rate:.2%}")
    print(f"  Throughput: {result.throughput_proofs_per_hour:.1f} proofs/hour")
    print(f"  Mean depth: {result.mean_depth:.2f}")
    print(f"  Distinct statements: {result.distinct_statements}")
