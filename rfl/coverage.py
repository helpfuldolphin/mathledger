"""
Coverage Tracker for Reflexive Formal Learning

# NOTE: Canonical rfl module; backend.* imports are forbidden here (except via shims).

Measures statement novelty and coverage rates:
- Distinct statements generated per run
- Coverage as proportion of target statement space
- Novelty relative to existing corpus
- Cross-run coverage accumulation
"""

import hashlib
from typing import Set, List, Dict, Optional
from dataclasses import dataclass, field
import json
# TODO: Move get_or_create_system_id to canonical namespace (ledger or derivation)
from backend.axiom_engine.derive_utils import get_or_create_system_id


@dataclass
class CoverageMetrics:
    """Coverage metrics for a single run or aggregated runs."""
    total_statements: int
    distinct_statements: int
    novel_statements: int
    coverage_rate: float
    novelty_rate: float
    run_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "total_statements": self.total_statements,
            "distinct_statements": self.distinct_statements,
            "novel_statements": self.novel_statements,
            "coverage_rate": float(self.coverage_rate),
            "novelty_rate": float(self.novelty_rate)
        }
        if self.run_id:
            result["run_id"] = self.run_id
        return result


class CoverageTracker:
    """
    Tracks statement coverage across RFL experiments.

    Coverage Definition:
        coverage_rate = distinct_statements / target_statement_space

    Novelty Definition:
        novelty_rate = novel_statements / total_statements
        (novel = not in baseline corpus)
    """

    def __init__(self, baseline_statements: Optional[Set[str]] = None):
        """
        Initialize coverage tracker.

        Args:
            baseline_statements: Set of statement hashes already in corpus
                                (from database or previous runs)
        """
        self.baseline_statements = baseline_statements or set()
        self.accumulated_statements: Set[str] = set(self.baseline_statements)
        self.run_metrics: List[CoverageMetrics] = []

    def record_run(
        self,
        statement_hashes: List[str],
        run_id: Optional[str] = None,
        target_space_size: Optional[int] = None
    ) -> CoverageMetrics:
        """
        Record coverage metrics for a single run.

        Args:
            statement_hashes: List of statement hashes generated in this run
            run_id: Identifier for this run (e.g., "run_01")
            target_space_size: Size of target statement space for coverage calculation
                             If None, uses distinct_statements as denominator

        Returns:
            CoverageMetrics for this run
        """
        total_statements = len(statement_hashes)

        # Distinct statements in this run
        distinct_set = set(statement_hashes)
        distinct_statements = len(distinct_set)

        # Novel statements (not in accumulated corpus from previous runs)
        # This includes both the baseline and statements from prior runs in this experiment
        novel_set = distinct_set - self.accumulated_statements
        novel_statements = len(novel_set)

        # Coverage rate
        if target_space_size is not None and target_space_size > 0:
            # Coverage relative to target space
            coverage_rate = distinct_statements / target_space_size
        else:
            # Perfect coverage of what we generated
            coverage_rate = 1.0 if distinct_statements > 0 else 0.0

        # Novelty rate
        if total_statements > 0:
            novelty_rate = novel_statements / total_statements
        else:
            novelty_rate = 0.0

        metrics = CoverageMetrics(
            total_statements=total_statements,
            distinct_statements=distinct_statements,
            novel_statements=novel_statements,
            coverage_rate=coverage_rate,
            novelty_rate=novelty_rate,
            run_id=run_id
        )

        # Update accumulator
        self.accumulated_statements.update(distinct_set)
        self.run_metrics.append(metrics)

        return metrics

    def get_cumulative_coverage(self) -> float:
        """
        Get cumulative coverage rate across all runs.

        Returns:
            Proportion of target space covered by all runs combined
        """
        total_accumulated = len(self.accumulated_statements) - len(self.baseline_statements)
        if len(self.run_metrics) == 0:
            return 0.0

        # Average target space size from runs
        avg_target = sum(
            m.distinct_statements / m.coverage_rate
            for m in self.run_metrics
            if m.coverage_rate > 0
        )
        if avg_target == 0:
            return 1.0

        avg_target /= len([m for m in self.run_metrics if m.coverage_rate > 0])
        return min(total_accumulated / avg_target, 1.0)

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        Get aggregate statistics across all recorded runs.

        Returns:
            Dictionary with mean, std, min, max for coverage and novelty rates
        """
        if not self.run_metrics:
            return {
                "coverage_mean": 0.0,
                "coverage_std": 0.0,
                "coverage_min": 0.0,
                "coverage_max": 0.0,
                "novelty_mean": 0.0,
                "novelty_std": 0.0,
                "novelty_min": 0.0,
                "novelty_max": 0.0
            }

        import numpy as np

        coverage_rates = [m.coverage_rate for m in self.run_metrics]
        novelty_rates = [m.novelty_rate for m in self.run_metrics]

        return {
            "coverage_mean": float(np.mean(coverage_rates)),
            "coverage_std": float(np.std(coverage_rates, ddof=1)) if len(coverage_rates) > 1 else 0.0,
            "coverage_min": float(np.min(coverage_rates)),
            "coverage_max": float(np.max(coverage_rates)),
            "novelty_mean": float(np.mean(novelty_rates)),
            "novelty_std": float(np.std(novelty_rates, ddof=1)) if len(novelty_rates) > 1 else 0.0,
            "novelty_min": float(np.min(novelty_rates)),
            "novelty_max": float(np.max(novelty_rates)),
            "cumulative_coverage": float(self.get_cumulative_coverage()),
            "num_runs": len(self.run_metrics)
        }

    def export_results(self, filepath: str) -> None:
        """
        Export coverage results to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        data = {
            "baseline_count": len(self.baseline_statements),
            "accumulated_count": len(self.accumulated_statements),
            "runs": [m.to_dict() for m in self.run_metrics],
            "aggregate": self.get_aggregate_metrics()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=True)


def compute_statement_hash(text: str) -> str:
    """
    Compute canonical hash for a statement.

    Should match backend.logic.canon.compute_hash() for consistency.

    Args:
        text: Statement text (normalized)

    Returns:
        64-character hex SHA-256 hash
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_baseline_from_db(db_url: str, system_id: int = 1) -> Set[str]:
    """
    Load baseline statement hashes from database.

    Args:
        db_url: PostgreSQL connection string
        system_id: Theory system ID (default 1 = propositional logic)

    Returns:
        Set of statement hashes
    """
    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                sys_id = system_id
                if sys_id == 1:
                    sys_id = get_or_create_system_id(cur, "pl")

                cur.execute(
                    "SELECT hash FROM statements WHERE system_id = %s",
                    (sys_id,)
                )
                rows = cur.fetchall()
                return {row['hash'] for row in rows}

    except Exception as e:
        import warnings
        warnings.warn(f"Failed to load baseline from DB: {e}")
        return set()


if __name__ == "__main__":
    # Example usage
    import numpy as np

    np.random.seed(42)

    # Simulate 40 runs
    tracker = CoverageTracker()

    # Target space: 10000 possible statements
    target_space = 10000

    print("Simulating 40 RFL runs...")
    for i in range(40):
        run_id = f"run_{i+1:02d}"

        # Simulate statement generation (Poisson with increasing mean)
        num_statements = np.random.poisson(200 + i * 5)

        # Generate random statement hashes
        hashes = [
            hashlib.sha256(f"stmt_{i}_{j}".encode()).hexdigest()
            for j in range(num_statements)
        ]

        # Add some duplicates within run
        hashes.extend(hashes[:num_statements // 10])

        # Record metrics
        metrics = tracker.record_run(
            hashes,
            run_id=run_id,
            target_space_size=target_space
        )

        if (i + 1) % 10 == 0:
            print(f"  Run {i+1}: coverage={metrics.coverage_rate:.4f}, novelty={metrics.novelty_rate:.4f}")

    # Aggregate results
    agg = tracker.get_aggregate_metrics()
    print("\nAggregate Metrics:")
    print(f"  Coverage: {agg['coverage_mean']:.4f} ± {agg['coverage_std']:.4f}")
    print(f"  Novelty:  {agg['novelty_mean']:.4f} ± {agg['novelty_std']:.4f}")
    print(f"  Cumulative coverage: {agg['cumulative_coverage']:.4f}")
    print(f"  Total unique statements: {len(tracker.accumulated_statements)}")
