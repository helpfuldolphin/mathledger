"""
RFL Metrics Logger for Wide Slice Experiments

Provides JSONL logging for RFL runs to support uplift analysis and cross-referencing
with FO cycle logs (Dyno Chart).

STATUS: Infrastructure ready for Phase II experiments.
- Automatically enabled when RFLConfig.experiment_id contains "wide_slice"
- No experimental data generated yet (results/rfl_wide_slice_runs.jsonl does not exist)
- Phase I evidence uses first-organism-pl slice (see results/fo_rfl_50.jsonl)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from .runner import RunLedgerEntry


@dataclass
class RFLRunMetrics:
    """Structured metrics for a single RFL run, suitable for JSONL logging."""
    
    run_index: int
    slice_name: str
    abstention_rate_before: float
    abstention_rate_after: float
    symbolic_descent: float
    coverage_rate: float
    novelty_rate: float
    throughput: float
    success_rate: float
    policy_reward: float
    run_id: str
    timestamp: str
    # Additional context for cross-referencing
    composite_root: Optional[str] = None
    step_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_jsonl(self) -> str:
        """Convert to JSONL line (single-line JSON)."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)


class RFLMetricsLogger:
    """
    Logger for RFL run metrics in JSONL format.
    
    Writes metrics to `results/rfl_wide_slice_runs.jsonl` (or configurable path)
    for later analysis and cross-referencing with FO cycle logs.
    """
    
    def __init__(self, output_path: str = "results/rfl_wide_slice_runs.jsonl"):
        """
        Initialize metrics logger.
        
        Args:
            output_path: Path to JSONL output file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._previous_abstention_rate: Optional[float] = None
        self._run_index: int = 0
    
    def log_run(
        self,
        ledger_entry: RunLedgerEntry,
        run_index: Optional[int] = None,
        previous_abstention_rate: Optional[float] = None
    ) -> None:
        """
        Log a single RFL run to JSONL file.
        
        Args:
            ledger_entry: RunLedgerEntry from RFLRunner.policy_ledger
            run_index: Optional run index (auto-incremented if not provided)
            previous_abstention_rate: Abstention rate from previous run (for delta computation)
        """
        if run_index is None:
            self._run_index += 1
            run_index = self._run_index
        else:
            self._run_index = run_index
        
        # Compute abstention rate delta
        abstention_before = previous_abstention_rate if previous_abstention_rate is not None else ledger_entry.abstention_fraction
        abstention_after = ledger_entry.abstention_fraction
        
        metrics = RFLRunMetrics(
            run_index=run_index,
            slice_name=ledger_entry.slice_name,
            abstention_rate_before=abstention_before,
            abstention_rate_after=abstention_after,
            symbolic_descent=ledger_entry.symbolic_descent,
            coverage_rate=ledger_entry.coverage_rate,
            novelty_rate=ledger_entry.novelty_rate,
            throughput=ledger_entry.throughput,
            success_rate=ledger_entry.success_rate,
            policy_reward=ledger_entry.policy_reward,
            run_id=ledger_entry.run_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            composite_root=ledger_entry.composite_root,
        )
        
        # Append to JSONL file
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(metrics.to_jsonl() + '\n')
        
        # Update previous abstention rate for next run
        self._previous_abstention_rate = abstention_after
    
    def log_batch(self, ledger_entries: list[RunLedgerEntry], start_index: int = 1) -> None:
        """
        Log multiple runs in batch.
        
        Args:
            ledger_entries: List of RunLedgerEntry objects
            start_index: Starting run index
        """
        previous_rate: Optional[float] = None
        for i, entry in enumerate(ledger_entries):
            self.log_run(entry, run_index=start_index + i, previous_abstention_rate=previous_rate)
            previous_rate = entry.abstention_fraction

