#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
Budget Usage Summarizer
═══════════════════════════════════════════════════════════════════════════════

Summarize budget enforcement statistics from Phase II experiment JSONL logs.
Read-only analysis tool for post-hoc inspection.

Usage:
    # Summarize single log file
    uv run python experiments/summarize_budget_usage.py results/u2_test/uplift_u2_slice_uplift_goal_rfl.jsonl
    
    # Multiple log files
    uv run python experiments/summarize_budget_usage.py results/*.jsonl
    
    # JSON output
    uv run python experiments/summarize_budget_usage.py results/log.jsonl --json
    
    # With interpretation guidance
    uv run python experiments/summarize_budget_usage.py results/log.jsonl --explain

Output:
    slice: slice_uplift_goal
    cycles: 500
    budget_exhausted: 12 (2.4%)
    max_candidates_hit: 5 (1.0%)
    timeout_abstentions: 23 (avg 0.05/cycle)

═══════════════════════════════════════════════════════════════════════════════
BUDGET STATE DEFINITIONS (Quantitative)
═══════════════════════════════════════════════════════════════════════════════

budget_exhausted (bool per cycle):
    True when: time.monotonic() - cycle_start > cycle_budget_s
    
    Quantitative meaning:
        - The derivation cycle ran out of wall-clock time
        - Remaining candidates were NOT verified (skipped with "budget_skip")
        - These cycles are INCOMPLETE — the pipeline terminated early
        - High rate (>5%) suggests cycle_budget_s is too tight for the slice
    
    Formula: budget_exhausted_pct = 100 * (cycles where budget_exhausted=True) / total_cycles

max_candidates_hit (bool per cycle):
    True when: candidates_considered >= max_candidates_per_cycle
    
    Quantitative meaning:
        - The cycle processed exactly max_candidates_per_cycle candidates
        - Additional candidates existed but were NOT considered
        - This is EXPECTED behavior when ordering matters (RFL uplift)
        - High rate (>90%) indicates the budget is actively constraining exploration
        - Low rate (<10%) suggests slice generates few candidates naturally
    
    Formula: max_candidates_hit_pct = 100 * (cycles where max_candidates_hit=True) / total_cycles

timeout_abstentions (int per cycle):
    Count of candidates where: verification_time > taut_timeout_s
    
    Quantitative meaning:
        - Per-candidate truth-table verification timed out
        - These candidates received "abstain_timeout" outcome
        - They are EXCLUDED from success/failure counts
        - High avg (>1.0/cycle) suggests taut_timeout_s is too tight
        - Or: formula complexity exceeds truth-table feasibility
    
    Formula: avg_timeout_abstentions = sum(timeout_abstentions) / total_cycles

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# BUDGET HEALTH CLASSIFICATION (Configurable Thresholds)
# =============================================================================
# Status names are neutral (SAFE/TIGHT/STARVED), not prescriptive (GOOD/BAD).
# These thresholds are configurable constants — adjust based on slice characteristics.

class BudgetHealthStatus(Enum):
    """
    Budget health classification status.
    
    Neutral naming convention:
        SAFE    - Budget parameters are sufficient
        TIGHT   - Budget parameters are marginal (advisory)
        STARVED - Budget parameters are insufficient (attention needed)
        INVALID - Insufficient data to classify
    """
    SAFE = "SAFE"
    TIGHT = "TIGHT"
    STARVED = "STARVED"
    INVALID = "INVALID"


# Configurable thresholds for health classification
# These can be overridden via environment variables or config files

# budget_exhausted_pct thresholds
THRESHOLD_EXHAUSTED_SAFE = 1.0      # <1% = SAFE
THRESHOLD_EXHAUSTED_TIGHT = 5.0    # 1-5% = TIGHT, >5% = STARVED

# timeout_abstentions_avg thresholds
THRESHOLD_TIMEOUT_SAFE = 0.1       # <0.1 avg/cycle = SAFE
THRESHOLD_TIMEOUT_TIGHT = 1.0      # 0.1-1.0 = TIGHT, >1.0 = STARVED

# max_candidates_hit_pct is informational, not a health signal
# (high hit rate is expected and healthy for RFL experiments)


@dataclass
class BudgetSummary:
    """Aggregated budget statistics from a log file."""
    path: str = ""
    slice_name: str = ""
    mode: str = ""
    total_cycles: int = 0
    budget_exhausted_count: int = 0
    max_candidates_hit_count: int = 0
    timeout_abstentions_total: int = 0
    cycles_with_budget_field: int = 0
    
    @property
    def budget_exhausted_pct(self) -> float:
        """Percentage of cycles with budget exhaustion."""
        if self.total_cycles == 0:
            return 0.0
        return 100.0 * self.budget_exhausted_count / self.total_cycles
    
    @property
    def max_candidates_hit_pct(self) -> float:
        """Percentage of cycles hitting max candidates."""
        if self.total_cycles == 0:
            return 0.0
        return 100.0 * self.max_candidates_hit_count / self.total_cycles
    
    @property
    def avg_timeout_abstentions(self) -> float:
        """Average timeout abstentions per cycle."""
        if self.total_cycles == 0:
            return 0.0
        return self.timeout_abstentions_total / self.total_cycles
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "slice": self.slice_name,
            "mode": self.mode,
            "total_cycles": self.total_cycles,
            "budget": {
                "exhausted_count": self.budget_exhausted_count,
                "exhausted_pct": round(self.budget_exhausted_pct, 2),
                "max_candidates_hit_count": self.max_candidates_hit_count,
                "max_candidates_hit_pct": round(self.max_candidates_hit_pct, 2),
                "timeout_abstentions_total": self.timeout_abstentions_total,
                "timeout_abstentions_avg": round(self.avg_timeout_abstentions, 3),
            },
            "cycles_with_budget_field": self.cycles_with_budget_field,
        }


@dataclass
class BudgetHealthResult:
    """
    Result of budget health classification.
    
    Attributes:
        status: Overall health status (SAFE/TIGHT/STARVED/INVALID)
        reasons: List of human-readable reasons for the classification
        metrics: Dictionary of metric values used in classification
    """
    status: BudgetHealthStatus
    reasons: List[str]
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


def classify_budget_health(summary: BudgetSummary) -> BudgetHealthResult:
    """
    Classify budget health based on observed metrics.
    
    Classification rules (evaluated in order of severity):
    
    INVALID:
        - total_cycles == 0 (no data)
        - cycles_with_budget_field == 0 (no budget telemetry)
    
    STARVED (attention needed):
        - budget_exhausted_pct > 5% (cycles running out of time)
        - timeout_abstentions_avg > 1.0 (many timeouts per cycle)
    
    TIGHT (advisory):
        - budget_exhausted_pct in [1%, 5%] (marginal time budget)
        - timeout_abstentions_avg in [0.1, 1.0] (occasional timeouts)
    
    SAFE (sufficient):
        - budget_exhausted_pct < 1%
        - timeout_abstentions_avg < 0.1
    
    Args:
        summary: BudgetSummary with aggregated statistics
        
    Returns:
        BudgetHealthResult with status, reasons, and metrics
    """
    reasons: List[str] = []
    metrics: Dict[str, float] = {
        "budget_exhausted_pct": summary.budget_exhausted_pct,
        "max_candidates_hit_pct": summary.max_candidates_hit_pct,
        "timeout_abstentions_avg": summary.avg_timeout_abstentions,
        "total_cycles": float(summary.total_cycles),
    }
    
    # Check for INVALID conditions
    if summary.total_cycles == 0:
        return BudgetHealthResult(
            status=BudgetHealthStatus.INVALID,
            reasons=["No cycles in log (total_cycles=0)"],
            metrics=metrics,
        )
    
    if summary.cycles_with_budget_field == 0:
        return BudgetHealthResult(
            status=BudgetHealthStatus.INVALID,
            reasons=["No budget telemetry found in log (cycles_with_budget_field=0)"],
            metrics=metrics,
        )
    
    # Evaluate budget_exhausted_pct
    exhausted_pct = summary.budget_exhausted_pct
    if exhausted_pct > THRESHOLD_EXHAUSTED_TIGHT:
        reasons.append(
            f"budget_exhausted={exhausted_pct:.1f}% exceeds {THRESHOLD_EXHAUSTED_TIGHT}% (STARVED)"
        )
    elif exhausted_pct >= THRESHOLD_EXHAUSTED_SAFE:
        reasons.append(
            f"budget_exhausted={exhausted_pct:.1f}% in [{THRESHOLD_EXHAUSTED_SAFE}%, {THRESHOLD_EXHAUSTED_TIGHT}%] (TIGHT)"
        )
    else:
        reasons.append(
            f"budget_exhausted={exhausted_pct:.1f}% below {THRESHOLD_EXHAUSTED_SAFE}% (SAFE)"
        )
    
    # Evaluate timeout_abstentions_avg
    timeout_avg = summary.avg_timeout_abstentions
    if timeout_avg > THRESHOLD_TIMEOUT_TIGHT:
        reasons.append(
            f"timeout_abstentions_avg={timeout_avg:.2f} exceeds {THRESHOLD_TIMEOUT_TIGHT} (STARVED)"
        )
    elif timeout_avg >= THRESHOLD_TIMEOUT_SAFE:
        reasons.append(
            f"timeout_abstentions_avg={timeout_avg:.2f} in [{THRESHOLD_TIMEOUT_SAFE}, {THRESHOLD_TIMEOUT_TIGHT}] (TIGHT)"
        )
    else:
        reasons.append(
            f"timeout_abstentions_avg={timeout_avg:.2f} below {THRESHOLD_TIMEOUT_SAFE} (SAFE)"
        )
    
    # Note max_candidates_hit (informational only)
    maxcand_pct = summary.max_candidates_hit_pct
    if maxcand_pct > 90:
        reasons.append(f"max_candidates_hit={maxcand_pct:.1f}% (expected for RFL)")
    elif maxcand_pct > 50:
        reasons.append(f"max_candidates_hit={maxcand_pct:.1f}% (mixed completion)")
    else:
        reasons.append(f"max_candidates_hit={maxcand_pct:.1f}% (slice naturally small)")
    
    # Determine overall status (worst case wins)
    if exhausted_pct > THRESHOLD_EXHAUSTED_TIGHT or timeout_avg > THRESHOLD_TIMEOUT_TIGHT:
        status = BudgetHealthStatus.STARVED
    elif exhausted_pct >= THRESHOLD_EXHAUSTED_SAFE or timeout_avg >= THRESHOLD_TIMEOUT_SAFE:
        status = BudgetHealthStatus.TIGHT
    else:
        status = BudgetHealthStatus.SAFE
    
    return BudgetHealthResult(status=status, reasons=reasons, metrics=metrics)


def parse_log_file(path: Path) -> BudgetSummary:
    """
    Parse a JSONL log file and extract budget statistics.
    
    Supports two formats:
    1. Main results JSONL (cycle records with optional 'budget' field)
    2. Trace JSONL (event records with BudgetConsumptionEvent type)
    
    Args:
        path: Path to the JSONL file.
        
    Returns:
        BudgetSummary with aggregated statistics.
    """
    summary = BudgetSummary(path=str(path))
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
                continue
            
            # Handle trace JSONL format (event_type wrapper)
            if "event_type" in record:
                event_type = record.get("event_type", "")
                payload = record.get("payload", {})
                
                # Extract slice/mode from any event
                if "slice_name" in payload:
                    if not summary.slice_name:
                        summary.slice_name = payload["slice_name"]
                if "mode" in payload:
                    if not summary.mode:
                        summary.mode = payload["mode"]
                
                # Handle BudgetConsumptionEvent
                if event_type == "BudgetConsumptionEvent":
                    summary.total_cycles += 1
                    summary.cycles_with_budget_field += 1
                    
                    if payload.get("budget_exhausted", False):
                        summary.budget_exhausted_count += 1
                    
                    # Note: BudgetConsumptionEvent doesn't have max_candidates_hit or timeout_abstentions
                    # Those would be in a separate event or in the main results
                
                # Handle CycleTelemetryEvent (contains raw_record with budget)
                elif event_type == "CycleTelemetryEvent":
                    raw = payload.get("raw_record", {})
                    budget_data = raw.get("budget", {})
                    
                    if budget_data:
                        summary.cycles_with_budget_field += 1
                        
                        if budget_data.get("exhausted", False):
                            summary.budget_exhausted_count += 1
                        if budget_data.get("max_candidates_hit", False):
                            summary.max_candidates_hit_count += 1
                        summary.timeout_abstentions_total += budget_data.get("timeout_abstentions", 0)
            
            # Handle main results JSONL format (direct cycle records)
            elif "cycle" in record:
                summary.total_cycles += 1
                
                # Extract slice/mode
                if "slice" in record:
                    if not summary.slice_name:
                        summary.slice_name = record["slice"]
                if "mode" in record:
                    if not summary.mode:
                        summary.mode = record["mode"]
                
                # Handle budget field (nested or flat)
                budget_data = record.get("budget", {})
                if budget_data:
                    summary.cycles_with_budget_field += 1
                    
                    if budget_data.get("exhausted", False):
                        summary.budget_exhausted_count += 1
                    if budget_data.get("max_candidates_hit", False):
                        summary.max_candidates_hit_count += 1
                    summary.timeout_abstentions_total += budget_data.get("timeout_abstentions", 0)
                
                # Also check flat fields (older format)
                if record.get("budget_exhausted", False):
                    if not budget_data:  # Don't double-count
                        summary.budget_exhausted_count += 1
                        summary.cycles_with_budget_field += 1
                if record.get("max_candidates_hit", False):
                    if not budget_data:
                        summary.max_candidates_hit_count += 1
    
    return summary


def format_summary(summary: BudgetSummary) -> str:
    """Format a single summary for human-readable output."""
    lines = []
    
    lines.append(f"slice: {summary.slice_name or '(unknown)'}")
    lines.append(f"mode: {summary.mode or '(unknown)'}")
    lines.append(f"cycles: {summary.total_cycles}")
    
    if summary.cycles_with_budget_field == 0:
        lines.append("budget: (no budget fields found in log)")
    else:
        lines.append(f"budget_exhausted: {summary.budget_exhausted_count} ({summary.budget_exhausted_pct:.1f}%)")
        lines.append(f"max_candidates_hit: {summary.max_candidates_hit_count} ({summary.max_candidates_hit_pct:.1f}%)")
        lines.append(f"timeout_abstentions: {summary.timeout_abstentions_total} (avg {summary.avg_timeout_abstentions:.2f}/cycle)")
    
    return "\n".join(lines)


def format_multi_summary(summaries: List[BudgetSummary]) -> str:
    """Format multiple summaries as a comparison table."""
    lines = []
    
    # Header
    header = (
        f"{'File':<40} {'Slice':<25} {'Mode':<10} "
        f"{'Cycles':>8} {'Exhausted':>12} {'MaxCand':>12} {'Timeouts':>12}"
    )
    lines.append(header)
    lines.append("─" * len(header))
    
    # Rows
    for s in summaries:
        filename = Path(s.path).name[:38]
        slice_name = (s.slice_name or "(unknown)")[:23]
        mode = (s.mode or "?")[:8]
        
        exhausted_str = f"{s.budget_exhausted_count} ({s.budget_exhausted_pct:.1f}%)"
        maxcand_str = f"{s.max_candidates_hit_count} ({s.max_candidates_hit_pct:.1f}%)"
        timeout_str = f"{s.timeout_abstentions_total}"
        
        row = (
            f"{filename:<40} {slice_name:<25} {mode:<10} "
            f"{s.total_cycles:>8} {exhausted_str:>12} {maxcand_str:>12} {timeout_str:>12}"
        )
        lines.append(row)
    
    lines.append("─" * len(header))
    
    # Totals
    total_cycles = sum(s.total_cycles for s in summaries)
    total_exhausted = sum(s.budget_exhausted_count for s in summaries)
    total_maxcand = sum(s.max_candidates_hit_count for s in summaries)
    total_timeouts = sum(s.timeout_abstentions_total for s in summaries)
    
    exhausted_pct = 100.0 * total_exhausted / total_cycles if total_cycles > 0 else 0.0
    maxcand_pct = 100.0 * total_maxcand / total_cycles if total_cycles > 0 else 0.0
    
    totals = (
        f"{'TOTAL':<40} {'':<25} {'':<10} "
        f"{total_cycles:>8} {f'{total_exhausted} ({exhausted_pct:.1f}%)':>12} "
        f"{f'{total_maxcand} ({maxcand_pct:.1f}%)':>12} {total_timeouts:>12}"
    )
    lines.append(totals)
    
    return "\n".join(lines)


def format_explanation() -> str:
    """Return quantitative explanation of budget states."""
    return """
BUDGET STATE DEFINITIONS (Quantitative)
════════════════════════════════════════════════════════════════════════════════

┌─────────────────────┬────────────────────────────────────────────────────────┐
│ budget_exhausted    │ Cycle terminated early due to wall-clock timeout       │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ Trigger condition   │ time.monotonic() - cycle_start > cycle_budget_s        │
│ Effect              │ Remaining candidates skipped (outcome: "budget_skip")  │
│ Interpretation      │                                                        │
│   >5%  exhausted    │ ⚠ Budget too tight — cycles incomplete                 │
│   1-5% exhausted    │ ◐ Marginal — some cycles hit limit                     │
│   <1%  exhausted    │ ✓ Budget sufficient for slice complexity               │
└─────────────────────┴────────────────────────────────────────────────────────┘

┌─────────────────────┬────────────────────────────────────────────────────────┐
│ max_candidates_hit  │ Cycle reached candidate cap before natural completion  │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ Trigger condition   │ candidates_considered >= max_candidates_per_cycle      │
│ Effect              │ Additional candidates not explored                     │
│ Interpretation      │                                                        │
│   >90% hit          │ ✓ Budget actively constrains (ordering matters)        │
│   50-90% hit        │ ◐ Mixed — some cycles naturally complete               │
│   <50% hit          │ ◎ Slice generates few candidates naturally             │
└─────────────────────┴────────────────────────────────────────────────────────┘

┌─────────────────────┬────────────────────────────────────────────────────────┐
│ timeout_abstentions │ Candidates where verification exceeded taut_timeout_s  │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ Trigger condition   │ verification_time > taut_timeout_s (per candidate)     │
│ Effect              │ Candidate outcome = "abstain_timeout" (excluded)       │
│ Interpretation      │                                                        │
│   >1.0 avg/cycle    │ ⚠ Timeout too tight OR formulas too complex            │
│   0.1-1.0 avg/cycle │ ◐ Some pathological formulas                           │
│   <0.1 avg/cycle    │ ✓ Timeout sufficient for formula complexity            │
└─────────────────────┴────────────────────────────────────────────────────────┘

FORMULAS:
  budget_exhausted_pct    = 100 × (cycles with budget_exhausted=True) / total_cycles
  max_candidates_hit_pct  = 100 × (cycles with max_candidates_hit=True) / total_cycles
  avg_timeout_abstentions = Σ(timeout_abstentions) / total_cycles
"""


# Status emoji mapping for visual annotations
STATUS_EMOJI = {
    "SAFE": "✅",
    "TIGHT": "⚠️",
    "STARVED": "🔥",
    "INVALID": "❓",
}

# Human-friendly hints for each status (non-binding suggestions)
STATUS_HINTS = {
    "SAFE": "Budget parameters are sufficient for this slice.",
    "TIGHT": "Consider increasing budget by 10–20% if degradation continues.",
    "STARVED": "Review cycle_budget_s and taut_timeout_s — significant tuning may help.",
    "INVALID": "Insufficient data to assess budget health.",
}


def format_docs_report(summary: BudgetSummary, health: BudgetHealthResult) -> str:
    """
    Format a docs-friendly, copy-pastable report for a single slice.
    
    Output is Markdown-compatible for easy inclusion in documentation or reports.
    Includes emoji annotations and human-friendly hints.
    
    Example output:
        ### Slice `slice_uplift_goal`
        
        | Metric | Value | Status |
        |--------|-------|--------|
        | Budget exhausted | 2.4% of cycles | ⚠️ TIGHT |
        | Candidate limit hit | 95.0% of cycles | ✅ expected for RFL |
        | Timeout abstentions | 0.05 avg/cycle | ✅ SAFE |
        
        **Overall Health: ⚠️ TIGHT**
        
        💡 *Hint: Consider increasing budget by 10–20% if degradation continues.*
    """
    lines = []
    
    slice_name = summary.slice_name or "(unknown)"
    overall_emoji = STATUS_EMOJI.get(health.status.value, "❓")
    
    lines.append(f"### Slice `{slice_name}`")
    lines.append("")
    
    # Determine status labels with emojis
    exhausted_pct = summary.budget_exhausted_pct
    if exhausted_pct > THRESHOLD_EXHAUSTED_TIGHT:
        exhausted_status = "🔥 STARVED"
        exhausted_label = "STARVED"
    elif exhausted_pct >= THRESHOLD_EXHAUSTED_SAFE:
        if exhausted_pct < 2.5:
            exhausted_status = "⚠️ TIGHT (near SAFE)"
        else:
            exhausted_status = "⚠️ TIGHT"
        exhausted_label = "TIGHT"
    else:
        exhausted_status = "✅ SAFE"
        exhausted_label = "SAFE"
    
    timeout_avg = summary.avg_timeout_abstentions
    if timeout_avg > THRESHOLD_TIMEOUT_TIGHT:
        timeout_status = "🔥 STARVED"
        timeout_label = "STARVED"
    elif timeout_avg >= THRESHOLD_TIMEOUT_SAFE:
        timeout_status = "⚠️ TIGHT"
        timeout_label = "TIGHT"
    else:
        timeout_status = "✅ SAFE"
        timeout_label = "SAFE"
    
    maxcand_pct = summary.max_candidates_hit_pct
    if maxcand_pct > 90:
        maxcand_status = "✅ expected for RFL"
    elif maxcand_pct > 50:
        maxcand_status = "◐ mixed completion"
    else:
        maxcand_status = "◎ slice naturally small"
    
    # Table with emoji annotations
    lines.append("| Metric | Value | Status |")
    lines.append("|--------|-------|--------|")
    lines.append(f"| Budget exhausted | {exhausted_pct:.1f}% of cycles | {exhausted_status} |")
    lines.append(f"| Candidate limit hit | {maxcand_pct:.1f}% of cycles | {maxcand_status} |")
    lines.append(f"| Timeout abstentions | {timeout_avg:.2f} avg/cycle | {timeout_status} |")
    lines.append("")
    
    # Overall health with emoji
    lines.append(f"**Overall Health: {overall_emoji} {health.status.value}**")
    lines.append("")
    
    # Human-friendly hint
    hint = STATUS_HINTS.get(health.status.value, "")
    if hint:
        lines.append(f"💡 *Hint: {hint}*")
        lines.append("")
    
    # Prose summary with emojis
    lines.append(f"- Budget exhausted in {exhausted_pct:.1f}% of cycles ({exhausted_status}).")
    lines.append(f"- Candidate limit hit in {maxcand_pct:.1f}% of cycles ({maxcand_status}).")
    lines.append(f"- Timeout abstentions {timeout_avg:.2f} average per cycle ({timeout_status}).")
    
    return "\n".join(lines)


def format_health_json(summaries: List[BudgetSummary]) -> str:
    """
    Format budget health report as JSON for governance consumption.
    
    Output structure:
        {
            "phase": "PHASE II — NOT USED IN PHASE I",
            "health_report": [
                {
                    "slice": "slice_uplift_goal",
                    "mode": "rfl",
                    "total_cycles": 500,
                    "health": {
                        "status": "TIGHT",
                        "reasons": [...],
                        "metrics": {...}
                    }
                },
                ...
            ],
            "aggregate": {
                "total_slices": 4,
                "safe_count": 2,
                "tight_count": 1,
                "starved_count": 1,
                "invalid_count": 0
            }
        }
    """
    health_results = []
    status_counts = {s: 0 for s in BudgetHealthStatus}
    
    for summary in summaries:
        health = classify_budget_health(summary)
        status_counts[health.status] += 1
        
        health_results.append({
            "slice": summary.slice_name,
            "mode": summary.mode,
            "path": summary.path,
            "total_cycles": summary.total_cycles,
            "health": health.to_dict(),
        })
    
    output = {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "health_report": health_results,
        "aggregate": {
            "total_slices": len(summaries),
            "safe_count": status_counts[BudgetHealthStatus.SAFE],
            "tight_count": status_counts[BudgetHealthStatus.TIGHT],
            "starved_count": status_counts[BudgetHealthStatus.STARVED],
            "invalid_count": status_counts[BudgetHealthStatus.INVALID],
        },
    }
    
    return json.dumps(output, indent=2)


def format_markdown_summary(summaries: List[BudgetSummary]) -> str:
    """
    Format a Markdown summary suitable for GitHub Step Summary.
    
    Used by CI workflow to produce human-readable budget health report.
    """
    lines = []
    lines.append("## 📊 Phase II Budget Health Report")
    lines.append("")
    lines.append("| Slice | Cycles | Exhausted | Max Candidates | Timeouts | Health |")
    lines.append("|-------|--------|-----------|----------------|----------|--------|")
    
    status_emoji = {
        BudgetHealthStatus.SAFE: "✅",
        BudgetHealthStatus.TIGHT: "⚠️",
        BudgetHealthStatus.STARVED: "🔴",
        BudgetHealthStatus.INVALID: "❓",
    }
    
    for summary in summaries:
        health = classify_budget_health(summary)
        emoji = status_emoji.get(health.status, "❓")
        
        lines.append(
            f"| `{summary.slice_name or '?'}` | {summary.total_cycles} | "
            f"{summary.budget_exhausted_pct:.1f}% | {summary.max_candidates_hit_pct:.1f}% | "
            f"{summary.avg_timeout_abstentions:.2f}/cycle | {emoji} {health.status.value} |"
        )
    
    lines.append("")
    
    # Aggregate summary
    status_counts = {s: 0 for s in BudgetHealthStatus}
    for summary in summaries:
        health = classify_budget_health(summary)
        status_counts[health.status] += 1
    
    lines.append("### Summary")
    lines.append(f"- **SAFE**: {status_counts[BudgetHealthStatus.SAFE]} slices")
    lines.append(f"- **TIGHT**: {status_counts[BudgetHealthStatus.TIGHT]} slices")
    lines.append(f"- **STARVED**: {status_counts[BudgetHealthStatus.STARVED]} slices")
    
    if status_counts[BudgetHealthStatus.STARVED] > 0:
        lines.append("")
        lines.append("⚠️ **Advisory**: Some slices have STARVED budget health. Review `cycle_budget_s` and `taut_timeout_s` parameters.")
    
    return "\n".join(lines)


def format_reconciliation(summaries: List["BudgetSummary"]) -> str:
    """
    Cross-log budget reconciliation report.
    
    Compares budget enforcement across multiple experiment logs to identify:
    - Consistency of budget parameters across runs
    - Outlier slices with unusual budget behavior
    - Potential configuration drift
    """
    if len(summaries) < 2:
        return ""
    
    lines = []
    lines.append("")
    lines.append("CROSS-LOG RECONCILIATION")
    lines.append("═" * 60)
    
    # Group by slice
    by_slice: Dict[str, List["BudgetSummary"]] = {}
    for s in summaries:
        key = s.slice_name or "(unknown)"
        if key not in by_slice:
            by_slice[key] = []
        by_slice[key].append(s)
    
    # Check for consistency within each slice
    for slice_name, slice_summaries in sorted(by_slice.items()):
        if len(slice_summaries) < 2:
            continue
        
        lines.append(f"\nSlice: {slice_name} ({len(slice_summaries)} logs)")
        lines.append("─" * 40)
        
        exhausted_pcts = [s.budget_exhausted_pct for s in slice_summaries]
        maxcand_pcts = [s.max_candidates_hit_pct for s in slice_summaries]
        timeout_avgs = [s.avg_timeout_abstentions for s in slice_summaries]
        
        # Calculate variance
        def variance(vals: List[float]) -> float:
            if len(vals) < 2:
                return 0.0
            mean = sum(vals) / len(vals)
            return sum((x - mean) ** 2 for x in vals) / len(vals)
        
        exhausted_var = variance(exhausted_pcts)
        maxcand_var = variance(maxcand_pcts)
        timeout_var = variance(timeout_avgs)
        
        lines.append(f"  budget_exhausted_pct:    mean={sum(exhausted_pcts)/len(exhausted_pcts):.2f}%  var={exhausted_var:.4f}")
        lines.append(f"  max_candidates_hit_pct:  mean={sum(maxcand_pcts)/len(maxcand_pcts):.2f}%  var={maxcand_var:.4f}")
        lines.append(f"  avg_timeout_abstentions: mean={sum(timeout_avgs)/len(timeout_avgs):.4f}  var={timeout_var:.6f}")
        
        # Flag high variance
        if exhausted_var > 25:  # >5% std dev
            lines.append(f"  ⚠ HIGH VARIANCE in budget_exhausted — check config consistency")
        if maxcand_var > 400:  # >20% std dev
            lines.append(f"  ⚠ HIGH VARIANCE in max_candidates_hit — check slice params")
    
    return "\n".join(lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Summarize budget usage from Phase II experiment JSONL logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "logs",
        nargs="+",
        type=str,
        help="Path(s) to JSONL log file(s)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable format."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show quantitative explanation of budget states."
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Show cross-log budget reconciliation (requires multiple logs)."
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate docs-friendly Markdown report (copy-pastable)."
    )
    parser.add_argument(
        "--health-json",
        type=str,
        metavar="PATH",
        help="Write budget health classification to JSON file."
    )
    parser.add_argument(
        "--markdown-summary",
        type=str,
        metavar="PATH",
        help="Write Markdown summary to file (for GitHub Step Summary)."
    )
    
    args = parser.parse_args()
    
    print("PHASE II — NOT USED IN PHASE I", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Parse all log files
    summaries = []
    for log_path in args.logs:
        path = Path(log_path)
        if not path.exists():
            print(f"WARNING: File not found, skipping: {path}", file=sys.stderr)
            continue
        
        try:
            summary = parse_log_file(path)
            summaries.append(summary)
        except Exception as e:
            print(f"ERROR: Failed to parse {path}: {e}", file=sys.stderr)
            continue
    
    if not summaries:
        print("ERROR: No valid log files found.", file=sys.stderr)
        sys.exit(1)
    
    # Show explanation if requested
    if args.explain:
        print(format_explanation())
        if not args.json:
            print("")
    
    # Write health JSON if requested
    if args.health_json:
        health_json = format_health_json(summaries)
        health_path = Path(args.health_json)
        health_path.parent.mkdir(parents=True, exist_ok=True)
        health_path.write_text(health_json, encoding="utf-8")
        print(f"Budget health JSON written to: {args.health_json}", file=sys.stderr)
    
    # Write Markdown summary if requested (for CI)
    if args.markdown_summary:
        md_summary = format_markdown_summary(summaries)
        md_path = Path(args.markdown_summary)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_summary, encoding="utf-8")
        print(f"Markdown summary written to: {args.markdown_summary}", file=sys.stderr)
    
    # Output
    if args.report:
        # Docs-friendly report for each slice
        for summary in summaries:
            health = classify_budget_health(summary)
            print(format_docs_report(summary, health))
            print("")
    elif args.json:
        output = {
            "phase": "PHASE II — NOT USED IN PHASE I",
            "logs": [s.to_dict() for s in summaries],
        }
        print(json.dumps(output, indent=2))
    else:
        if len(summaries) == 1:
            print(format_summary(summaries[0]))
            # Also show health classification
            health = classify_budget_health(summaries[0])
            print("")
            print(f"Health: {health.status.value}")
            for reason in health.reasons:
                print(f"  - {reason}")
        else:
            print(f"Budget Usage Summary ({len(summaries)} files)")
            print("")
            print(format_multi_summary(summaries))
            
            # Show health classification for each
            print("")
            print("HEALTH CLASSIFICATION")
            print("─" * 60)
            for summary in summaries:
                health = classify_budget_health(summary)
                print(f"  {summary.slice_name or '?'}: {health.status.value}")
        
        # Show reconciliation if requested and multiple logs
        if args.reconcile and len(summaries) >= 2:
            print(format_reconciliation(summaries))


if __name__ == "__main__":
    main()

