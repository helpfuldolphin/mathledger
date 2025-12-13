"""
TDA Console Tile for Phase X Dashboard

Provides a summary tile rendering of TDA metrics (SNS, PCS, DRS, HSS)
for console output and dashboard integration.

See: docs/system_law/TDA_PhaseX_Binding.md

SHADOW MODE CONTRACT:
- Display only, no governance modification
- All metrics are observational
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.tda.metrics import TDAMetrics, TDAWindowMetrics
from backend.tda.monitor import TDASummary

__all__ = [
    "TDAConsoleTile",
    "render_tda_tile",
    "render_tda_summary",
]


@dataclass
class TDAConsoleTile:
    """
    TDA metrics summary for console/dashboard display.

    Renders a compact summary of SNS, PCS, DRS, HSS with status indicators.
    """

    # Current metrics
    sns: float = 0.0
    pcs: float = 1.0
    drs: float = 0.0
    hss: float = 1.0

    # Status indicators
    sns_status: str = "OK"  # OK, WARN, CRITICAL
    pcs_status: str = "OK"
    drs_status: str = "OK"
    hss_status: str = "OK"

    # Envelope status
    in_envelope: bool = True
    envelope_exit_streak: int = 0

    # Summary stats (optional)
    total_cycles: int = 0
    red_flag_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "metrics": {
                "sns": {"value": round(self.sns, 4), "status": self.sns_status},
                "pcs": {"value": round(self.pcs, 4), "status": self.pcs_status},
                "drs": {"value": round(self.drs, 4), "status": self.drs_status},
                "hss": {"value": round(self.hss, 4), "status": self.hss_status},
            },
            "envelope": {
                "in_envelope": self.in_envelope,
                "exit_streak": self.envelope_exit_streak,
            },
            "summary": {
                "total_cycles": self.total_cycles,
                "red_flag_count": self.red_flag_count,
            },
            "mode": "SHADOW",
        }

    def render(self, use_color: bool = True) -> str:
        """
        Render console tile as formatted string.

        Args:
            use_color: Whether to use ANSI color codes

        Returns:
            Formatted string for console display
        """
        # Status indicators with optional coloring
        def status_indicator(status: str) -> str:
            if not use_color:
                return f"[{status}]"
            colors = {
                "OK": "\033[92m",      # Green
                "WARN": "\033[93m",    # Yellow
                "CRITICAL": "\033[91m", # Red
            }
            reset = "\033[0m"
            color = colors.get(status, "")
            return f"{color}[{status}]{reset}"

        # Build tile
        lines = [
            "+------------------------------------------+",
            "|        TDA Mind Scanner (SHADOW)         |",
            "+------------------------------------------+",
            f"| SNS (Structural Novelty): {self.sns:>6.3f} {status_indicator(self.sns_status):>10} |",
            f"| PCS (Proof Coherence):    {self.pcs:>6.3f} {status_indicator(self.pcs_status):>10} |",
            f"| DRS (Drift Rate):         {self.drs:>6.3f} {status_indicator(self.drs_status):>10} |",
            f"| HSS (Homological Stab.):  {self.hss:>6.3f} {status_indicator(self.hss_status):>10} |",
            "+------------------------------------------+",
        ]

        # Envelope status line
        envelope_char = "IN" if self.in_envelope else "OUT"
        envelope_color = "\033[92m" if self.in_envelope else "\033[91m" if use_color else ""
        reset = "\033[0m" if use_color else ""
        lines.append(f"| Envelope: {envelope_color}{envelope_char}{reset}  Exit streak: {self.envelope_exit_streak:>3}          |")

        if self.total_cycles > 0:
            lines.append(f"| Cycles: {self.total_cycles:>6}  Red-flags: {self.red_flag_count:>4}         |")

        lines.append("+------------------------------------------+")

        return "\n".join(lines)


def render_tda_tile(
    metrics: Optional[TDAMetrics] = None,
    summary: Optional[TDASummary] = None,
    use_color: bool = True,
) -> str:
    """
    Render TDA tile from metrics or summary.

    Args:
        metrics: Current TDA metrics (optional)
        summary: TDA summary (optional)
        use_color: Whether to use ANSI colors

    Returns:
        Formatted tile string
    """
    if metrics is not None:
        # Determine status from thresholds
        sns_status = "CRITICAL" if metrics.sns > 0.6 else ("WARN" if metrics.sns > 0.4 else "OK")
        pcs_status = "CRITICAL" if metrics.pcs < 0.4 else ("WARN" if metrics.pcs < 0.6 else "OK")
        drs_status = "CRITICAL" if metrics.drs > 0.2 else ("WARN" if metrics.drs > 0.1 else "OK")
        hss_status = "CRITICAL" if metrics.hss < 0.4 else ("WARN" if metrics.hss < 0.6 else "OK")

        tile = TDAConsoleTile(
            sns=metrics.sns,
            pcs=metrics.pcs,
            drs=metrics.drs,
            hss=metrics.hss,
            sns_status=sns_status,
            pcs_status=pcs_status,
            drs_status=drs_status,
            hss_status=hss_status,
            in_envelope=metrics.in_tda_envelope,
        )
    elif summary is not None:
        # Build from summary
        sns_status = "CRITICAL" if summary.sns_anomaly_count > 0 else ("WARN" if summary.sns_max > 0.4 else "OK")
        pcs_status = "CRITICAL" if summary.pcs_collapse_count > 0 else ("WARN" if summary.pcs_min < 0.6 else "OK")
        drs_status = "CRITICAL" if summary.drs_critical_count > 0 else ("WARN" if summary.drs_max > 0.1 else "OK")
        hss_status = "CRITICAL" if summary.hss_degradation_count > 0 else ("WARN" if summary.hss_min < 0.6 else "OK")

        tile = TDAConsoleTile(
            sns=summary.sns_mean,
            pcs=summary.pcs_mean,
            drs=summary.drs_mean,
            hss=summary.hss_mean,
            sns_status=sns_status,
            pcs_status=pcs_status,
            drs_status=drs_status,
            hss_status=hss_status,
            in_envelope=summary.envelope_occupancy >= 0.9,
            envelope_exit_streak=summary.max_envelope_exit_streak,
            total_cycles=summary.total_cycles,
            red_flag_count=summary.total_red_flags,
        )
    else:
        # Empty tile
        tile = TDAConsoleTile()

    return tile.render(use_color=use_color)


def render_tda_summary(summary: TDASummary, use_color: bool = True) -> str:
    """
    Render detailed TDA summary for console.

    Args:
        summary: TDA summary to render
        use_color: Whether to use ANSI colors

    Returns:
        Formatted summary string
    """
    def color(text: str, code: str) -> str:
        if not use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    lines = [
        "=" * 50,
        color("        TDA Mind Scanner Summary (SHADOW)", "1"),
        "=" * 50,
        "",
        f"Total Cycles: {summary.total_cycles}",
        "",
        color("--- Structural Novelty Score (SNS) ---", "4"),
        f"  Mean: {summary.sns_mean:.4f}",
        f"  Max:  {summary.sns_max:.4f}",
        f"  Anomalies (>0.6): {summary.sns_anomaly_count}",
        "",
        color("--- Proof Coherence Score (PCS) ---", "4"),
        f"  Mean: {summary.pcs_mean:.4f}",
        f"  Min:  {summary.pcs_min:.4f}",
        f"  Collapses (<0.4): {summary.pcs_collapse_count}",
        "",
        color("--- Drift Rate Score (DRS) ---", "4"),
        f"  Mean: {summary.drs_mean:.4f}",
        f"  Max:  {summary.drs_max:.4f}",
        f"  Critical (>0.2): {summary.drs_critical_count}",
        "",
        color("--- Homological Stability Score (HSS) ---", "4"),
        f"  Mean: {summary.hss_mean:.4f}",
        f"  Min:  {summary.hss_min:.4f}",
        f"  Degradations (<0.4): {summary.hss_degradation_count}",
        "",
        color("--- Envelope (Omega_TDA) ---", "4"),
        f"  Occupancy: {summary.envelope_occupancy:.2%}",
        f"  Exit Events: {summary.envelope_exit_total}",
        f"  Max Exit Streak: {summary.max_envelope_exit_streak}",
        "",
        color("--- Red Flags ---", "4"),
        f"  Total: {summary.total_red_flags}",
    ]

    for flag_type, count in summary.red_flags_by_type.items():
        lines.append(f"    {flag_type}: {count}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


def format_tda_json(
    metrics: Optional[TDAMetrics] = None,
    window_metrics: Optional[TDAWindowMetrics] = None,
    summary: Optional[TDASummary] = None,
) -> Dict[str, Any]:
    """
    Format TDA data as JSON for API response.

    Args:
        metrics: Current cycle metrics
        window_metrics: Window aggregated metrics
        summary: Run summary

    Returns:
        JSON-serializable dictionary
    """
    result: Dict[str, Any] = {"mode": "SHADOW"}

    if metrics is not None:
        result["current"] = metrics.to_dict()

    if window_metrics is not None:
        result["window"] = window_metrics.to_dict()

    if summary is not None:
        result["summary"] = summary.to_dict()

    return result
