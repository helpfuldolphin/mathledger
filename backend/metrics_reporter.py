"""
ASCII-Only Metrics Reporter

Generates reproducible, diff-friendly ASCII dashboards for terminal display.
All output is pure ASCII (no emoji, no Unicode box-drawing) for maximum reproducibility.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List


class ASCIIReporter:
    """Generate ASCII-only reproducible metric summaries"""

    def __init__(self, metrics_file: Path):
        self.metrics_file = metrics_file
        with open(metrics_file) as f:
            self.data = json.load(f)

    def _format_number(self, value: float, precision: int = 2) -> str:
        """Format number with fixed precision"""
        if value >= 1000:
            return f"{value:,.{precision}f}"
        return f"{value:.{precision}f}"

    def _box_line(self, width: int, char: str = "=") -> str:
        """Generate box border line"""
        return char * width

    def _centered(self, text: str, width: int) -> str:
        """Center text within width"""
        padding = (width - len(text)) // 2
        return " " * padding + text

    def _row(self, label: str, value: str, width: int = 70) -> str:
        """Generate labeled row"""
        dots = "." * (width - len(label) - len(value) - 2)
        return f"{label} {dots} {value}"

    def generate_header(self) -> str:
        """Generate report header"""
        width = 70
        title = "MATHLEDGER METRICS CARTOGRAPHER"
        subtitle = "Canonical Metrics Summary"

        notes = self.data.get('notes') or []
        lines = [
            self._box_line(width),
            self._centered(title, width),
            self._centered(subtitle, width),
            self._box_line(width),
            f"Session: {self.data.get('session_id', 'unknown')}",
            f"Timestamp: {self.data.get('timestamp', 'unknown')}",
            f"Source: {self.data.get('source', 'unknown')}",
            self._box_line(width, "-"),
        ]
        if notes:
            lines.append("Warnings:")
            lines.extend(f"  - {note}" for note in notes)
            lines.append("")
        lines.append("")
        return "\n".join(lines)

    def generate_throughput_section(self) -> str:
        """Generate throughput metrics section"""
        metrics = self.data.get('metrics', {})
        throughput = metrics.get('throughput', {})

        if not throughput:
            return ""

        lines = [
            "THROUGHPUT METRICS",
            self._box_line(70, "-"),
            self._row(
                "Proofs/sec",
                self._format_number(throughput.get('proofs_per_sec', 0))
            ),
            self._row(
                "Proofs/hour",
                self._format_number(throughput.get('proofs_per_hour', 0))
            ),
            self._row(
                "Total proofs",
                f"{throughput.get('proof_count_total', 0):,}"
            ),
            self._row(
                "Successful proofs",
                f"{throughput.get('proof_success_count', 0):,}"
            ),
            self._row(
                "Failed proofs",
                f"{throughput.get('proof_failure_count', 0):,}"
            )
        ]

        lines.append("")
        return "\n".join(lines)

    def generate_success_rates_section(self) -> str:
        """Generate success rates section"""
        metrics = self.data.get('metrics', {})
        success_rates = metrics.get('success_rates', {})

        if not success_rates:
            return ""

        lines = [
            "SUCCESS RATES",
            self._box_line(70, "-"),
            self._row(
                "Proof success rate",
                f"{self._format_number(success_rates.get('proof_success_rate', 0))}%"
            ),
            self._row(
                "Abstention rate",
                f"{self._format_number(success_rates.get('abstention_rate', 0))}%"
            ),
            self._row(
                "Verification success",
                f"{self._format_number(success_rates.get('verification_success_rate', 0))}%"
            ),
            ""
        ]
        return "\n".join(lines)

    def generate_coverage_section(self) -> str:
        """Generate coverage metrics section"""
        metrics = self.data.get('metrics', {})
        coverage = metrics.get('coverage', {})

        if not coverage:
            return ""

        lines = [
            "COVERAGE METRICS",
            self._box_line(70, "-"),
            self._row(
                "Max depth reached",
                str(coverage.get('max_depth_reached', 0))
            ),
            self._row(
                "Unique statements",
                f"{coverage.get('unique_statements', 0):,}"
            ),
            self._row(
                "Unique proofs",
                f"{coverage.get('unique_proofs', 0):,}"
            ),
            ""
        ]
        return "\n".join(lines)

    def generate_uplift_section(self) -> str:
        """Generate uplift metrics section"""
        metrics = self.data.get('metrics', {})
        uplift = metrics.get('uplift', {})

        if not uplift:
            return ""

        ratio = uplift.get('uplift_ratio', 0)
        ci_lower = uplift.get('confidence_interval_lower', 0)
        ci_upper = uplift.get('confidence_interval_upper', 0)
        ci_width = uplift.get('ci_width', 0)

        lines = [
            "UPLIFT METRICS (A/B TESTING)",
            self._box_line(70, "-"),
            self._row(
                "Uplift ratio",
                f"{self._format_number(ratio)}x"
            ),
            self._row(
                "Baseline mean",
                self._format_number(uplift.get('baseline_mean', 0))
            ),
            self._row(
                "Guided mean",
                self._format_number(uplift.get('guided_mean', 0))
            ),
            self._row(
                "Delta from baseline",
                self._format_number(uplift.get('delta_from_baseline', 0))
            ),
            self._row(
                "P-value",
                f"{uplift.get('p_value', 0):.4f}"
            ),
            self._row(
                "Confidence interval",
                f"[{self._format_number(ci_lower)}, {self._format_number(ci_upper)}]"
            ),
            self._row(
                "CI width",
                self._format_number(ci_width)
            ),
            ""
        ]
        return "\n".join(lines)

    def generate_performance_section(self) -> str:
        """Generate performance metrics section"""
        metrics = self.data.get('metrics', {})
        performance = metrics.get('performance', {})

        if not performance:
            return ""

        regression = performance.get('regression_detected', False)
        regression_status = "YES" if regression else "NO"

        lines = [
            "PERFORMANCE METRICS",
            self._box_line(70, "-"),
            self._row(
                "Mean latency",
                f"{self._format_number(performance.get('mean_latency_ms', 0))} ms"
            ),
            self._row(
                "P50 latency",
                f"{self._format_number(performance.get('p50_latency_ms', 0))} ms"
            ),
            self._row(
                "P95 latency",
                f"{self._format_number(performance.get('p95_latency_ms', 0))} ms"
            ),
            self._row(
                "P99 latency",
                f"{self._format_number(performance.get('p99_latency_ms', 0))} ms"
            ),
            self._row(
                "Max latency",
                f"{self._format_number(performance.get('max_latency_ms', 0))} ms"
            ),
            self._row(
                "Mean memory",
                f"{self._format_number(performance.get('mean_memory_mb', 0))} MB"
            ),
            self._row(
                "Max memory",
                f"{self._format_number(performance.get('max_memory_mb', 0))} MB"
            ),
            self._row(
                "Sample size",
                str(performance.get('sample_size', 0))
            ),
            self._row(
                "Regression detected",
                regression_status
            ),
            ""
        ]
        return "\n".join(lines)

    def generate_trends_section(self) -> str:
        """Generate trends summary section"""
        metrics = self.data.get('metrics', {})
        trends = metrics.get('trends', {})

        if not trends:
            return ""

        def format_trend(name: str, data: Dict[str, Any]) -> List[str]:
            if not data:
                return []
            return [
                self._row(f"{name} latest", self._format_number(data.get('latest', 0))),
                self._row("Delta from previous", self._format_number(data.get('delta_from_previous', 0))),
                self._row("Moving average (short)", self._format_number(data.get('moving_average_short', 0))),
                self._row("Moving average (long)", self._format_number(data.get('moving_average_long', 0))),
                self._row("Samples", str(data.get('samples', 0))),
                self._row("Trend", data.get('trend', 'flat')),
                ""
            ]

        lines = [
            "TREND SYNTHESIS",
            self._box_line(70, "-"),
        ]

        lines.extend(format_trend("Proofs/sec", trends.get('proofs_per_sec', {})))
        lines.extend(format_trend("Proof success rate", trends.get('proof_success_rate', {})))
        lines.extend(format_trend("P95 latency (ms)", trends.get('p95_latency_ms', {})))
        lines.append(self._row("History retention", str(trends.get('retention', 0))))
        lines.append("")

        return "\n".join(line for line in lines if line != "")

    def generate_first_organism_trends_section(self) -> str:
        """Generate First Organism trends section"""
        metrics = self.data.get('metrics', {})
        trends = metrics.get('trends', {})
        fo_trends = trends.get('first_organism', {})

        if not fo_trends:
            return ""

        def format_fo_trend(name: str, data: Dict[str, Any], unit: str = "") -> List[str]:
            if not data:
                return []
            latest = data.get('latest', 0)
            delta = data.get('delta_from_previous', 0)
            trend = data.get('trend', 'flat')
            samples = data.get('samples', 0)

            # Format trend indicator
            trend_arrow = "^" if trend == "up" else "v" if trend == "down" else "-"
            delta_sign = "+" if delta > 0 else ""

            return [
                self._row(
                    f"{name}",
                    f"{self._format_number(latest)}{unit} [{trend_arrow}] ({delta_sign}{self._format_number(delta)}, n={samples})"
                ),
            ]

        lines = [
            "FIRST ORGANISM HEALTH TRENDS",
            self._box_line(70, "-"),
        ]

        lines.extend(format_fo_trend("Duration", fo_trends.get('duration_seconds', {}), "s"))
        lines.extend(format_fo_trend("Abstentions", fo_trends.get('abstention_count', {})))
        lines.extend(format_fo_trend("Success rate", fo_trends.get('success_rate', {}), "%"))
        lines.extend(format_fo_trend("Total runs", fo_trends.get('runs_total', {})))
        lines.append("")

        return "\n".join(line for line in lines if line != "")

    def generate_blockchain_section(self) -> str:
        """Generate blockchain metrics section"""
        metrics = self.data.get('metrics', {})
        blockchain = metrics.get('blockchain', {})

        if not blockchain:
            return ""

        merkle = blockchain.get('merkle_root', '')
        merkle_display = merkle[:16] + "..." if len(merkle) > 16 else merkle

        lines = [
            "BLOCKCHAIN METRICS",
            self._box_line(70, "-"),
            self._row(
                "Block height",
                str(blockchain.get('block_height', 0))
            ),
            self._row(
                "Total blocks",
                str(blockchain.get('total_blocks', 0))
            ),
            self._row(
                "Merkle root",
                merkle_display
            ),
            ""
        ]
        return "\n".join(lines)

    def generate_first_organism_section(self) -> str:
        """Generate First Organism vital signs section"""
        metrics = self.data.get('metrics', {})
        fo = metrics.get('first_organism', {})
        if not fo:
            return ""

        ht = fo.get('last_ht_hash', '')
        ht_display = (ht[:16] + "...") if ht else "N/A"
        duration = fo.get('last_duration_seconds', 0.0)
        avg_duration = fo.get('average_duration_seconds', 0.0)
        median_duration = fo.get('median_duration_seconds', 0.0)
        abstentions = fo.get('abstention_count', 0)
        timestamp = fo.get('last_run_timestamp', self.data.get('timestamp', 'unknown'))
        runs_total = fo.get('runs_total', 0)
        last_status = fo.get('last_status', 'unknown')
        success_rate = fo.get('success_rate', 0.0)
        duration_delta = fo.get('duration_delta', 0.0)
        abstention_delta = fo.get('abstention_delta', 0)

        # Format duration delta with sign
        delta_sign = "+" if duration_delta > 0 else ""
        delta_str = f"{delta_sign}{duration_delta:.2f}s" if duration_delta != 0 else "0.00s"

        # Format abstention delta with sign
        abs_delta_sign = "+" if abstention_delta > 0 else ""
        abs_delta_str = f"{abs_delta_sign}{abstention_delta}" if abstention_delta != 0 else "0"

        # Health status (prefer computed health_status, fall back to last_status)
        health_status = fo.get('health_status', '')
        if not health_status:
            if last_status == "success" and success_rate >= 80.0:
                health_status = "ALIVE"
            elif success_rate >= 50.0:
                health_status = "DEGRADED"
            elif runs_total > 0:
                health_status = "CRITICAL"
            else:
                health_status = "UNKNOWN"

        # Trend indicators
        duration_trend = fo.get('duration_trend', 'flat')
        abstention_trend = fo.get('abstention_trend', 'flat')

        def trend_arrow(trend: str) -> str:
            if trend == "up":
                return "/\\"
            elif trend == "down":
                return "\\/"
            return "--"

        lines = [
            "FIRST ORGANISM VITAL SIGNS",
            self._box_line(70, "-"),
            self._row("Health Status", health_status),
            self._row("Last run", timestamp if timestamp else "N/A"),
            self._row("H_t (short)", ht_display),
            self._row("Total runs", str(runs_total)),
            self._row("Success rate", f"{success_rate:.1f}%"),
            self._row("Run duration", f"{duration:.2f}s (delta: {delta_str}) {trend_arrow(duration_trend)}"),
            self._row("Avg duration", f"{avg_duration:.2f}s"),
            self._row("Median duration", f"{median_duration:.2f}s"),
            self._row("Abstentions", f"{abstentions} (delta: {abs_delta_str}) {trend_arrow(abstention_trend)}"),
            "",
        ]

        # Add duration trend sparkline if we have history
        duration_history = fo.get('duration_history', [])
        if duration_history and len(duration_history) > 1:
            sparkline = self._generate_sparkline(duration_history[:10])
            lines.insert(-1, self._row("Duration trend", f"[{sparkline}] (last {len(duration_history[:10])} runs)"))

        # Add abstention trend sparkline if we have history
        abstention_history = fo.get('abstention_history', [])
        if abstention_history and len(abstention_history) > 1:
            sparkline = self._generate_sparkline([float(x) for x in abstention_history[:10]])
            lines.insert(-1, self._row("Abstention trend", f"[{sparkline}] (last {len(abstention_history[:10])} runs)"))

        # H_t verification status
        ht_verification = fo.get('ht_verification', {})
        if ht_verification:
            verified = ht_verification.get('verified', False)
            unique_ratio = ht_verification.get('uniqueness_ratio', 0.0)
            verify_status = "OK" if verified else "WARN"
            lines.insert(-1, self._row("H_t Verification", f"{verify_status} (uniqueness: {unique_ratio:.2%})"))

        return "\n".join(lines)

    def _generate_sparkline(self, values: List[float], width: int = 10) -> str:
        """Generate ASCII sparkline from values (most recent first)."""
        if not values:
            return ""

        recent = values[:width]
        if not recent:
            return ""

        min_val = min(recent)
        max_val = max(recent)
        range_val = max_val - min_val if max_val > min_val else 1.0

        chars = "_.-="

        def bar_char(val: float) -> str:
            if range_val == 0:
                return chars[len(chars) // 2]
            normalized = (val - min_val) / range_val
            idx = min(int(normalized * len(chars)), len(chars) - 1)
            return chars[idx]

        # Reverse to show oldest-to-newest left-to-right
        return "".join(bar_char(v) for v in reversed(recent))

    def generate_variance_section(self) -> str:
        """Generate variance metrics section"""
        variance = self.data.get('variance', {})

        if not variance:
            return ""

        cv = variance.get('coefficient_of_variation', 0)
        epsilon = variance.get('epsilon_tolerance', 0)
        within = variance.get('within_tolerance', True)
        status = "PASS" if within else "FAIL"

        lines = [
            "VARIANCE ANALYSIS",
            self._box_line(70, "-"),
            self._row(
                "Coefficient of variation",
                f"{cv:.6f}"
            ),
            self._row(
                "Epsilon tolerance",
                f"{epsilon:.6f}"
            ),
            self._row(
                "Within tolerance",
                status
            ),
            ""
        ]
        return "\n".join(lines)

    def generate_provenance_section(self) -> str:
        """Generate provenance section"""
        provenance = self.data.get('provenance', {})

        if not provenance:
            return ""

        merkle = provenance.get('merkle_hash', '')
        merkle_display = merkle[:32] + "..." if len(merkle) > 32 else merkle

        lines = [
            "PROVENANCE & ATTESTATION",
            self._box_line(70, "-"),
            self._row(
                "Collector",
                provenance.get('collector', 'unknown')
            ),
            self._row(
                "Merkle hash",
                merkle_display
            )
        ]

        policy = provenance.get('policy_hash')
        if policy:
            policy_display = policy[:32] + "..." if len(policy) > 32 else policy
            lines.append(self._row("Policy hash", policy_display))

        history_merkle = provenance.get('history_merkle')
        if history_merkle:
            history_display = history_merkle[:32] + "..." if len(history_merkle) > 32 else history_merkle
            lines.append(self._row("History merkle", history_display))

        collectors = provenance.get('collectors', [])
        if collectors:
            lines.append("")
            lines.append("Collectors:")
            for collector in collectors:
                name = collector.get('name', 'unknown')
                status = collector.get('status', 'unknown')
                lines.append(f"  - {name}: {status}")

        sources = provenance.get('sources', [])
        if sources:
            lines.append("")
            lines.append("Data sources:")
            for source in sources:
                lines.append(f"  - {source}")

        warnings = provenance.get('warnings', [])
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for warn in warnings:
                lines.append(f"  - {warn}")

        lines.append("")
        return "\n".join(lines)

    def generate_footer(self) -> str:
        """Generate report footer"""
        width = 70
        variance = self.data.get('variance', {})
        within_tolerance = variance.get('within_tolerance', True) if variance else True
        epsilon = variance.get('epsilon_tolerance', 0.01) if variance else 0.01

        total_entries = sum(
            len(v) if isinstance(v, dict) else 1
            for v in self.data.get('metrics', {}).values()
        )

        if within_tolerance:
            status = f"[PASS] Metrics Canonicalized entries={total_entries} variance<=epsilon={epsilon}"
        else:
            cv = variance.get('coefficient_of_variation', 0) if variance else 0
            status = f"[WARN] Metrics Canonicalized entries={total_entries} variance={cv:.4f} > epsilon={epsilon}"

        lines = [
            self._box_line(width),
            self._centered(status, width),
            self._box_line(width)
        ]
        return "\n".join(lines)

    def generate_full_report(self) -> str:
        """Generate complete ASCII report"""
        sections = [
            self.generate_header(),
            self.generate_throughput_section(),
            self.generate_success_rates_section(),
            self.generate_coverage_section(),
            self.generate_uplift_section(),
            self.generate_performance_section(),
            self.generate_blockchain_section(),
            self.generate_first_organism_section(),
            self.generate_first_organism_trends_section(),
            self.generate_trends_section(),
            self.generate_variance_section(),
            self.generate_provenance_section(),
            self.generate_footer()
        ]

        # Filter out empty sections
        return "\n".join(s for s in sections if s)


def main():
    """CLI entry point"""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    metrics_file = project_root / "artifacts" / "metrics" / "latest.json"

    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        print("Run metrics_cartographer.py first to generate metrics.")
        return 1

    reporter = ASCIIReporter(metrics_file)
    report = reporter.generate_full_report()

    # Print to stdout
    print(report)

    # Also save to file
    output_file = project_root / "artifacts" / "metrics" / "latest_report.txt"
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
