"""
Markdown Report Generator for Metrics Cartographer

Generates concise markdown reports (<=400 words) for reports/metrics_{date}.md
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class MarkdownReportGenerator:
    """Generate concise markdown metrics reports (<=400 words)"""

    def __init__(self, metrics_file: Path):
        self.metrics_file = metrics_file
        with open(metrics_file) as f:
            self.data = json.load(f)

    def generate_report(self) -> str:
        """Generate markdown report (≤400 words)"""
        lines = []

        # Header
        lines.append("# MathLedger Metrics Report")
        lines.append("")

        # Metadata
        timestamp = self.data.get('timestamp', 'unknown')
        session = self.data.get('session_id', 'unknown')
        date = timestamp.split('T')[0] if 'T' in timestamp else timestamp

        lines.append(f"**Date**: {date}")
        lines.append(f"**Session**: `{session}`")
        lines.append(f"**Source**: {self.data.get('source', 'unknown')}")
        lines.append("")

        # Provenance
        prov = self.data.get('provenance', {})
        merkle = prov.get('merkle_hash', '')[:16] + '...' if prov.get('merkle_hash') else 'N/A'
        lines.append(f"**Provenance**: `{merkle}` (SHA-256)")
        lines.append("")

        # Metrics summary
        metrics = self.data.get('metrics', {})

        lines.append("## Performance Summary")
        lines.append("")

        # Throughput
        throughput = metrics.get('throughput', {})
        if throughput:
            pps = throughput.get('proofs_per_sec', 0)
            pph = throughput.get('proofs_per_hour', 0)
            delta = throughput.get('delta_from_baseline', 0)
            lines.append(f"**Throughput**: {pps:.2f} proofs/sec ({pph:.1f}/hour)")
            if delta != 0:
                sign = "+" if delta > 0 else ""
                lines.append(f"- Delta from baseline: {sign}{delta:.1f}")

        # Success rates
        success = metrics.get('success_rates', {})
        if success:
            proof_rate = success.get('proof_success_rate', 0)
            abstain_rate = success.get('abstention_rate', 0)
            lines.append(f"**Success**: {proof_rate:.1f}% proof rate, {abstain_rate:.1f}% abstention")

        # Coverage
        coverage = metrics.get('coverage', {})
        if coverage:
            depth = coverage.get('max_depth_reached', 0)
            statements = coverage.get('unique_statements', 0)
            proofs = coverage.get('unique_proofs', 0)
            lines.append(f"**Coverage**: {statements:,} statements, {proofs:,} proofs (depth {depth})")

        lines.append("")

        # Uplift (if available)
        uplift = metrics.get('uplift', {})
        if uplift and uplift.get('uplift_ratio', 0) > 0:
            lines.append("## A/B Testing Results")
            lines.append("")
            ratio = uplift.get('uplift_ratio', 0)
            baseline = uplift.get('baseline_mean', 0)
            guided = uplift.get('guided_mean', 0)
            p_value = uplift.get('p_value', 1)
            ci_lower = uplift.get('confidence_interval_lower', 0)
            ci_upper = uplift.get('confidence_interval_upper', 0)
            delta = uplift.get('delta_from_baseline', guided - baseline)

            lines.append(f"**Uplift Ratio**: {ratio:.2f}x (baseline {baseline:.1f} -> guided {guided:.1f})")
            lines.append(f"**Delta**: {delta:.2f}")
            lines.append(f"**Confidence Interval**: [{ci_lower:.2f}, {ci_upper:.2f}]")
            lines.append(f"**Statistical Significance**: p={p_value:.4f}")
            lines.append("")

        # Performance
        perf = metrics.get('performance', {})
        if perf:
            lines.append("## Latency & Memory")
            lines.append("")
            mean_lat = perf.get('mean_latency_ms', 0)
            p95_lat = perf.get('p95_latency_ms', 0)
            mean_mem = perf.get('mean_memory_mb', 0)
            regression = perf.get('regression_detected', False)

            lines.append(f"- Mean latency: {mean_lat:.3f}ms (p95: {p95_lat:.3f}ms)")
            lines.append(f"- Mean memory: {mean_mem:.3f}MB")
            lines.append(f"- Regression: {'YES' if regression else 'NO'}")
            lines.append("")

        # Blockchain
        blockchain = metrics.get('blockchain', {})
        if blockchain:
            height = blockchain.get('block_height', 0)
            total = blockchain.get('total_blocks', 0)
            merkle_root = blockchain.get('merkle_root', '')[:16] + '...' if blockchain.get('merkle_root') else 'N/A'
            lines.append(f"**Blockchain**: Height {height}, {total} blocks, root `{merkle_root}`")
            lines.append("")

        # Variance analysis
        variance = self.data.get('variance', {})
        if variance:
            lines.append("## Variance Analysis")
            lines.append("")
            cv = variance.get('coefficient_of_variation', 0)
            epsilon = variance.get('epsilon_tolerance', 0)
            within = variance.get('within_tolerance', True)

            status = "PASS" if within else "WARN"
            lines.append(f"**Status**: {status}")
            lines.append(f"- Coefficient of Variation: {cv:.6f}")
            lines.append(f"- Epsilon Tolerance: {epsilon:.6f}")
            lines.append("")

        trends = metrics.get('trends', {})
        if trends:
            lines.append("## Cross-Run Trends")
            lines.append("")
            def trend_summary(name: str, data: Dict[str, Any]) -> None:
                if not data:
                    return
                lines.append(f"- {name}: latest={data.get('latest', 0):.4f}, delta={data.get('delta_from_previous', 0):.4f}, short_avg={data.get('moving_average_short', 0):.4f}, trend={data.get('trend', 'flat')}")
            trend_summary("Proofs/sec", trends.get('proofs_per_sec', {}))
            trend_summary("Proof success rate", trends.get('proof_success_rate', {}))
            trend_summary("P95 latency (ms)", trends.get('p95_latency_ms', {}))
            lines.append(f"- History retention: {trends.get('retention', 0)}")
            lines.append("")

        first_org = metrics.get('first_organism', {})
        if first_org:
            lines.append("## First Organism Vital Signs")
            lines.append("")
            ht = first_org.get('last_ht_hash', '')
            ht_display = ht[:16] + '...' if ht else 'N/A'
            run_timestamp = first_org.get('last_run_timestamp', timestamp)
            last_duration = first_org.get('last_duration_seconds', 0.0)
            avg_duration = first_org.get('average_duration_seconds', 0.0)
            median_duration = first_org.get('median_duration_seconds', 0.0)
            abstentions = first_org.get('abstention_count', 0)
            runs_total = first_org.get('runs_total', 0)
            last_status = first_org.get('last_status', '')
            success_rate = first_org.get('success_rate', 0.0)
            duration_delta = first_org.get('duration_delta', 0.0)
            abstention_delta = first_org.get('abstention_delta', 0)

            # Status emoji/indicator
            status_icon = "ALIVE" if last_status == "success" else "WARN" if last_status == "failure" else "?"

            # Delta formatting
            dur_delta_sign = "+" if duration_delta > 0 else ""
            dur_delta_str = f"{dur_delta_sign}{duration_delta:.2f}s" if duration_delta != 0 else "0s"
            abs_delta_sign = "+" if abstention_delta > 0 else ""
            abs_delta_str = f"{abs_delta_sign}{abstention_delta}" if abstention_delta != 0 else "0"

            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Status | **{status_icon}** |")
            lines.append(f"| Last run | {run_timestamp} |")
            lines.append(f"| H_t hash | `{ht_display}` |")
            lines.append(f"| Total runs | {runs_total} |")
            lines.append(f"| Success rate | {success_rate:.1f}% |")
            lines.append(f"| Duration | {last_duration:.2f}s (delta: {dur_delta_str}) |")
            lines.append(f"| Avg duration | {avg_duration:.2f}s |")
            lines.append(f"| Median duration | {median_duration:.2f}s |")
            lines.append(f"| Abstentions | {abstentions} (delta: {abs_delta_str}) |")
            lines.append("")

            # Add duration history if available
            duration_history = first_org.get('duration_history', [])
            if duration_history and len(duration_history) > 1:
                recent = duration_history[:10]
                history_str = ", ".join(f"{d:.2f}" for d in reversed(recent))
                lines.append(f"**Duration trend** (last {len(recent)} runs): `{history_str}`")
                lines.append("")

        # Seal
        total_entries = sum(len(v) if isinstance(v, dict) else 1 for v in metrics.values())
        variance_ok = variance.get('within_tolerance', True) if variance else True
        epsilon_val = variance.get('epsilon_tolerance', 0.01) if variance else 0.01

        lines.append("## Attestation")
        lines.append("")
        if variance_ok:
            lines.append("```")
            lines.append(f"[PASS] Metrics Canonicalized entries={total_entries} variance<=epsilon={epsilon_val}")
            lines.append("```")
        else:
            cv_val = variance.get('coefficient_of_variation', 0) if variance else 0
            lines.append("```")
            lines.append(f"[WARN] variance={cv_val:.4f} > epsilon={epsilon_val}")
            lines.append("```")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("**Handoffs**:")
        lines.append("- Session JSON -> Codex M (digest)")
        lines.append("- Session JSON -> Codex K (snapshot timeline)")

        return "\n".join(lines)

    def word_count(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())


def main():
    """CLI entry point"""
    project_root = Path(__file__).parent.parent
    metrics_file = project_root / "artifacts" / "metrics" / "latest.json"

    if not metrics_file.exists():
        print(f"[ERROR] Metrics file not found: {metrics_file}")
        print("Run metrics_cartographer.py first")
        return 1

    # Generate report
    generator = MarkdownReportGenerator(metrics_file)
    report = generator.generate_report()

    # Check word count
    word_count = generator.word_count(report)
    if word_count > 400:
        print(f"[WARN] Report exceeds 400 words: {word_count} words")
    else:
        print(f"[OK] Report word count: {word_count}/400 words")

    # Save to reports/metrics_{date}.md
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    date_str = datetime.now().strftime('%Y-%m-%d')
    output_file = reports_dir / f"metrics_{date_str}.md"

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"[OK] Report saved to: {output_file}")
    print()
    print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
