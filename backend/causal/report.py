"""
Causal analysis reporting and visualization.

Generates reports and visualizations of causal structures and effects.
"""

from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from backend.repro.determinism import deterministic_timestamp

_GLOBAL_SEED = 0


def generate_causal_report(
    causal_graph: Any,
    coefficients: Dict[tuple, Any],
    run_deltas: List[Any],
    output_path: Optional[str] = None
) -> Dict:
    """
    Generate comprehensive causal analysis report.

    Args:
        causal_graph: Estimated causal graph
        coefficients: Dictionary of estimated causal coefficients
        run_deltas: List of RunDelta objects from historical runs
        output_path: Optional file path to save JSON report

    Returns:
        Report dictionary
    """
    report = {
        'schema': 'causal_report_v1',
        'generated_at': deterministic_timestamp(_GLOBAL_SEED).isoformat(),
        'graph_structure': _summarize_graph(causal_graph),
        'causal_effects': _summarize_coefficients(coefficients),
        'empirical_deltas': _summarize_deltas(run_deltas),
        'key_findings': _extract_key_findings(coefficients, run_deltas),
        'validation': _validation_summary(causal_graph, coefficients)
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    return report


def _summarize_graph(causal_graph: Any) -> Dict:
    """Summarize causal graph structure."""
    return {
        'n_nodes': len(causal_graph.nodes),
        'n_edges': len(causal_graph.edges),
        'nodes': list(causal_graph.nodes.keys()),
        'edges': [
            {
                'source': edge.source.name,
                'target': edge.target.name,
                'mechanism': edge.mechanism
            }
            for edge in causal_graph.edges
        ],
        'topological_order': causal_graph.topological_sort(),
        'graph_hash': causal_graph.hash()
    }


def _summarize_coefficients(coefficients: Dict[tuple, Any]) -> List[Dict]:
    """Summarize causal coefficients."""
    return [
        {
            'edge': f"{source} → {target}",
            **coef.to_dict()
        }
        for (source, target), coef in coefficients.items()
    ]


def _summarize_deltas(run_deltas: List[Any]) -> Dict:
    """Summarize empirical run deltas."""
    if not run_deltas:
        return {
            'n_deltas': 0,
            'policy_changes': 0,
            'mean_deltas': {}
        }

    from backend.causal.variables import compute_mean_deltas, stratify_by_policy_change

    # Stratify by policy change
    policy_changed, policy_unchanged = stratify_by_policy_change(run_deltas)

    return {
        'n_deltas': len(run_deltas),
        'policy_changes': len(policy_changed),
        'policy_unchanged': len(policy_unchanged),
        'all_runs': compute_mean_deltas(run_deltas),
        'policy_changed_only': compute_mean_deltas(policy_changed),
        'policy_unchanged_only': compute_mean_deltas(policy_unchanged)
    }


def _extract_key_findings(coefficients: Dict[tuple, Any], run_deltas: List[Any]) -> List[str]:
    """Extract key findings from causal analysis."""
    findings = []

    # Find strongest causal effects
    if coefficients:
        sorted_coefs = sorted(
            coefficients.items(),
            key=lambda x: abs(x[1].coefficient),
            reverse=True
        )

        for (source, target), coef in sorted_coefs[:3]:
            if coef.is_significant:
                direction = "increases" if coef.coefficient > 0 else "decreases"
                findings.append(
                    f"{source} significantly {direction} {target} "
                    f"(β={coef.coefficient:.3f}, p={coef.p_value:.4f})"
                )

    # Policy update insights
    if run_deltas:
        from backend.causal.variables import stratify_by_policy_change, compute_mean_deltas

        policy_changed, _ = stratify_by_policy_change(run_deltas)
        if policy_changed:
            stats = compute_mean_deltas(policy_changed)
            findings.append(
                f"Policy updates associated with "
                f"{stats['mean_delta_throughput']:.2f} proofs/sec change "
                f"and {stats['mean_delta_abstain']:.1f}pp abstention change"
            )

    if not findings:
        findings.append("Insufficient data for significant findings")

    return findings


def _validation_summary(causal_graph: Any, coefficients: Dict[tuple, Any]) -> Dict:
    """Summarize validation checks."""
    from backend.causal.graph import validate_graph

    warnings = validate_graph(causal_graph)

    # Count significant effects
    n_significant = sum(1 for coef in coefficients.values() if coef.is_significant)
    n_total = len(coefficients)

    return {
        'structure_valid': len(warnings) == 0,
        'warnings': warnings,
        'n_estimated_edges': n_total,
        'n_significant_edges': n_significant,
        'significance_rate': n_significant / n_total if n_total > 0 else 0.0
    }


def format_causal_graph_ascii(causal_graph: Any, coefficients: Dict[tuple, Any]) -> str:
    """
    Format causal graph as ASCII art.

    Returns:
        Multi-line string with graph visualization
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CAUSAL GRAPH")
    lines.append("=" * 60)
    lines.append("")

    # Sort edges by topological order of source
    try:
        topo_order = causal_graph.topological_sort()
        order_map = {node: i for i, node in enumerate(topo_order)}

        sorted_edges = sorted(
            causal_graph.edges,
            key=lambda e: order_map.get(e.source.name, 999)
        )
    except:
        sorted_edges = list(causal_graph.edges)

    # Display each edge
    for edge in sorted_edges:
        source = edge.source.name
        target = edge.target.name

        # Get coefficient if available
        coef = coefficients.get((source, target))

        if coef:
            sig = "***" if coef.p_value < 0.001 else (
                  "**" if coef.p_value < 0.01 else (
                  "*" if coef.p_value < 0.05 else ""))

            arrow = "──→" if coef.coefficient > 0 else "──↓"
            coef_str = f"β={coef.coefficient:.3f}{sig}"

            lines.append(f"  {source:20} {arrow} {target:20}  {coef_str}")
        else:
            lines.append(f"  {source:20} ──→ {target:20}  [not estimated]")

    lines.append("")
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_pass_summary(coefficients: Dict[tuple, Any]) -> List[str]:
    """
    Format causal coefficients as [PASS] messages.

    Returns:
        List of formatted pass messages
    """
    from backend.causal.estimator import format_pass_message

    messages = []

    for (source, target), coef in coefficients.items():
        msg = format_pass_message(source, target, coef)
        messages.append(msg)

    return messages


def export_for_dagitty(causal_graph: Any) -> str:
    """
    Export causal graph in DAGitty format for visualization.

    DAGitty format:
    dag {
      X -> Y
      Z -> X
      Z -> Y
    }

    Returns:
        DAGitty-formatted string
    """
    lines = ["dag {"]

    for edge in causal_graph.edges:
        lines.append(f"  {edge.source.name} -> {edge.target.name}")

    lines.append("}")

    return "\n".join(lines)


def export_for_graphviz(
    causal_graph: Any,
    coefficients: Dict[tuple, Any],
    output_path: str
) -> None:
    """
    Export causal graph as Graphviz DOT file.

    Args:
        causal_graph: Causal graph
        coefficients: Estimated coefficients
        output_path: Path to save .dot file
    """
    lines = ["digraph CausalGraph {"]
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box, style=rounded];")
    lines.append("")

    # Add nodes with colors by type
    for name, node in causal_graph.nodes.items():
        color = {
            'policy': 'lightblue',
            'abstention': 'lightyellow',
            'throughput': 'lightgreen',
            'verification_time': 'lightpink',
            'depth': 'lightgray'
        }.get(node.var_type.value, 'white')

        lines.append(f'  "{name}" [fillcolor={color}, style=filled];')

    lines.append("")

    # Add edges with coefficients as labels
    for edge in causal_graph.edges:
        source = edge.source.name
        target = edge.target.name

        coef = coefficients.get((source, target))

        if coef and coef.is_significant:
            label = f"{coef.coefficient:.3f}"
            color = "green" if coef.coefficient > 0 else "red"
            weight = min(abs(coef.coefficient) * 2, 5)

            lines.append(
                f'  "{source}" -> "{target}" '
                f'[label="{label}", color={color}, penwidth={weight}];'
            )
        else:
            lines.append(f'  "{source}" -> "{target}" [style=dashed];')

    lines.append("}")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def generate_markdown_summary(
    causal_graph: Any,
    coefficients: Dict[tuple, Any],
    run_deltas: List[Any]
) -> str:
    """
    Generate markdown-formatted summary report.

    Returns:
        Markdown string
    """
    lines = []

    lines.append("# Causal Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {deterministic_timestamp(_GLOBAL_SEED).isoformat()}")
    lines.append("")

    # Graph structure
    lines.append("## Causal Graph Structure")
    lines.append("")
    lines.append(f"- **Nodes:** {len(causal_graph.nodes)}")
    lines.append(f"- **Edges:** {len(causal_graph.edges)}")
    lines.append("")

    # Causal effects table
    lines.append("## Estimated Causal Effects")
    lines.append("")
    lines.append("| Source | Target | Coefficient | Std Error | p-value | Significant |")
    lines.append("|--------|--------|-------------|-----------|---------|-------------|")

    for (source, target), coef in coefficients.items():
        sig = "✓" if coef.is_significant else ""
        lines.append(
            f"| {source} | {target} | {coef.coefficient:.3f} | "
            f"{coef.std_error:.3f} | {coef.p_value:.4f} | {sig} |"
        )

    lines.append("")

    # Key findings
    findings = _extract_key_findings(coefficients, run_deltas)
    lines.append("## Key Findings")
    lines.append("")
    for finding in findings:
        lines.append(f"- {finding}")

    lines.append("")

    # Empirical deltas
    if run_deltas:
        from backend.causal.variables import compute_mean_deltas

        stats = compute_mean_deltas(run_deltas)
        lines.append("## Empirical Deltas (Mean)")
        lines.append("")
        lines.append(f"- **Δ Abstention:** {stats['mean_delta_abstain']:.2f} pp")
        lines.append(f"- **Δ Throughput:** {stats['mean_delta_throughput']:.3f} proofs/sec")
        lines.append(f"- **Δ Proofs:** {stats['mean_delta_proof']:.1f}")
        lines.append(f"- **N:** {stats['n_deltas']}")
        lines.append("")

    return "\n".join(lines)


def print_causal_summary(
    causal_graph: Any,
    coefficients: Dict[tuple, Any],
    run_deltas: List[Any]
) -> None:
    """
    Print causal analysis summary to stdout.

    Args:
        causal_graph: Causal graph
        coefficients: Estimated coefficients
        run_deltas: Empirical run deltas
    """
    # Print ASCII graph
    print(format_causal_graph_ascii(causal_graph, coefficients))
    print()

    # Print key findings
    print("KEY FINDINGS")
    print("=" * 60)
    findings = _extract_key_findings(coefficients, run_deltas)
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")
    print()

    # Print pass messages
    print("CAUSAL MODEL STATUS")
    print("=" * 60)
    for msg in format_pass_summary(coefficients):
        print(msg)
    print()
