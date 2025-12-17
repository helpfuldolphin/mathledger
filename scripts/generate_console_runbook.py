"""
Generate a markdown runbook from the Global Health Console planning graph.

CI usage example (GitHub Actions)::

    - name: Console runbook drift check
      run: |
        uv run python scripts/generate_console_runbook.py --output docs/GlobalHealthConsoleRunbook.md
        git diff --exit-code docs/GlobalHealthConsoleRunbook.md

The second command should fail the job if the freshly generated runbook drifts
from the committed artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

from analysis.plan.global_console_plan import (
    TileNode,
    build_console_plan,
    dependents_map,
    minimal_deployment_sequence,
    plan_summary,
)


def build_runbook(plan: Mapping[str, TileNode] | None = None) -> str:
    """
    Render the console runbook as a Markdown document.
    """

    source = dict(plan or build_console_plan())
    order = minimal_deployment_sequence(source)
    downstream = dependents_map(source)

    def fmt_sequence(items: Iterable[str]) -> str:
        block = []
        for idx, item in enumerate(items, 1):
            block.append(f"{idx}. {item}")
        return "\n".join(block)

    lines: list[str] = [
        "# Global Health Console Runbook",
        "",
        "## Minimal Deployment Check Sequence",
        "",
        "Follow this ordered checklist during a fresh deployment to ensure each",
        "prerequisite tile is green before unlocking downstream investigations:",
        "",
        fmt_sequence(order),
        "",
        "## Tile-Level Playbooks",
        "",
    ]

    for tile_name in order:
        node = source[tile_name]
        dependencies = node.depends_on
        dependents = downstream[tile_name]

        dep_label = ", ".join(dependencies) if dependencies else "None (root control)"
        dependent_label = ", ".join(dependents) if dependents else "None (terminal)"

        if node.hard_gate:
            gate_text = (
                "Hard gate: downstream signal quality is undefined until this tile is green."
            )
        else:
            gate_text = "Soft gate: downstream checks may proceed, but interpret with caution."

        lines.extend(
            [
                f"### {tile_name}",
                gate_text,
                "",
                f"- Dependencies: {dep_label}",
                f"- Unlocks: {dependent_label}",
                "- If RED, what to check next?",
                f"  1. Inspect the dependencies ({dep_label}) for latent faults.",
            ]
        )

        if dependents:
            lines.append(
                f"  2. If dependencies are green, pivot to dependent tiles ({dependent_label}) "
                "to trace propagation."
            )
        else:
            lines.append("  2. No downstream tiles remain; escalate to the incident channel.")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Global Health Console runbook."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional destination for plan metadata (node/edge counts, missing tiles).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_console_plan()
    payload = build_runbook(plan)

    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)

    if args.summary_json:
        summary_payload = json.dumps(plan_summary(plan), indent=2) + "\n"
        args.summary_json.write_text(summary_payload, encoding="utf-8")


if __name__ == "__main__":
    main()
