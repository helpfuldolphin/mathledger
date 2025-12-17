from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from analysis.plan.global_console_plan import (
    build_console_plan,
    minimal_deployment_sequence,
)
from scripts.generate_console_runbook import build_runbook


def test_runbook_covers_tiles_in_order() -> None:
    plan = build_console_plan()
    runbook = build_runbook(plan)

    for tile_name in plan:
        assert f"### {tile_name}" in runbook

    section_order = [
        line.replace("### ", "")
        for line in runbook.splitlines()
        if line.startswith("### ")
    ]
    assert tuple(section_order) == minimal_deployment_sequence(plan)


def test_runbook_drift_detects_missing_tile(tmp_path: Path) -> None:
    plan = build_console_plan()
    broken_plan = {name: node for name, node in plan.items() if name != "Security"}

    # Update Conjectures node to drop the missing dependency to keep the plan consistent
    broken_plan["Conjectures"] = replace(
        broken_plan["Conjectures"],
        depends_on=tuple(
            dependency
            for dependency in broken_plan["Conjectures"].depends_on
            if dependency != "Security"
        ),
    )

    mutated_runbook = build_runbook(broken_plan)
    tmp_file = tmp_path / "mutated_runbook.md"
    tmp_file.write_text(mutated_runbook, encoding="utf-8")

    canonical_runbook = Path("docs/GlobalHealthConsoleRunbook.md").read_text(
        encoding="utf-8"
    )
    assert canonical_runbook != mutated_runbook
