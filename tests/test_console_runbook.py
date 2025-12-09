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
