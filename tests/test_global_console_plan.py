from analysis.plan.global_console_plan import (
    REQUIRED_TILES,
    build_console_plan,
    plan_summary,
    topological_order,
)


def test_console_plan_is_acyclic() -> None:
    plan = build_console_plan()
    order = topological_order(plan)

    assert set(order) == set(plan)

    rank = {name: idx for idx, name in enumerate(order)}
    for node in plan.values():
        for dependency in node.depends_on:
            assert (
                rank[dependency] < rank[node.name]
            ), f"{dependency} should precede {node.name}"


def test_console_plan_contains_required_tiles() -> None:
    plan = build_console_plan()
    assert set(REQUIRED_TILES).issubset(plan.keys())


def test_plan_summary_reports_no_missing_required_tiles() -> None:
    summary = plan_summary()
    assert summary["missing_required_tiles"] == ()
