from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANNER = ROOT / "src" / "ree" / "planner"


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def offenders(folder: str, forbidden: tuple[str, ...]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted((PLANNER / folder).glob("*.py")):
        bad = sorted(m for m in imported_modules(path) if m.startswith(forbidden))
        if bad:
            found[path.name] = bad
    return found


def test_the_checker_never_imports_a_solver_or_the_daily_planner() -> None:
    forbidden = (
        "ree.planner.roster",
        "ree.planner.daily",
        "ree.planner.live",
        "ree.planner.sim",
        "ree.planner.eval",
        "ortools",
    )
    assert offenders("checker", forbidden) == {}


def test_the_domain_and_demand_layers_do_not_depend_on_solvers_or_the_checker() -> None:
    forbidden = (
        "ree.planner.roster",
        "ree.planner.daily",
        "ree.planner.live",
        "ree.planner.sim",
        "ree.planner.eval",
        "ree.planner.checker",
        "ortools",
    )
    assert offenders("domain", forbidden) == {}
    assert offenders("demand", forbidden) == {}
    assert offenders("staffing", forbidden) == {}


def test_the_solvers_and_live_planning_never_see_hidden_truth_or_the_simulator() -> None:
    forbidden = (
        "ree.planner.sim",
        "ree.planner.eval",
        "ree.simulator",
        "ree.storage",
        "ree.engine",
        "ree.experiments",
    )
    for folder in ("roster", "daily", "live"):
        if (PLANNER / folder).exists():
            assert offenders(folder, forbidden) == {}, folder


def test_the_planner_reads_part_one_only_through_the_score_interface() -> None:
    allowed = (
        "ree.api.probe",
        "ree.api.provider",
        "ree.config",
        "ree.domain.models",
        "ree.planner",
    )
    for path in sorted(PLANNER.rglob("*.py")):
        relative = path.relative_to(PLANNER).parts
        if relative[0] in ("sim", "eval"):
            continue
        outside = sorted(
            module
            for module in imported_modules(path)
            if module.startswith("ree.") and not module.startswith(allowed)
        )
        assert outside == [], (str(path.relative_to(ROOT)), outside)
