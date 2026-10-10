"""Stability of the roster: how many assignments change when it is re-solved after a change."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from ree.config import make_rng
from ree.planner.config import PlannerConfig
from ree.planner.domain.plan import RosterPlan, WorkerRecord
from ree.planner.eval.stats import bootstrap_mean
from ree.planner.roster.solver import plan_roster
from ree.planner.sim.roster_gate import scenario_inputs

STABILITY_SEEDS = 10
FIRST_STABILITY_SEED = 100
BLOCKED_WORKER_DAYS = 2
FULL_DAY = (0, 1440)
STAY_CLOSE_SHARE = 0.25
REPORT_FILE = "stability.md"
TABLE_FILE = "stability.csv"


def keyed(plan: RosterPlan) -> dict[tuple[str, date], str]:
    """Pattern identifier of each worker and day."""
    return {(a.worker_id, a.day): a.pattern.pattern_id for a in plan.assignments}


def changed_assignments(
    before: dict[tuple[str, date], str], after: dict[tuple[str, date], str]
) -> int:
    """Number of worker-days whose pattern differs between two rosters."""
    return sum(1 for key in set(before) | set(after) if before.get(key) != after.get(key))


def block_worker_days(
    workers: Sequence[WorkerRecord], plan: RosterPlan, seed: int
) -> tuple[list[WorkerRecord], list[tuple[str, date]]]:
    """Block a few worker-days of a roster, for the re-solve test."""
    rows = sorted({(a.worker_id, a.day) for a in plan.assignments})
    picks = make_rng(seed, "planner-stability").choice(
        len(rows), size=min(BLOCKED_WORKER_DAYS, len(rows)), replace=False
    )
    chosen = [rows[int(index)] for index in picks]
    updated: list[WorkerRecord] = []
    for worker in workers:
        days = [day for worker_id, day in chosen if worker_id == worker.worker_id]
        if days:
            blocked = dict(worker.blocked)
            for day in days:
                blocked[day] = (FULL_DAY,)
            worker = dataclasses.replace(worker, blocked=blocked)
        updated.append(worker)
    return updated, chosen


def stability_seed(config: PlannerConfig, seed: int) -> dict[str, object]:
    """Solve, block some worker-days, re-solve with and without a stay-close term."""
    used, workers, requirement, scores = scenario_inputs(seed, config)
    first = plan_roster(workers, requirement, used, scores, weeks=1)
    before = keyed(first.plan)
    changed_workers, chosen = block_worker_days(workers, first.plan, seed)
    with_reference = plan_roster(
        changed_workers, requirement, used, scores, weeks=1, last_roster=before
    )
    without_reference = plan_roster(changed_workers, requirement, used, scores, weeks=1)
    assert with_reference.report is not None and without_reference.report is not None
    return {
        "seed": seed,
        "assignments": float(len(before)),
        "blocked_worker_days": float(len(chosen)),
        "changed_with_reference": float(changed_assignments(before, keyed(with_reference.plan))),
        "changed_without_reference": float(
            changed_assignments(before, keyed(without_reference.plan))
        ),
        "shortfall_with_reference": float(with_reference.report.shortfall_minutes()),
        "shortfall_without_reference": float(without_reference.report.shortfall_minutes()),
        "errors_with_reference": float(len(with_reference.report.errors)),
        "errors_without_reference": float(len(without_reference.report.errors)),
    }


def _job(job: tuple[PlannerConfig, int]) -> dict[str, object]:
    return stability_seed(*job)


def run_stability(
    config: PlannerConfig, seeds: Sequence[int], jobs: int = 2, progress: bool = False
) -> pd.DataFrame:
    """Run the stability study over seeds, in parallel if asked."""
    work = [(config, seed) for seed in seeds]
    rows: list[dict[str, object]] = []
    if jobs <= 1 or len(work) == 1:
        for done, item in enumerate(work, start=1):
            rows.append(_job(item))
            if progress:
                print(f"stability seed {item[1]} done ({done}/{len(work)})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for done, row in enumerate(pool.map(_job, work), start=1):
                rows.append(row)
                if progress:
                    print(f"stability {done}/{len(work)} seeds done", flush=True)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("assignments", "Assignments in the first roster", 1),
    ("changed_with_reference", "Changed, previous roster as reference", 1),
    ("changed_without_reference", "Changed, no reference", 1),
    ("shortfall_with_reference", "Uncovered staff-minutes, with reference", 1),
    ("shortfall_without_reference", "Uncovered staff-minutes, no reference", 1),
    ("errors_with_reference", "Checker errors, with reference", 1),
    ("errors_without_reference", "Checker errors, no reference", 1),
)


def stability_statement(frame: pd.DataFrame, config: PlannerConfig) -> str:
    """One paragraph with the result."""
    seeds = float(len(frame))
    total = float(frame["assignments"].sum())
    with_reference = float(frame["changed_with_reference"].sum())
    without_reference = float(frame["changed_without_reference"].sum())
    weight = config.objective.stability_weight
    if with_reference < without_reference:
        verdict = (
            f"The reference removes {without_reference - with_reference:.0f} of "
            f"{without_reference:.0f} changes "
            f"({(without_reference - with_reference) / without_reference:.0%})."
        )
    else:
        verdict = "The reference does not reduce the number of changes."
    if with_reference / total > STAY_CLOSE_SHARE:
        closeness = (
            "A repair that moved only the blocked shifts and the few that take them over would "
            "change a handful of assignments, so at this search budget the solver does not stay "
            "close to the previous roster; a repair mode that fixes everything outside a "
            "neighbourhood of the change is not built."
        )
    else:
        closeness = "The solver stays close to the previous roster."
    return (
        f"Blocking {BLOCKED_WORKER_DAYS} worker-days of a solved one-week roster of "
        f"{total / seeds:.0f} assignments and solving again changes "
        f"{with_reference / seeds:.0f} assignments on average when the previous roster is given "
        f"as the stability reference (a priced term of {weight:g} per changed assignment) and "
        f"{without_reference / seeds:.0f} when it is not; a shift that moves to another day "
        f"counts twice. {verdict} {closeness}"
    )


def stability_report(frame: pd.DataFrame, config: PlannerConfig) -> str:
    """Markdown text of the stability study."""
    header = ["Measure", "Mean over seeds [95 percent interval]"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for column, label, digits in COLUMNS:
        mean, low, high = bootstrap_mean(frame[column].to_list())
        lines.append(f"| {label} | {mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}] |")
    return "\n".join(
        [
            "# Roster stability after a change",
            "",
            "Generated by `python -m ree.cli planner stability`. Each seed solves a one-week "
            "roster of the 10-worker store, blocks two of its worker-days (the whole day, "
            "chosen at random from the shifts of that roster), and solves again twice: with "
            "the first roster as the stability reference and without it.",
            "",
            f"Seeds: {frame['seed'].nunique()}.",
            "",
            *lines,
            "",
            stability_statement(frame, config),
            "",
        ]
    )


def stability_table_path(config: PlannerConfig, reduced: bool) -> Path:
    """Path of the stability table."""
    root = Path(config.paths.outputs_dir)
    return root / ("reduced" if reduced else "reports") / TABLE_FILE


def write_stability(frame: pd.DataFrame, config: PlannerConfig, reduced: bool) -> list[Path]:
    """Write the stability table and report."""
    table = stability_table_path(config, reduced)
    table.parent.mkdir(parents=True, exist_ok=True)
    report = table.parent / REPORT_FILE
    frame.to_csv(table, index=False)
    report.write_text(stability_report(frame, config), encoding="utf-8")
    return [table, report]
