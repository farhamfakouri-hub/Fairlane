"""The roster gate: many small scenarios that must all produce a legal roster."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

from ree.planner.config import PlannerConfig
from ree.planner.demand.forecast import default_model
from ree.planner.domain.demand_types import PlannerCalendar, RequirementTable
from ree.planner.domain.plan import WorkerRecord
from ree.planner.roster.fairness import hours_deviation
from ree.planner.roster.model import ScoreInputs
from ree.planner.roster.solver import RosterSolution, plan_roster
from ree.planner.sim.scenarios import make_workforce, random_scores, store_config
from ree.planner.staffing.requirement import requirement_for_cycle

GATE_SCENARIOS = 50
GATE_FIRST_SEED = 100
GATE_STORE = "tiny"
GATE_MEAN_SHORTFALL_LIMIT = 120.0


def scenario_inputs(
    seed: int,
    config: PlannerConfig | None = None,
    weeks: int = 1,
    store: str = GATE_STORE,
) -> tuple[PlannerConfig, list[WorkerRecord], RequirementTable, ScoreInputs]:
    """Configuration, workers, requirement and scores of one gate scenario."""
    used = config if config is not None else store_config(store)
    calendar = PlannerCalendar(used)
    start = date.fromisoformat(used.start_date)
    requirement = requirement_for_cycle(default_model(calendar, used), calendar, used, start, weeks)
    workers = make_workforce(used, seed=seed, weeks=weeks)
    scores = random_scores([w.worker_id for w in workers], seed, used)
    return used, workers, requirement, scores


def solve_scenario(
    seed: int,
    config: PlannerConfig | None = None,
    weeks: int = 1,
    store: str = GATE_STORE,
) -> RosterSolution:
    """Solve the roster of one gate scenario."""
    used, workers, requirement, scores = scenario_inputs(seed, config, weeks, store)
    return plan_roster(workers, requirement, used, scores, weeks=weeks)


def summarise_scenario(seed: int) -> dict[str, object]:
    """Solve one scenario and summarise its checks, shortfall and fairness."""
    started = time.perf_counter()
    solution = solve_scenario(seed)
    report = solution.report
    assert report is not None
    return {
        "seed": seed,
        "status": solution.status,
        "backend": solution.backend,
        "errors": [item.code for item in report.errors],
        "warnings": len(report.warnings),
        "shortfall_minutes": report.shortfall_minutes(),
        "max_hours_deviation": max(hours_deviation(solution.plan).values()),
        "max_tier_gap": solution.metrics.max_tier_gap if solution.metrics else 0.0,
        "relaxed": solution.relaxed,
        "objective": solution.objective,
        "seconds": round(time.perf_counter() - started, 1),
        "assignments": sorted(
            (a.worker_id, a.day.isoformat(), a.pattern.pattern_id)
            for a in solution.plan.assignments
        ),
    }


def default_processes() -> int:
    """Number of worker processes to use by default."""
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def run_gate(
    scenarios: int = GATE_SCENARIOS,
    first_seed: int = GATE_FIRST_SEED,
    processes: int | None = None,
) -> dict[str, object]:
    """Run the gate over many seeds and summarise the results."""
    seeds = list(range(first_seed, first_seed + scenarios))
    started = time.perf_counter()
    workers = processes if processes is not None else default_processes()
    if workers <= 1:
        rows = [summarise_scenario(seed) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(summarise_scenario, seeds))
    failures = [row for row in rows if row["errors"] or row["relaxed"]]
    mean_shortfall = sum(float(row["shortfall_minutes"]) for row in rows) / len(rows)
    max_deviation = max(float(row["max_hours_deviation"]) for row in rows)
    max_gap = max(float(row["max_tier_gap"]) for row in rows)
    return {
        "store": GATE_STORE,
        "scenarios": len(rows),
        "first_seed": first_seed,
        "legal_rosters": len(rows) - len(failures),
        "failed_seeds": [row["seed"] for row in failures],
        "mean_shortfall_minutes": round(mean_shortfall, 1),
        "max_hours_deviation": round(max_deviation, 3),
        "max_tier_gap": round(max_gap, 3),
        "all_solved": all(row["status"] in ("OPTIMAL", "FEASIBLE") for row in rows),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "rows": [
            {key: value for key, value in row.items() if key != "assignments"} for row in rows
        ],
    }


def gate_passed(summary: dict[str, object], config: PlannerConfig | None = None) -> bool:
    """Whether every roster was legal, solved and within the hours tolerance and shortfall limit."""
    used = config if config is not None else PlannerConfig()
    return (
        not summary["failed_seeds"]
        and bool(summary["all_solved"])
        and float(summary["max_hours_deviation"]) <= used.rules.hours_tolerance_hours
        and float(summary["mean_shortfall_minutes"]) < GATE_MEAN_SHORTFALL_LIMIT
    )


def write_gate_report(summary: dict[str, object], path: Path) -> None:
    """Write the gate summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
