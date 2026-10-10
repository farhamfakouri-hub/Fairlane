"""The plans compared in the evaluation, A to F, and how each is built."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date

from ree.planner.config import PlannerConfig
from ree.planner.daily.daily_plan import DailyResult, plan_days
from ree.planner.demand.forecast import default_model
from ree.planner.domain.demand_types import PlannerCalendar, RequirementTable
from ree.planner.domain.plan import WorkerRecord
from ree.planner.roster.model import CoverageStart, ScoreInputs, neutral_scores, score_inputs
from ree.planner.roster.solver import RosterSolution, plan_roster, solve_coverage
from ree.planner.sim.part1_workforce import (
    Part1Workforce,
    oracle_inputs,
    part1_workforce,
    shuffled_inputs,
)
from ree.planner.sim.scenarios import make_workforce
from ree.planner.staffing.requirement import requirement_for_cycle

PLAN_NAMES = ("A", "B", "C", "D", "E")
POSTS_ONLY_PLAN = "F"
EVALUATED_PLANS = (*PLAN_NAMES, POSTS_ONLY_PLAN)
LIVE_PLAN = "L"
ALL_PLANS = (*PLAN_NAMES, LIVE_PLAN)
PLAN_LABELS = {
    "A": "A headcount only",
    "B": "B demand-aware",
    "C": "C full planner",
    "D": "D oracle",
    "E": "E placebo",
    "F": "F scores in the posts only",
    "L": "L full planner with live re-planning",
}


@dataclass
class PlanInputs:
    """Everything the plans of one seed are built from: workers, requirement and scores."""

    config: PlannerConfig
    seed: int
    workers: list[WorkerRecord]
    workforce: Part1Workforce
    requirement: RequirementTable
    flat: RequirementTable
    scores: ScoreInputs
    oracle: ScoreInputs
    placebo: ScoreInputs
    coverage: dict[str, CoverageStart | None] = field(default_factory=dict)


@dataclass
class PlanBundle:
    """A plan: its requirement, scores, roster and daily plans."""

    name: str
    requirement: RequirementTable
    scores: ScoreInputs
    roster: RosterSolution
    daily: dict[date, DailyResult]


def restrict_scores(scores: ScoreInputs, ids: set[str]) -> ScoreInputs:
    """Score inputs restricted to some workers."""
    return ScoreInputs(
        z={w: v for w, v in scores.z.items() if w in ids},
        tiers={w: v for w, v in scores.tiers.items() if w in ids},
        exploration=frozenset(w for w in scores.exploration if w in ids),
    )


def flat_requirement(table: RequirementTable) -> RequirementTable:
    """Requirement with the same number of registers in every rush and every calm slot."""
    rush_levels = [
        row.registers for day in table.dates() for _, row in table.slots(day) if row.rush
    ]
    calm_levels = [
        row.registers for day in table.dates() for _, row in table.slots(day) if not row.rush
    ]
    rush_level = round(sum(rush_levels) / len(rush_levels)) if rush_levels else 2
    calm_level = round(sum(calm_levels) / len(calm_levels)) if calm_levels else 2
    flat = RequirementTable(table.slot_minutes, opening_order=table.opening_order)
    for day in table.dates():
        flat.days[day] = tuple(
            (
                start,
                dataclasses.replace(
                    row,
                    registers=rush_level if row.rush else calm_level,
                    headcount=(rush_level if row.rush else calm_level) + row.k_sc + row.floaters,
                ),
            )
            for start, row in table.slots(day)
        )
    return flat


def build_inputs(
    config: PlannerConfig, seed: int, team_share: float = 1.0, weeks: int = 1
) -> PlanInputs:
    """Build the inputs of a seed: a Part 1 workforce, workers, requirement and scores."""
    count = config.workforce.n_workers
    workforce = part1_workforce(config, seed, count)
    ids = [f"W{index:03d}" for index in range(1, count + 1)]
    scores = score_inputs(workforce.provider, ids, config)
    workers = make_workforce(config, seed=seed, weeks=weeks)
    if team_share < 1.0:
        keep = max(1, round(team_share * count))
        workers = workers[:keep]
        scores = restrict_scores(scores, {w.worker_id for w in workers})
    kept = {w.worker_id for w in workers}
    oracle = restrict_scores(oracle_inputs(workforce.truth, config), kept)
    calendar = PlannerCalendar(config)
    start = date.fromisoformat(config.start_date)
    requirement = requirement_for_cycle(
        default_model(calendar, config), calendar, config, start, weeks
    )
    return PlanInputs(
        config=config,
        seed=seed,
        workers=workers,
        workforce=workforce,
        requirement=requirement,
        flat=flat_requirement(requirement),
        scores=scores,
        oracle=oracle,
        placebo=shuffled_inputs(scores, seed),
    )


def plan_specification(inputs: PlanInputs, name: str) -> tuple[RequirementTable, ScoreInputs]:
    """Requirement and scores that a named plan uses."""
    neutral = neutral_scores([w.worker_id for w in inputs.workers])
    if name == "A":
        return inputs.flat, neutral
    if name == "B":
        return inputs.requirement, neutral
    if name == "C":
        return inputs.requirement, inputs.scores
    if name == "D":
        return inputs.requirement, inputs.oracle
    if name == "E":
        return inputs.requirement, inputs.placebo
    raise ValueError(f"unknown plan {name!r}")


def build_posts_only_plan(
    inputs: PlanInputs, weeks: int = 1, base: PlanBundle | None = None
) -> PlanBundle:
    """Plan F: the roster of plan B, with scores used only to choose the posts."""
    neutral = base if base is not None else build_plan(inputs, "B", weeks)
    daily = plan_days(neutral.roster.plan, neutral.requirement, inputs.config, inputs.scores)
    return PlanBundle(POSTS_ONLY_PLAN, neutral.requirement, inputs.scores, neutral.roster, daily)


def build_plan(
    inputs: PlanInputs, name: str, weeks: int = 1, base: PlanBundle | None = None
) -> PlanBundle:
    """Solve the roster and the daily plans of a named plan."""
    if name == POSTS_ONLY_PLAN:
        return build_posts_only_plan(inputs, weeks, base)
    requirement, scores = plan_specification(inputs, name)
    key = "flat" if name == "A" else "forecast"
    config = inputs.config
    if config.solver.coverage_first and key not in inputs.coverage:
        inputs.coverage[key] = solve_coverage(inputs.workers, requirement, config, weeks=weeks)
    roster = plan_roster(
        inputs.workers,
        requirement,
        config,
        scores,
        weeks=weeks,
        coverage_start=inputs.coverage.get(key),
    )
    daily = plan_days(roster.plan, requirement, config, scores)
    return PlanBundle(name, requirement, scores, roster, daily)
