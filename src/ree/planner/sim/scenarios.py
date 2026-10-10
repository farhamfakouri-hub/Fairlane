"""Synthetic workforces, availability, scores and store sizes for tests and the evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

from ree.planner.config import PlannerConfig
from ree.planner.domain.plan import WorkerRecord
from ree.planner.roster.model import UNRATED_TIER, ScoreInputs, clipped_z


def worker_ids(count: int) -> list[str]:
    """Worker ids W001, W002 and so on."""
    return [f"W{index:03d}" for index in range(1, count + 1)]


def contract_hours_for(
    rng: np.random.Generator, config: PlannerConfig, count: int
) -> tuple[list[float], list[str]]:
    """Contract type and weekly hours of each worker, drawn from the configured pools."""
    workforce = config.workforce
    student_count = round(workforce.student_share * count)
    kinds = ["student"] * student_count + ["full_time"] * (count - student_count)
    order = rng.permutation(count)
    hours: list[float] = []
    shuffled = [kinds[i] for i in order]
    for kind in shuffled:
        pool = workforce.student_hours if kind == "student" else workforce.full_time_hours
        hours.append(float(pool[int(rng.integers(len(pool)))]))
    return hours, shuffled


def availability_for(
    rng: np.random.Generator,
    config: PlannerConfig,
    contract_type: str,
    start: date,
    weeks: int,
) -> tuple[dict[date, tuple[tuple[int, int], ...]], dict[date, tuple[tuple[int, int], ...]]]:
    """Days or parts of days when a worker cannot work, and possible absences."""
    workforce = config.workforce
    rate = (
        workforce.blocked_days_per_week_students
        if contract_type == "student"
        else workforce.blocked_days_per_week_full_time
    )
    cap = 3 if contract_type == "student" else 1
    blocked: dict[date, tuple[tuple[int, int], ...]] = {}
    soft: dict[date, tuple[tuple[int, int], ...]] = {}
    for week in range(weeks):
        count = min(int(rng.poisson(rate)), cap)
        offsets = rng.choice(7, size=count, replace=False) if count else []
        for offset in offsets:
            day = start + timedelta(days=7 * week + int(offset))
            kind = int(rng.integers(3))
            if kind == 0:
                blocked[day] = ((0, 1440),)
            elif kind == 1:
                blocked[day] = ((0, 13 * 60),)
            else:
                blocked[day] = ((13 * 60, 1440),)
        if rng.random() < 0.15:
            day = start + timedelta(days=7 * week + int(rng.integers(7)))
            if day not in blocked:
                soft[day] = ((8 * 60, 14 * 60),)
    return blocked, soft


def make_workforce(
    config: PlannerConfig,
    seed: int,
    count: int | None = None,
    weeks: int | None = None,
    start: date | None = None,
    with_availability: bool = True,
) -> list[WorkerRecord]:
    """Build the workers of a store, with contracts, self-checkout abilities and availability."""
    from ree.config import make_rng

    rng = make_rng(seed, "planner-workforce")
    size = count if count is not None else config.workforce.n_workers
    horizon = weeks if weeks is not None else config.rules.planning_weeks
    first = start if start is not None else date.fromisoformat(config.start_date)
    ids = worker_ids(size)
    hours, kinds = contract_hours_for(rng, config, size)
    only_count = round(config.workforce.self_checkout_only_share * size)
    preferred_count = round(config.workforce.preferred_self_checkout_share * size)
    cleared_count = round(config.workforce.cleared_self_checkout_share * size)
    permutation = [int(i) for i in rng.permutation(size)]
    only = set(permutation[:only_count])
    preferred = set(permutation[only_count : only_count + preferred_count])
    cleared = set(permutation[:cleared_count]) | only | preferred
    workers: list[WorkerRecord] = []
    for index, worker_id in enumerate(ids):
        blocked, soft = (
            availability_for(rng, config, kinds[index], first, horizon)
            if with_availability
            else ({}, {})
        )
        workers.append(
            WorkerRecord(
                worker_id=worker_id,
                contract_hours=hours[index],
                contract_type=kinds[index],
                self_checkout_only=index in only,
                preferred_self_checkout=index in preferred,
                cleared_self_checkout=index in cleared,
                blocked=blocked,
                soft_blocked=soft,
            )
        )
    return workers


def random_scores(
    ids: Sequence[str],
    seed: int,
    config: PlannerConfig,
    unrated_share: float = 0.1,
) -> ScoreInputs:
    """Random score inputs, with a share of workers left unrated."""
    from ree.config import make_rng

    rng = make_rng(seed, "planner-scores")
    count = len(ids)
    raw = rng.normal(50.0, 15.0, count)
    unrated_count = round(unrated_share * count)
    unrated_index = {int(i) for i in rng.permutation(count)[:unrated_count]}
    rated = [i for i in range(count) if i not in unrated_index]
    ranking = sorted(rated, key=lambda i: raw[i])
    low_cut = max(1, round(0.2 * len(ranking)))
    z: dict[str, float] = {}
    tiers: dict[str, str] = {}
    for position, index in enumerate(ranking):
        tier = "C" if position < low_cut else "A" if position >= len(ranking) - low_cut else "B"
        tiers[ids[index]] = tier
        z[ids[index]] = clipped_z(float(raw[index]), config.objective.score_clip)
    for index in unrated_index:
        tiers[ids[index]] = UNRATED_TIER
        z[ids[index]] = 0.0
    exploration = frozenset(ids[i] for i in unrated_index)
    return ScoreInputs(z, tiers, exploration)


SMALL_STORE_OVERRIDES: dict[str, object] = {
    "staffing.k_sc_calm": 1,
    "staffing.k_sc_rush": 2,
    "staffing.sco_zone1_calm": 1,
    "staffing.sco_zone2_calm": 0,
    "staffing.sco_zone1_rush": 1,
    "staffing.sco_zone2_rush": 1,
    "staffing.floaters_morning": 1,
    "staffing.floaters_afternoon": 1,
    "workforce.n_workers": 16,
    "solver.deterministic_time": 20.0,
}


def small_store_config(
    config: PlannerConfig | None = None, demand_scale: float = 0.22, **extra: object
) -> PlannerConfig:
    """Configuration of a small store (demand and staffing scaled down)."""
    base = config if config is not None else PlannerConfig()
    levels = {day: level * demand_scale for day, level in base.demand.daily_customers.items()}
    overrides = {**SMALL_STORE_OVERRIDES, "demand.daily_customers": levels, **extra}
    return base.with_overrides(overrides)


MEDIUM_STORE_OVERRIDES: dict[str, object] = {
    "staffing.k_sc_calm": 1,
    "staffing.k_sc_rush": 2,
    "staffing.sco_zone1_calm": 1,
    "staffing.sco_zone2_calm": 0,
    "staffing.sco_zone1_rush": 1,
    "staffing.sco_zone2_rush": 1,
    "staffing.floaters_morning": 1,
    "staffing.floaters_afternoon": 1,
    "workforce.n_workers": 24,
    "solver.deterministic_time": 60.0,
}


def medium_store_config(
    config: PlannerConfig | None = None, demand_scale: float = 0.33, **extra: object
) -> PlannerConfig:
    """Configuration of a medium store."""
    base = config if config is not None else PlannerConfig()
    levels = {day: level * demand_scale for day, level in base.demand.daily_customers.items()}
    overrides = {**MEDIUM_STORE_OVERRIDES, "demand.daily_customers": levels, **extra}
    return base.with_overrides(overrides)


TINY_STORE_OVERRIDES: dict[str, object] = {
    "staffing.k_sc_calm": 0,
    "staffing.k_sc_rush": 1,
    "staffing.sco_zone1_calm": 0,
    "staffing.sco_zone2_calm": 0,
    "staffing.sco_zone1_rush": 1,
    "staffing.sco_zone2_rush": 0,
    "staffing.floaters_morning": 0,
    "staffing.floaters_afternoon": 0,
    "staffing.break_share": 0.0,
    "workforce.n_workers": 10,
    "solver.deterministic_time": 4.0,
    "solver.coverage_first": False,
}


def tiny_store_config(
    config: PlannerConfig | None = None, demand_scale: float = 0.15, **extra: object
) -> PlannerConfig:
    """Configuration of a tiny store, for fast tests."""
    base = config if config is not None else PlannerConfig()
    levels = {day: level * demand_scale for day, level in base.demand.daily_customers.items()}
    overrides = {**TINY_STORE_OVERRIDES, "demand.daily_customers": levels, **extra}
    return base.with_overrides(overrides)


EVAL_STORE_OVERRIDES: dict[str, object] = {
    "workforce.n_workers": 24,
    "solver.deterministic_time": 30.0,
}

STORE_SIZES = ("tiny", "small", "medium", "eval", "full")


def store_config(name: str, **extra: object) -> PlannerConfig:
    """Configuration for a named store size: tiny, small, medium, eval or full."""
    if name == "tiny":
        return tiny_store_config(**extra)
    if name == "small":
        return small_store_config(**extra)
    if name == "medium":
        return medium_store_config(**extra)
    if name == "eval":
        return medium_store_config(demand_scale=0.6, **{**EVAL_STORE_OVERRIDES, **extra})
    if name == "full":
        return PlannerConfig().with_overrides(extra) if extra else PlannerConfig()
    raise ValueError(f"unknown store size {name!r}, choose one of {STORE_SIZES}")
