"""Simulated worlds: the customers and absences a plan is tested against."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ree.config import make_rng
from ree.planner.config import PlannerConfig
from ree.planner.daily.daily_plan import DailyResult
from ree.planner.demand.forecast import default_model, forecast_day
from ree.planner.domain.demand_types import PlannerCalendar, RequirementTable
from ree.planner.domain.plan import is_register_post, register_number
from ree.planner.eval.plans import PlanBundle
from ree.planner.sim.demand_gen import draw_day_customers
from ree.planner.sim.part1_workforce import WorkerTruth, service_multiplier
from ree.planner.sim.queue_sim import (
    DayCustomers,
    DayOutcome,
    QueueSettings,
    StaffInterval,
    simulate_day,
)

SECONDS = 60.0
SATURDAY = 5


@dataclass(frozen=True)
class Scenario:
    """A test scenario: its group, team size, surge, demand error and absence rate."""

    name: str
    group: str
    start_date: str | None
    team_share: float
    saturday_surge: float
    level_sd: float
    absence_rate: float
    description: str


SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            "normal_week", "base", None, 1.0, 1.0, 0.0, 0.0, "the forecast is right on average"
        ),
        Scenario(
            "saturday_surge",
            "base",
            None,
            1.0,
            1.3,
            0.0,
            0.0,
            "Saturday brings 30 percent more customers",
        ),
        Scenario(
            "absences", "base", None, 1.0, 1.0, 0.0, 0.08, "8 percent of rostered shifts are missed"
        ),
        Scenario(
            "season_peak",
            "season",
            "2026-12-20",
            1.0,
            1.0,
            0.0,
            0.0,
            "the week from 20 December with bigger baskets",
        ),
        Scenario(
            "forecast_error",
            "base",
            None,
            1.0,
            1.0,
            0.15,
            0.0,
            "the real demand level differs from the forecast by a random 15 percent",
        ),
        Scenario(
            "small_team",
            "small",
            None,
            0.75,
            1.0,
            0.0,
            0.0,
            "a quarter of the cashiers are missing",
        ),
    )
}


@dataclass
class World:
    """Customers and absence draws of one scenario, seed and repetition."""

    scenario: Scenario
    seed: int
    config: PlannerConfig
    days: list[date]
    customers: dict[date, DayCustomers]
    absence_draw: dict[tuple[str, date], float]


def scenario_config(config: PlannerConfig, scenario: Scenario) -> PlannerConfig:
    """Configuration of a scenario, with its start date when it has one."""
    if scenario.start_date is None:
        return config
    return config.with_overrides({"start_date": scenario.start_date})


def build_world(
    config: PlannerConfig,
    scenario: Scenario,
    seed: int,
    worker_ids: Sequence[str],
    weeks: int = 1,
    replication: int = 0,
) -> World:
    """Draw the customers and absences of a scenario."""
    calendar = PlannerCalendar(config)
    model = default_model(calendar, config)
    start = date.fromisoformat(config.start_date)
    days = [start + timedelta(days=offset) for offset in range(7 * weeks)]
    level_sd = scenario.level_sd
    level_rng = make_rng(seed, f"planner-level:{replication}")
    level = float(np.exp(level_rng.normal(0.0, level_sd) - 0.5 * level_sd**2)) if level_sd else 1.0
    day_sd = config.demand.day_noise_sd
    customers: dict[date, DayCustomers] = {}
    for day in days:
        forecast = forecast_day(model, calendar, config, day)
        shock_rng = make_rng(seed, f"planner-day:{replication}:{day.isoformat()}")
        shock = float(np.exp(shock_rng.normal(0.0, day_sd) - 0.5 * day_sd**2))
        surge = scenario.saturday_surge if day.weekday() == SATURDAY else 1.0
        rng = make_rng(seed, f"planner-world:{replication}:{scenario.name}:{day.isoformat()}")
        customers[day] = draw_day_customers(
            rng,
            forecast.slot_starts,
            forecast.mean,
            config.demand.slot_minutes,
            config.demand.dispersion_k,
            config.service,
            forecast.season_service,
            level * shock * surge,
        )
    absence_rng = make_rng(seed, f"planner-absence:{replication}")
    draws = {
        (worker_id, day): float(absence_rng.random())
        for worker_id in sorted(worker_ids)
        for day in days
    }
    return World(scenario, seed, config, days, customers, draws)


def staff_intervals(result: DailyResult, absent: set[str]) -> list[StaffInterval]:
    """Register staffing of a daily plan, leaving out absent workers."""
    return [
        StaffInterval(
            register_number(segment.post),
            segment.start * SECONDS,
            segment.end * SECONDS,
            segment.worker_id,
        )
        for segment in result.plan.segments
        if is_register_post(segment.post) and segment.worker_id not in absent
    ]


def rush_lookup(requirement: RequirementTable, day: date):
    """Function that tells whether a second of the day is in a rush slot."""
    slots = requirement.slots(day)
    starts = np.array([start for start, _ in slots], dtype=float) * SECONDS
    flags = np.array([row.rush for _, row in slots], dtype=bool)
    slot_seconds = requirement.slot_minutes * SECONDS

    def lookup(second: float) -> bool:
        if not len(starts):
            return False
        index = int(np.searchsorted(starts, second, side="right")) - 1
        if index < 0 or second >= starts[index] + slot_seconds:
            return False
        return bool(flags[index])

    return lookup


@dataclass
class DayRun:
    """Outcome of one simulated day with its rush flags."""

    day: date
    outcome: DayOutcome
    rush: np.ndarray
    open_seconds: float
    absent: int


def simulate_plan(
    bundle: PlanBundle,
    world: World,
    truth: dict[str, WorkerTruth],
) -> list[DayRun]:
    """Run a plan through a world, day by day, using the hidden true speeds."""
    config = world.config
    runs: list[DayRun] = []
    positions = tuple((r, i) for i, r in enumerate(bundle.requirement.opening_order))
    for day in world.days:
        result = bundle.daily.get(day)
        customers = world.customers[day]
        hours = bundle.requirement.opening_hours(day)
        if result is None or hours is None:
            continue
        absent = {
            worker_id
            for worker_id in result.plan.shifts
            if world.absence_draw.get((worker_id, day), 1.0) < world.scenario.absence_rate
        }
        intervals = staff_intervals(result, absent)
        rush = rush_lookup(bundle.requirement, day)
        multiplier_cache: dict[tuple[str, bool], float] = {}

        def multiplier(worker_id: str, second: float, rush=rush, cache=multiplier_cache) -> float:
            flag = rush(second)
            key = (worker_id, flag)
            if key not in cache:
                cache[key] = service_multiplier(truth[worker_id], flag)
            return cache[key]

        settings = QueueSettings(
            discipline=config.service.queue_discipline,
            position_penalty=config.service.position_penalty,
            queue_limit=config.staffing.queue_limit_customers,
            close_second=hours[1] * SECONDS,
            register_positions=positions,
        )
        outcome = simulate_day(customers, intervals, multiplier, settings)
        flags = np.array([rush(float(t)) for t in outcome.arrival], dtype=bool)
        runs.append(DayRun(day, outcome, flags, (hours[1] - hours[0]) * SECONDS, len(absent)))
    return runs
