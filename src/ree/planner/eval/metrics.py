"""Metrics of a simulated week and of a plan."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ree.planner.config import PlannerConfig
from ree.planner.domain.plan import is_register_post
from ree.planner.eval.plans import PlanBundle
from ree.planner.eval.world import DayRun
from ree.planner.roster.fairness import burden_rates, burden_spread, max_tier_gap
from ree.planner.sim.part1_workforce import WorkerTruth

WAIT_METRICS = (
    "mean_wait_s",
    "p90_wait_s",
    "p95_wait_s",
    "share_over_target",
    "abandon_rate",
    "rush_mean_wait_s",
    "rush_p95_wait_s",
    "rush_share_over_target",
    "idle_share",
    "mean_queue_per_register",
    "share_time_over_limit",
)


def _percentile(values: np.ndarray, level: float) -> float:
    return float(np.percentile(values, level)) if len(values) else 0.0


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def week_metrics(runs: Sequence[DayRun], target_seconds: float) -> dict[str, float]:
    """Waiting, abandonment, queue and idle measures of a simulated week."""
    if not runs:
        return {name: 0.0 for name in WAIT_METRICS} | {"customers": 0.0}
    wait = np.concatenate([r.outcome.wait for r in runs])
    abandoned = np.concatenate([r.outcome.abandoned for r in runs])
    rush = np.concatenate([r.rush for r in runs])
    bad = (wait > target_seconds) | abandoned
    busy = sum(r.outcome.busy_seconds for r in runs)
    open_register_seconds = sum(r.outcome.open_seconds for r in runs)
    queue_area = sum(r.outcome.queue_area for r in runs)
    over = sum(r.outcome.over_limit_seconds for r in runs)
    open_day_seconds = sum(r.open_seconds for r in runs)
    return {
        "customers": float(len(wait)),
        "mean_wait_s": _mean(wait),
        "p90_wait_s": _percentile(wait, 90.0),
        "p95_wait_s": _percentile(wait, 95.0),
        "share_over_target": _mean(bad.astype(float)),
        "abandon_rate": _mean(abandoned.astype(float)),
        "rush_mean_wait_s": _mean(wait[rush]),
        "rush_p95_wait_s": _percentile(wait[rush], 95.0),
        "rush_share_over_target": _mean(bad[rush].astype(float)),
        "idle_share": 1.0 - busy / open_register_seconds if open_register_seconds else 0.0,
        "mean_queue_per_register": queue_area / open_register_seconds
        if open_register_seconds
        else 0.0,
        "share_time_over_limit": over / open_day_seconds if open_day_seconds else 0.0,
    }


def roster_metrics(
    bundle: PlanBundle, config: PlannerConfig, tiers: dict[str, str]
) -> dict[str, float]:
    """Measures of a roster and its daily plans: shortfall, hours, burden and handovers."""
    plan = bundle.roster.plan
    report = bundle.roster.report
    rates = burden_rates(plan, config)
    shifts = len(plan.assignments)
    handovers = sum(len(result.plan.handovers) for result in bundle.daily.values())
    return {
        "roster_shortfall_minutes": float(report.shortfall_minutes())
        if report is not None
        else 0.0,
        "roster_errors": float(len(report.errors)) if report is not None else 0.0,
        "paid_hours": sum(a.pattern.paid_minutes for a in plan.assignments) / 60.0,
        "planned_staff_hours": sum(
            bundle.requirement.staff_minutes(day) for day in bundle.requirement.dates()
        )
        / 60.0,
        "burden_spread": burden_spread(rates),
        "max_tier_gap": max_tier_gap(rates, {w: t for w, t in tiers.items() if w in rates}),
        "handovers_per_shift": handovers / shifts if shifts else 0.0,
        "break_gap_minutes": float(
            sum(r.metrics.capacity_gap_minutes for r in bundle.daily.values())
        ),
        "relaxed_rules": float(len(bundle.roster.relaxed)),
    }


def rush_overlap_minutes(bundle: PlanBundle, day, start: int, end: int) -> int:
    """Minutes of an interval that fall in rush slots."""
    requirement = bundle.requirement
    total = 0
    for slot_start, row in requirement.slots(day):
        if row.rush:
            total += max(
                0, min(end, slot_start + requirement.slot_minutes) - max(start, slot_start)
            )
    return total


def register_staffing(bundle: PlanBundle) -> dict[str, float]:
    """Share of the rush and the calm slots in which the required registers were staffed."""
    rush_minutes = 0
    rush_register_minutes = 0
    calm_register_minutes = 0
    calm_minutes = 0
    requirement = bundle.requirement
    for day in requirement.dates():
        for _, row in requirement.slots(day):
            if row.rush:
                rush_minutes += requirement.slot_minutes
            else:
                calm_minutes += requirement.slot_minutes
        result = bundle.daily.get(day)
        if result is None:
            continue
        for segment in result.plan.segments:
            if not is_register_post(segment.post):
                continue
            in_rush = rush_overlap_minutes(bundle, day, segment.start, segment.end)
            rush_register_minutes += in_rush
            calm_register_minutes += (segment.end - segment.start) - in_rush
    return {
        "rush_registers_staffed": rush_register_minutes / rush_minutes if rush_minutes else 0.0,
        "calm_registers_staffed": calm_register_minutes / calm_minutes if calm_minutes else 0.0,
    }


def priority_true_gap(bundle: PlanBundle, truth: dict[str, WorkerTruth]) -> float:
    """True speed on registers 1 and 4 in rush slots minus that on other registers."""
    priority = {f"R{number:02d}" for number in bundle.requirement.opening_order[:2]}
    weighted = {True: 0.0, False: 0.0}
    minutes = {True: 0.0, False: 0.0}
    for day, result in bundle.daily.items():
        for segment in result.plan.segments:
            if not is_register_post(segment.post):
                continue
            rush = rush_overlap_minutes(bundle, day, segment.start, segment.end)
            if not rush:
                continue
            is_priority = segment.post in priority
            weighted[is_priority] += rush * truth[segment.worker_id].log_speed
            minutes[is_priority] += rush
    if not minutes[True] or not minutes[False]:
        return 0.0
    return weighted[True] / minutes[True] - weighted[False] / minutes[False]
