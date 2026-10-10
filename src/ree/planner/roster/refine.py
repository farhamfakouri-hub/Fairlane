"""Refinement after solving: swaps of shifts that improve score fit, and repairs of tier burden."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from ree.planner.checker.checks import check_worker_rules
from ree.planner.config import PlannerConfig
from ree.planner.domain.plan import Assignment, RosterPlan, WorkerRecord
from ree.planner.domain.shifts import ShiftPattern
from ree.planner.roster.fairness import burden_rates, burden_spread, day_burden, max_tier_gap
from ree.planner.roster.model import RosterProblem, rush_weights

SPREAD_TOLERANCE = 1.05


@dataclass(frozen=True)
class Swap:
    """Two workers exchange their patterns on one day."""

    day: date
    first: str
    second: str
    gain: float


def _pressure(problem: RosterProblem, weights: dict, day: date, pattern: ShiftPattern) -> float:
    slot_minutes = problem.requirement.slot_minutes
    return (
        sum(
            weights.get((day, start), 0.0) * pattern.coverage_minutes(start, start + slot_minutes)
            for start, _ in problem.requirement.slots(day)
        )
        / slot_minutes
    )


def _roles(worker: WorkerRecord) -> tuple[bool, bool]:
    return worker.self_checkout_only, worker.cleared_self_checkout


def _interchangeable(
    config: PlannerConfig, day: date, first: ShiftPattern, second: ShiftPattern
) -> bool:
    return first.paid_minutes == second.paid_minutes and day_burden(
        first, day, config
    ) == day_burden(second, day, config)


def _legal(problem: RosterProblem, plan: RosterPlan, worker_id: str) -> bool:
    worker = plan.workers[worker_id]
    single = RosterPlan(
        plan.start_date,
        plan.weeks,
        {worker_id: worker},
        [a for a in plan.assignments if a.worker_id == worker_id],
    )
    return not any(v.severity == "error" for v in check_worker_rules(single, problem.config.rules))


def _blocked(worker: WorkerRecord, day: date, pattern: ShiftPattern) -> bool:
    return any(worker.is_blocked(day, b.paid_start, b.paid_end) for b in pattern.blocks)


def refine_scores(
    problem: RosterProblem, plan: RosterPlan, max_passes: int = 6
) -> tuple[RosterPlan, list[Swap]]:
    """Swap shifts on the same day so that higher scores sit on busier slots.

    In neutral mode only swaps of equal paid minutes and equal burden are allowed. Every
    swap must keep both workers legal and available.
    """
    z = problem.scores.z
    if not any(abs(value) > 0.0 for value in z.values()):
        return plan, []
    config = problem.config
    neutral = config.objective.neutral_swaps
    tiers = problem.scores.tiers
    history = problem.burden_history
    weights = rush_weights(problem.requirement)
    swaps: list[Swap] = []
    current = RosterPlan(plan.start_date, plan.weeks, plan.workers, list(plan.assignments))
    for _ in range(max_passes):
        limits: tuple[float, float] | None = None
        if not neutral:
            rates = burden_rates(current, config, history)
            limits = (max_tier_gap(rates, tiers), burden_spread(rates))
        by_day: dict[date, list[Assignment]] = defaultdict(list)
        for assignment in current.assignments:
            by_day[assignment.day].append(assignment)
        candidates: list[tuple[float, date, Assignment, Assignment]] = []
        for day, assignments in sorted(by_day.items()):
            ordered = sorted(assignments, key=lambda a: a.worker_id)
            for i, first in enumerate(ordered):
                for second in ordered[i + 1 :]:
                    if first.pattern.pattern_id == second.pattern.pattern_id:
                        continue
                    if _roles(current.workers[first.worker_id]) != _roles(
                        current.workers[second.worker_id]
                    ):
                        continue
                    if neutral and not _interchangeable(config, day, first.pattern, second.pattern):
                        continue
                    before = z.get(first.worker_id, 0.0) * _pressure(
                        problem, weights, day, first.pattern
                    ) + z.get(second.worker_id, 0.0) * _pressure(
                        problem, weights, day, second.pattern
                    )
                    after = z.get(first.worker_id, 0.0) * _pressure(
                        problem, weights, day, second.pattern
                    ) + z.get(second.worker_id, 0.0) * _pressure(
                        problem, weights, day, first.pattern
                    )
                    if after > before + 1e-9:
                        candidates.append((after - before, day, first, second))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2].worker_id, item[3].worker_id))
        applied = False
        touched: set[tuple[str, date]] = set()
        for gain, day, first, second in candidates:
            keys = {(first.worker_id, day), (second.worker_id, day)}
            if keys & touched:
                continue
            worker_a, worker_b = current.workers[first.worker_id], current.workers[second.worker_id]
            if _blocked(worker_a, day, second.pattern) or _blocked(worker_b, day, first.pattern):
                continue
            trial = RosterPlan(
                current.start_date,
                current.weeks,
                current.workers,
                [
                    Assignment(first.worker_id, day, second.pattern)
                    if a is first
                    else Assignment(second.worker_id, day, first.pattern)
                    if a is second
                    else a
                    for a in current.assignments
                ],
            )
            if not (
                _legal(problem, trial, first.worker_id) and _legal(problem, trial, second.worker_id)
            ):
                continue
            if limits is not None:
                trial_rates = burden_rates(trial, config, history)
                gap = max_tier_gap(trial_rates, tiers)
                if gap > max(limits[0], config.objective.tier_epsilon):
                    continue
                if burden_spread(trial_rates) > limits[1] * SPREAD_TOLERANCE + 1e-9:
                    continue
                limits = (gap, burden_spread(trial_rates))
            current = trial
            touched |= keys
            swaps.append(Swap(day, first.worker_id, second.worker_id, gain))
            applied = True
        if not applied:
            break
    return current, swaps


def _swapped(plan: RosterPlan, day: date, first: Assignment, second: Assignment) -> RosterPlan:
    return RosterPlan(
        plan.start_date,
        plan.weeks,
        plan.workers,
        [
            Assignment(first.worker_id, day, second.pattern)
            if a is first
            else Assignment(second.worker_id, day, first.pattern)
            if a is second
            else a
            for a in plan.assignments
        ],
    )


def repair_tiers(
    problem: RosterProblem, plan: RosterPlan, max_swaps: int = 60
) -> tuple[RosterPlan, list[Swap]]:
    """Swap shifts to bring the burden of every tier back inside the guard limit."""
    config = problem.config
    tiers = problem.scores.tiers
    limit = config.objective.tier_epsilon
    history = problem.burden_history
    weeks = plan.weeks + (config.rules.history_weeks if history else 0)
    current = RosterPlan(plan.start_date, plan.weeks, plan.workers, list(plan.assignments))
    rates = burden_rates(current, config, history)
    gap = max_tier_gap(rates, tiers)
    swaps: list[Swap] = []
    while gap > limit and len(swaps) < max_swaps:
        by_day: dict[date, list[Assignment]] = defaultdict(list)
        for assignment in current.assignments:
            by_day[assignment.day].append(assignment)
        candidates: list[tuple[float, date, Assignment, Assignment]] = []
        for day, assignments in sorted(by_day.items()):
            ordered = sorted(assignments, key=lambda a: a.worker_id)
            for i, first in enumerate(ordered):
                for second in ordered[i + 1 :]:
                    a, b = first.worker_id, second.worker_id
                    if tiers.get(a) == tiers.get(b) or a not in rates or b not in rates:
                        continue
                    if first.pattern.pattern_id == second.pattern.pattern_id:
                        continue
                    if first.pattern.paid_minutes != second.pattern.paid_minutes:
                        continue
                    if _roles(current.workers[a]) != _roles(current.workers[b]):
                        continue
                    burden_a = day_burden(first.pattern, day, config)
                    burden_b = day_burden(second.pattern, day, config)
                    if burden_a == burden_b:
                        continue
                    trial_rates = dict(rates)
                    trial_rates[a] += (burden_b - burden_a) / (
                        current.workers[a].contract_hours * weeks
                    )
                    trial_rates[b] += (burden_a - burden_b) / (
                        current.workers[b].contract_hours * weeks
                    )
                    new_gap = max_tier_gap(trial_rates, tiers)
                    if new_gap < gap - 1e-9:
                        candidates.append((new_gap, day, first, second))
        candidates.sort(key=lambda item: (item[0], item[1], item[2].worker_id, item[3].worker_id))
        applied = False
        for new_gap, day, first, second in candidates:
            worker_a, worker_b = current.workers[first.worker_id], current.workers[second.worker_id]
            if _blocked(worker_a, day, second.pattern) or _blocked(worker_b, day, first.pattern):
                continue
            trial = _swapped(current, day, first, second)
            if not (
                _legal(problem, trial, first.worker_id) and _legal(problem, trial, second.worker_id)
            ):
                continue
            swaps.append(Swap(day, first.worker_id, second.worker_id, gap - new_gap))
            current = trial
            rates = burden_rates(current, config, history)
            gap = max_tier_gap(rates, tiers)
            applied = True
            break
        if not applied:
            break
    return current, swaps
