"""Plain-language reasons for the assignments of a roster, and the trade-offs it made."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from ree.planner.config import PlannerConfig
from ree.planner.domain.demand_types import RequirementTable
from ree.planner.domain.plan import RosterPlan
from ree.planner.domain.shifts import ShiftPattern
from ree.planner.roster.fairness import burden_rates, day_burden
from ree.planner.roster.model import ScoreInputs

RUSH_MINUTES_FOR_REASON = 60
UNPOPULAR_BURDEN = 2
TOP_SHARE = 1 / 3


@dataclass(frozen=True)
class Reason:
    """One reason for an assignment, with a code and its text."""

    code: str
    text: str


def rush_minutes(pattern: ShiftPattern, day: date, requirement: RequirementTable) -> int:
    """Minutes of a pattern that fall in rush slots."""
    return sum(
        pattern.coverage_minutes(start, start + requirement.slot_minutes)
        for start, row in requirement.slots(day)
        if row.rush
    )


def _available_ranks(
    plan: RosterPlan, scores: ScoreInputs, day: date, pattern: ShiftPattern
) -> dict[str, int]:
    able = [
        worker_id
        for worker_id, worker in plan.workers.items()
        if not any(worker.is_blocked(day, b.paid_start, b.paid_end) for b in pattern.blocks)
    ]
    ordered = sorted(able, key=lambda w: (-scores.z.get(w, 0.0), w))
    return {worker_id: position + 1 for position, worker_id in enumerate(ordered)}


def _shift_kind(pattern: ShiftPattern, day: date, config: PlannerConfig) -> str:
    if day.weekday() == 6:
        return "Sunday"
    if day.weekday() == 5:
        return "Saturday"
    if len(pattern.blocks) > 1:
        return "split"
    return "late"


def explain_roster(
    plan: RosterPlan,
    requirement: RequirementTable,
    scores: ScoreInputs,
    config: PlannerConfig,
    burden_history: Mapping[str, float] | None = None,
    last_roster: Mapping[tuple[str, date], str] | None = None,
) -> dict[tuple[str, date], list[Reason]]:
    """Reasons for each assignment.

    Rush cover, burden rotation, same shift as before and contract hours; at most four
    reasons are kept for each assignment.
    """
    history = burden_history or {}
    previous = last_roster or {}
    rates = burden_rates(plan, config, history)
    average = sum(rates.values()) / len(rates) if rates else 0.0
    weekly: dict[tuple[str, int], float] = defaultdict(float)
    for assignment in plan.assignments:
        weekly[(assignment.worker_id, plan.week_of(assignment.day))] += (
            assignment.pattern.paid_minutes / 60.0
        )
    explanations: dict[tuple[str, date], list[Reason]] = {}
    for assignment in plan.assignments:
        worker_id, day, pattern = assignment.worker_id, assignment.day, assignment.pattern
        worker = plan.workers[worker_id]
        reasons: list[Reason] = []
        in_rush = rush_minutes(pattern, day, requirement)
        if in_rush >= RUSH_MINUTES_FOR_REASON:
            if worker_id in scores.exploration:
                reasons.append(
                    Reason(
                        "RUSH_EXPLORATION",
                        "Peak slot so that new register data can be collected "
                        "(the score is still provisional)",
                    )
                )
            else:
                ranks = _available_ranks(plan, scores, day, pattern)
                rank = ranks.get(worker_id, len(ranks))
                if scores.z.get(worker_id, 0.0) > 0 and rank <= max(
                    1, round(len(ranks) * TOP_SHARE)
                ):
                    reasons.append(
                        Reason(
                            "RUSH_TOP_SCORE",
                            f"Among the strongest available peak scores for a rush slot "
                            f"(rank {rank} of {len(ranks)} available)",
                        )
                    )
        burden = day_burden(pattern, day, config)
        if burden >= UNPOPULAR_BURDEN and rates.get(worker_id, 0.0) <= average:
            reasons.append(
                Reason(
                    "BURDEN_ROTATION",
                    f"Fewer {_shift_kind(pattern, day, config)} shifts than average "
                    f"over the last {config.rules.history_weeks} weeks",
                )
            )
        if previous.get((worker_id, day)) == pattern.pattern_id:
            reasons.append(Reason("SAME_AS_BEFORE", "Same shift as in the previous roster"))
        if in_rush and not reasons:
            reasons.append(Reason("COVERAGE_NEED", "Needed to cover the rush slots of the day"))
        week = plan.week_of(day)
        reasons.append(
            Reason(
                "CONTRACT_HOURS",
                f"Contract hours to fill: {weekly[(worker_id, week)]:.2f} paid hours "
                f"this week against {worker.contract_hours:g} contracted",
            )
        )
        explanations[(worker_id, day)] = reasons[:4]
    return explanations


def trade_offs(
    plan: RosterPlan,
    requirement: RequirementTable,
    scores: ScoreInputs,
    config: PlannerConfig,
    relaxed: list[str],
    shortfall_minutes: float,
    max_tier_gap: float,
) -> list[str]:
    """Notes on what a roster gave up: rules that gave way, uncovered hours, burden, quota."""
    notes: list[str] = []
    for name in relaxed:
        notes.append(f"A rule group had to give way to find any roster: {name.replace('_', ' ')}")
    if shortfall_minutes > 0:
        hours = shortfall_minutes / 60.0
        notes.append(f"{hours:.1f} person-hours stay uncovered because no legal roster covers them")
    limit = config.objective.tier_epsilon
    if max_tier_gap > limit:
        notes.append(
            f"The burden of one score tier is {max_tier_gap:.0%} away from the average "
            f"(limit {limit:.0%})"
        )
    quota_missing = [
        worker_id
        for worker_id in sorted(scores.exploration)
        if worker_id in plan.workers
        and sum(
            rush_minutes(a.pattern, a.day, requirement)
            for a in plan.assignments
            if a.worker_id == worker_id
        )
        < config.objective.exploration_hours_per_week * 60 * plan.weeks
    ]
    if quota_missing:
        notes.append(
            f"{len(quota_missing)} new workers got fewer peak slots than the learning quota"
        )
    return notes
