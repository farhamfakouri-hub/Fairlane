"""The daily plan: breaks, posts, handovers and checks for one day."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ree.planner.checker.checks import check_daily
from ree.planner.checker.report import Report
from ree.planner.config import PlannerConfig
from ree.planner.daily.breaks import STEP, BreakSchedule, schedule_breaks, slot_lookup
from ree.planner.daily.posts import Timeline, assign_posts
from ree.planner.daily.relief import (
    RELIEF,
    build_segments,
    derive_handovers,
    describe_relief,
)
from ree.planner.domain.demand_types import RequirementTable
from ree.planner.domain.plan import (
    DailyPlan,
    RosterPlan,
    WorkerRecord,
    is_register_post,
    register_number,
)
from ree.planner.domain.shifts import ShiftPattern, minutes_to_clock
from ree.planner.roster.model import ScoreInputs, neutral_scores

ALERT_CODES = frozenset(
    {
        "COVERAGE_GAP",
        "POST_EMPTY",
        "REGISTERS_1_4_UNSTAFFED",
        "BREAKS_LIMIT_CASE",
        "TOO_MANY_BREAKS",
        "BREAK_MISSING",
        "BREAK_OUTSIDE_WINDOW",
    }
)


@dataclass(frozen=True)
class DailyMetrics:
    """Counts and gaps that summarise a daily plan."""

    breaks: int
    peak_simultaneous_breaks: int
    capacity_gap_minutes: int
    handovers: int
    relief_handovers: int
    priority_score_gap: float


@dataclass
class DailyResult:
    """A daily plan with its check report, metrics, relief lines and notes."""

    plan: DailyPlan
    report: Report
    metrics: DailyMetrics
    relief_lines: list[str]
    notes: list[str] = field(default_factory=list)


def shifts_for_day(roster: RosterPlan, day: date) -> dict[str, ShiftPattern]:
    """Pattern of each worker who works on a day."""
    return {a.worker_id: a.pattern for a in roster.assignments if a.day == day}


def priority_score_gap(
    day: date,
    timeline: Timeline,
    requirement: RequirementTable,
    z: Mapping[str, float],
) -> float:
    """Mean score on the priority registers in rush slots minus the mean elsewhere."""
    open_minute, close_minute = requirement.opening_hours(day) or (0, 0)
    slots = slot_lookup(requirement, day, range(open_minute, close_minute, STEP))
    priority = {f"R{number:02d}" for number in requirement.opening_order[:2]}
    on_priority: list[float] = []
    elsewhere: list[float] = []
    for worker_id, cells in timeline.items():
        for minute, post in cells.items():
            slot = slots.get(minute)
            if slot is None or not slot.rush or not is_register_post(post):
                continue
            (on_priority if post in priority else elsewhere).append(z.get(worker_id, 0.0))
    if not on_priority or not elsewhere:
        return 0.0
    return float(np.mean(on_priority) - np.mean(elsewhere))


def plan_day_from_shifts(
    day: date,
    shifts: Mapping[str, ShiftPattern],
    workers: Mapping[str, WorkerRecord],
    requirement: RequirementTable,
    scores: ScoreInputs,
    config: PlannerConfig,
    schedule: BreakSchedule | None = None,
) -> DailyResult:
    """Plan a day from given shifts: breaks, posts, segments and handovers, then check it."""
    used = schedule if schedule is not None else schedule_breaks(day, shifts, requirement, config)
    timeline = assign_posts(day, shifts, used.entries, requirement, scores.z, workers, config)
    segments = build_segments(day, timeline, shifts)
    handovers = derive_handovers(day, timeline)
    plan = DailyPlan(
        day=day,
        shifts=dict(shifts),
        workers={worker_id: workers[worker_id] for worker_id in shifts},
        segments=segments,
        breaks=list(used.entries),
        handovers=handovers,
    )
    report = check_daily(plan, config.rules, requirement)
    plan.alerts = [
        f"{minutes_to_clock(item.minute) if item.minute is not None else '--:--'} "
        f"{item.code}: {item.message}"
        for item in report.errors + report.warnings
        if item.code in ALERT_CODES
    ]
    metrics = DailyMetrics(
        breaks=len(used.entries),
        peak_simultaneous_breaks=used.peak_simultaneous,
        capacity_gap_minutes=used.capacity_gap_minutes,
        handovers=len(handovers),
        relief_handovers=sum(1 for item in handovers if item.reason == RELIEF),
        priority_score_gap=priority_score_gap(day, timeline, requirement, scores.z),
    )
    return DailyResult(
        plan=plan,
        report=report,
        metrics=metrics,
        relief_lines=describe_relief(handovers, list(used.entries)),
        notes=list(used.notes),
    )


def plan_day(
    roster: RosterPlan,
    day: date,
    requirement: RequirementTable,
    config: PlannerConfig,
    scores: ScoreInputs | None = None,
) -> DailyResult:
    """Plan one day of a roster."""
    used_scores = scores if scores is not None else neutral_scores(sorted(roster.workers))
    shifts = shifts_for_day(roster, day)
    return plan_day_from_shifts(day, shifts, roster.workers, requirement, used_scores, config)


def plan_days(
    roster: RosterPlan,
    requirement: RequirementTable,
    config: PlannerConfig,
    scores: ScoreInputs | None = None,
) -> dict[date, DailyResult]:
    """Plan every day of a roster that has shifts."""
    return {
        day: plan_day(roster, day, requirement, config, scores)
        for day in roster.dates()
        if any(a.day == day for a in roster.assignments)
    }


def register_rank(post: str, opening_order: tuple[int, ...]) -> int:
    """Position of a register in the opening order."""
    return opening_order.index(register_number(post))


def board_lines(plan: DailyPlan) -> list[str]:
    """Text board of a day: each worker's posts with start times and the end of the shift."""
    lines: list[str] = []
    for worker_id in sorted(plan.shifts):
        pieces = [
            f"{minutes_to_clock(segment.start)} {segment.post}"
            for segment in sorted(
                (s for s in plan.segments if s.worker_id == worker_id), key=lambda s: s.start
            )
        ]
        ends = minutes_to_clock(plan.shifts[worker_id].register_end)
        lines.append(f"{worker_id}: " + ", ".join(pieces) + f", until {ends}")
    return lines
