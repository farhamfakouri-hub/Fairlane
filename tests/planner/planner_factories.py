from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from datetime import date, datetime, timedelta

from ree.domain.models import CashierScoreRecord, Interval, ScoreComponents
from ree.planner.config import PlannerConfig, clock_to_minutes
from ree.planner.domain.demand_types import PlannerCalendar, RequirementTable, SlotRequirement
from ree.planner.domain.plan import (
    Assignment,
    DailyPlan,
    PostSegment,
    RosterPlan,
    WorkerRecord,
)
from ree.planner.domain.shifts import ShiftPattern, make_block, pattern_id_for
from ree.planner.roster.model import ScoreInputs, neutral_scores

CONFIG = PlannerConfig()
RULES = CONFIG.rules
CALENDAR = PlannerCalendar(CONFIG)
START = date(2026, 11, 1)
MONDAY = START + timedelta(days=1)
MORNING = ((clock_to_minutes("08:30"), clock_to_minutes("15:30")),)
EVENING = ((clock_to_minutes("14:30"), clock_to_minutes("21:30")),)


def minutes(clock: str) -> int:
    return clock_to_minutes(clock)


def pattern(*windows: tuple[int, int]) -> ShiftPattern:
    blocks = tuple(
        make_block(start, end, RULES.clock_in_margin_minutes, RULES.clock_out_margin_minutes)
        for start, end in windows
    )
    kind = "split" if len(blocks) > 1 else "continuous"
    return ShiftPattern(pattern_id_for(kind, blocks), kind, blocks)


def morning_pattern() -> ShiftPattern:
    return pattern(*MORNING)


def evening_pattern() -> ShiftPattern:
    return pattern(*EVENING)


def worker(worker_id: str, contract_hours: float = 29.0, **flags: object) -> WorkerRecord:
    return WorkerRecord(worker_id=worker_id, contract_hours=contract_hours, **flags)


def requirement_for(
    days: Iterable[date],
    headcount: int = 2,
    registers: int = 2,
    k_sc: int = 0,
    floaters: int = 0,
    zone1: int = 0,
    zone2: int = 0,
) -> RequirementTable:
    table = RequirementTable(slot_minutes=CONFIG.demand.slot_minutes)
    for day in days:
        rows = []
        for slot_start in CALENDAR.slot_starts(day):
            rows.append(
                (
                    slot_start,
                    SlotRequirement(
                        registers=registers,
                        k_sc=k_sc,
                        sco_zone1=zone1,
                        sco_zone2=zone2,
                        floaters=floaters,
                        headcount=headcount,
                        rush=False,
                        offered_load=1.0,
                        planned_customers=10.0,
                    ),
                )
            )
        table.days[day] = tuple(rows)
    return table


def with_rush(
    requirement: RequirementTable, day: date, start: int, end: int, load: float = 3.0
) -> RequirementTable:
    table = RequirementTable(slot_minutes=requirement.slot_minutes)
    for each_day, rows in requirement.days.items():
        table.days[each_day] = tuple(
            (
                slot_start,
                dataclasses.replace(row, rush=True, offered_load=load)
                if each_day == day and start <= slot_start < end
                else row,
            )
            for slot_start, row in rows
        )
    return table


def score_record(
    worker_id: str,
    score: float,
    confidence: float = 0.8,
    tier: str = "A",
    provisional: bool = False,
) -> CashierScoreRecord:
    return CashierScoreRecord(
        worker_id=worker_id,
        score_peak=score,
        score_off=score,
        interval_peak=Interval(lower=score - 3, upper=score + 3),
        interval_off=Interval(lower=score - 3, upper=score + 3),
        tier_peak=tier,
        tier_off=tier,
        n_transactions=900,
        weeks_of_data=10,
        provisional=provisional,
        review_flag=None,
        drivers=[],
        model_version="planner-test",
        computed_at=datetime(2026, 10, 1, 20, 0, 0),
        components=ScoreComponents(
            speed_score_peak=score,
            speed_score_off=score,
            accuracy_score=50.0,
            n_valid_peak=400,
            n_valid_off=500,
            error_rate=None,
            void_rate_per_100=None,
            confidence_peak=confidence,
            confidence_off=confidence,
            speed_method="v1",
        ),
    )


def valid_roster(days: int = 4, weeks: int = 1) -> tuple[RosterPlan, RequirementTable]:
    workers = {name: worker(name) for name in "ABFCDG"}
    assignments: list[Assignment] = []
    worked_days = [START + timedelta(days=1 + offset) for offset in range(days)]
    for name in "ABF":
        for day in worked_days:
            assignments.append(Assignment(name, day, morning_pattern()))
    for name in "CDG":
        for day in worked_days:
            assignments.append(Assignment(name, day, evening_pattern()))
    roster = RosterPlan(START, weeks, workers, assignments)
    return roster, requirement_for(worked_days)


def _segments(day: date, worker_id: str, items: list[tuple[int, int, str]]) -> list[PostSegment]:
    return [PostSegment(worker_id, day, start, end, post) for start, end, post in items]


def valid_daily() -> tuple[DailyPlan, RequirementTable]:
    from ree.planner.domain.plan import BreakEntry, Handover

    day = MONDAY
    requirement = requirement_for([day], headcount=3, registers=2, floaters=1)
    shifts = {name: morning_pattern() for name in "ABF"}
    shifts.update({name: evening_pattern() for name in "CDG"})
    workers = {name: worker(name) for name in "ABFCDG"}
    segments: list[PostSegment] = []
    segments += _segments(
        day,
        "A",
        [(510, 690, "R01"), (690, 715, "BREAK"), (715, 720, "FLOAT"), (720, 930, "R04")],
    )
    segments += _segments(
        day,
        "B",
        [(510, 720, "R04"), (720, 745, "BREAK"), (745, 750, "FLOAT"), (750, 930, "R01")],
    )
    segments += _segments(
        day,
        "F",
        [(510, 690, "FLOAT"), (690, 750, "R01"), (750, 775, "BREAK"), (775, 930, "FLOAT")],
    )
    segments += _segments(
        day,
        "C",
        [
            (870, 930, "FLOAT"),
            (930, 1050, "R01"),
            (1050, 1075, "BREAK"),
            (1075, 1080, "FLOAT"),
            (1080, 1290, "R04"),
        ],
    )
    segments += _segments(
        day,
        "D",
        [
            (870, 930, "FLOAT"),
            (930, 1080, "R04"),
            (1080, 1105, "BREAK"),
            (1105, 1110, "FLOAT"),
            (1110, 1290, "R01"),
        ],
    )
    segments += _segments(
        day,
        "G",
        [(870, 1050, "FLOAT"), (1050, 1110, "R01"), (1110, 1135, "BREAK"), (1135, 1290, "FLOAT")],
    )
    breaks = [
        BreakEntry("A", day, 0, 690, 25),
        BreakEntry("B", day, 0, 720, 25),
        BreakEntry("F", day, 0, 750, 25),
        BreakEntry("C", day, 0, 1050, 25),
        BreakEntry("D", day, 0, 1080, 25),
        BreakEntry("G", day, 0, 1110, 25),
    ]
    handovers = [
        Handover(day, 690, "F", "FLOAT", "R01", "relief", "A"),
        Handover(day, 720, "A", "FLOAT", "R04", "relief", "B"),
        Handover(day, 750, "B", "FLOAT", "R01", "relief", "F"),
    ]
    plan = DailyPlan(day, shifts, workers, segments, breaks, handovers)
    return plan, requirement


TUESDAY = START + timedelta(days=2)
OPENING_ORDER = CONFIG.staffing.opening_order
RUSH = (630, 840)
DEFAULT_Z = [1.8, 1.2, 0.6, 0.0, -0.6, -1.2, -1.8, -2.0, -2.0, -2.0]


def slot(registers: int = 4, zone1: int = 1, zone2: int = 0) -> SlotRequirement:
    return SlotRequirement(
        registers=registers,
        k_sc=zone1 + zone2,
        sco_zone1=zone1,
        sco_zone2=zone2,
        floaters=1,
        headcount=registers + zone1 + zone2 + 1,
        rush=False,
        offered_load=1.0,
        planned_customers=10.0,
    )


def staggered_shifts(pairs: int) -> dict[str, ShiftPattern]:
    return {
        f"W{index + 1:02d}": pattern((510 + 30 * (index // 2), 930 + 30 * (index // 2)))
        for index in range(2 * pairs)
    }


def requirement_from_duty(
    shifts: dict[str, ShiftPattern], rush: tuple[int, int] = RUSH, with_sco: bool = False
) -> RequirementTable:
    first = min(p.register_start for p in shifts.values())
    last = max(p.register_end for p in shifts.values())
    rows = []
    for start in range(first, last, 30):
        duty = sum(1 for p in shifts.values() if p.register_start <= start < p.register_end)
        supervisors = 1 if with_sco and duty >= 5 else 0
        registers = max(2, duty - 2 - supervisors)
        in_rush = rush[0] <= start < rush[1]
        rows.append(
            (
                start,
                SlotRequirement(
                    registers=registers,
                    k_sc=supervisors,
                    sco_zone1=supervisors,
                    sco_zone2=0,
                    floaters=2,
                    headcount=registers + supervisors + 2,
                    rush=in_rush,
                    offered_load=3.0 if in_rush else 1.0,
                    planned_customers=10.0,
                ),
            )
        )
    table = RequirementTable(slot_minutes=30)
    table.days[TUESDAY] = tuple(rows)
    return table


def staggered_day(
    pairs: int = 4,
    z: list[float] | None = None,
    flags: dict[str, dict] | None = None,
    rush: tuple[int, int] = RUSH,
    with_sco: bool = False,
):
    shifts = staggered_shifts(pairs)
    names = list(shifts)
    given_flags = flags or {}
    workers = {name: worker(name, 29.0, **given_flags.get(name, {})) for name in names}
    requirement = requirement_from_duty(shifts, rush, with_sco)
    values = (z if z is not None else DEFAULT_Z)[: len(names)]
    scores = ScoreInputs(
        z=dict(zip(names, values, strict=True)),
        tiers={name: "all" for name in names},
        exploration=frozenset(),
    )
    return names, workers, shifts, requirement, scores


def flat_day(count: int, registers: int, zone1: int):
    names = [f"W{index:02d}" for index in range(1, count + 1)]
    workers = {name: worker(name, 29.0) for name in names}
    shifts = {name: morning_pattern() for name in names}
    requirement = RequirementTable(slot_minutes=30)
    requirement.days[TUESDAY] = tuple(
        (start, slot(registers, zone1)) for start in range(510, 930, 30)
    )
    return names, workers, shifts, requirement, neutral_scores(names)
