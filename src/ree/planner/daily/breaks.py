"""Break scheduling: when each break starts, so that the floor stays covered."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from ortools.sat.python import cp_model

from ree.planner.config import LabourRules, PlannerConfig
from ree.planner.domain.demand_types import RequirementTable, SlotRequirement
from ree.planner.domain.plan import BreakEntry
from ree.planner.domain.rules import BREAK_GRID_MINUTES, block_break_length, break_window, grid_down
from ree.planner.domain.shifts import ShiftPattern

STEP = BREAK_GRID_MINUTES
DETERMINISTIC_BUDGET = 20.0
WALL_CLOCK_GUARD_FACTOR = 6.0


@dataclass(frozen=True)
class BreakRequest:
    """A break that has to be scheduled: its block, its length and its allowed start window."""

    worker_id: str
    block_index: int
    minutes: int
    low: int
    high: int
    nominal: int


@dataclass
class BreakSchedule:
    """Scheduled breaks with the status, the capacity gap and the peak of simultaneous breaks."""

    entries: list[BreakEntry]
    status: str
    capacity_gap_minutes: int
    peak_simultaneous: int
    notes: list[str] = field(default_factory=list)


def break_requests(shifts: Mapping[str, ShiftPattern], rules: LabourRules) -> list[BreakRequest]:
    """The breaks that the shifts of a day require."""
    requests: list[BreakRequest] = []
    for worker_id in sorted(shifts):
        for index, block in enumerate(shifts[worker_id].blocks):
            minutes = block_break_length(block, rules)
            window = break_window(block, rules) if minutes else None
            if minutes and window is not None:
                nominal = grid_down(block.register_start + block.register_minutes // 2)
                requests.append(
                    BreakRequest(worker_id, index, minutes, window[0], window[1], nominal)
                )
    return requests


def shift_span(shifts: Mapping[str, ShiftPattern]) -> tuple[int, int]:
    """First register start and last register end among a day's shifts."""
    starts = [block.register_start for pattern in shifts.values() for block in pattern.blocks]
    ends = [block.register_end for pattern in shifts.values() for block in pattern.blocks]
    return min(starts), max(ends)


def on_duty_counts(shifts: Mapping[str, ShiftPattern], times: range) -> dict[int, int]:
    """Number of workers on a register at each grid minute."""
    counts: dict[int, int] = {minute: 0 for minute in times}
    for pattern in shifts.values():
        for block in pattern.blocks:
            for minute in range(block.register_start, block.register_end, STEP):
                if minute in counts:
                    counts[minute] += 1
    return counts


def slot_lookup(
    requirement: RequirementTable | None, day: date, times: range
) -> dict[int, SlotRequirement | None]:
    """Requirement slot at each grid minute, or None."""
    if requirement is None:
        return {minute: None for minute in times}
    return {minute: requirement.at(day, minute) for minute in times}


def nominal_entries(requests: list[BreakRequest], day: date) -> list[BreakEntry]:
    """Breaks placed at the middle of their block, kept inside their window."""
    return [
        BreakEntry(
            request.worker_id,
            day,
            request.block_index,
            min(max(request.nominal, request.low), request.high),
            request.minutes,
        )
        for request in requests
    ]


def _entry_profile(entries: list[BreakEntry], times: range) -> dict[int, int]:
    counts: dict[int, int] = {minute: 0 for minute in times}
    for entry in entries:
        for minute in range(entry.start, entry.end, STEP):
            if minute in counts:
                counts[minute] += 1
    return counts


def schedule_breaks(
    day: date,
    shifts: Mapping[str, ShiftPattern],
    requirement: RequirementTable | None,
    config: PlannerConfig,
) -> BreakSchedule:
    """Choose the start of every break of a day with CP-SAT.

    Breaks stay inside their windows, few workers are on break at once, rush slots are
    avoided where possible and few breaks start at the same minute. If the solver fails,
    every break is placed at the middle of its block.
    """
    rules = config.rules
    daily = config.daily
    requests = break_requests(shifts, rules)
    if not requests:
        return BreakSchedule([], "NONE", 0, 0)
    hours = requirement.opening_hours(day) if requirement is not None else None
    open_minute, close_minute = hours if hours is not None else shift_span(shifts)
    times = range(open_minute, close_minute, STEP)
    duty = on_duty_counts(shifts, times)
    slots = slot_lookup(requirement, day, times)

    model = cp_model.CpModel()
    start_vars: dict[tuple[int, int], cp_model.IntVar] = {}
    covering: dict[int, list[cp_model.IntVar]] = defaultdict(list)
    starting: dict[int, list[cp_model.IntVar]] = defaultdict(list)
    costs: list[tuple[cp_model.IntVar, int]] = []
    for index, request in enumerate(requests):
        options: list[cp_model.IntVar] = []
        for start in range(request.low, request.high + 1, STEP):
            variable = model.new_bool_var(f"break_{index}_{start}")
            start_vars[(index, start)] = variable
            options.append(variable)
            starting[start].append(variable)
            cells = range(start, start + request.minutes, STEP)
            for minute in cells:
                covering[minute].append(variable)
            rush_cells = sum(1 for minute in cells if (slots.get(minute) or _CALM).rush)
            deviation = abs(start - request.nominal) // STEP
            costs.append(
                (
                    variable,
                    daily.break_deviation_weight * deviation + daily.break_rush_weight * rush_cells,
                )
            )
        model.add_exactly_one(options)

    gaps: dict[int, cp_model.IntVar] = {}
    peak = len(requests)
    for minute in times:
        slot = slots[minute]
        needed = slot.registers + slot.k_sc if slot is not None else 0
        simultaneous = sum(covering[minute]) if covering[minute] else 0
        gap = model.new_int_var(0, peak, f"gap_{minute}")
        model.add(duty[minute] - simultaneous + gap >= needed)
        gaps[minute] = gap
        costs.append((gap, daily.break_capacity_weight))
        over = model.new_int_var(0, peak, f"over_{minute}")
        model.add(simultaneous - rules.max_simultaneous_breaks <= over)
        costs.append((over, daily.break_limit_weight))
        beyond = model.new_int_var(0, peak, f"beyond_{minute}")
        model.add(simultaneous - rules.limit_case_simultaneous_breaks <= beyond)
        costs.append((beyond, daily.break_over_limit_weight))
    for start, variables in starting.items():
        if len(variables) > 1:
            same = model.new_int_var(0, len(variables), f"same_{start}")
            model.add(sum(variables) - 1 <= same)
            costs.append((same, daily.break_same_start_weight))
    model.minimize(cp_model.LinearExpr.weighted_sum([v for v, _ in costs], [w for _, w in costs]))

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = config.solver.random_seed
    solver.parameters.num_workers = 1
    solver.parameters.max_deterministic_time = DETERMINISTIC_BUDGET
    solver.parameters.max_time_in_seconds = daily.break_solver_seconds * WALL_CLOCK_GUARD_FACTOR
    status = solver.solve(model)
    name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        entries = nominal_entries(requests, day)
        profile = _entry_profile(entries, times)
        gap_minutes = sum(
            max(0, _needed(slots[minute]) - (duty[minute] - profile[minute])) * STEP
            for minute in times
        )
        return BreakSchedule(
            entries,
            "FALLBACK",
            gap_minutes,
            max(profile.values()),
            [f"break solver returned {name}; every break placed at the middle of its block"],
        )
    entries = []
    for (index, start), variable in sorted(start_vars.items()):
        if solver.boolean_value(variable):
            request = requests[index]
            entries.append(
                BreakEntry(request.worker_id, day, request.block_index, start, request.minutes)
            )
    entries.sort(key=lambda item: (item.worker_id, item.block_index))
    profile = _entry_profile(entries, times)
    gap_minutes = sum(solver.value(gap) * STEP for gap in gaps.values())
    return BreakSchedule(entries, name, gap_minutes, max(profile.values()))


def _needed(slot: SlotRequirement | None) -> int:
    return slot.registers + slot.k_sc if slot is not None else 0


_CALM = SlotRequirement(
    registers=0,
    k_sc=0,
    sco_zone1=0,
    sco_zone2=0,
    floaters=0,
    headcount=0,
    rush=False,
    offered_load=0.0,
    planned_customers=0.0,
)
