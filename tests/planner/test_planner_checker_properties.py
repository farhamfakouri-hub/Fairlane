from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from planner_factories import CONFIG, RULES, START

from ree.planner.checker.checks import check_roster
from ree.planner.domain.patterns import PatternSpec, full_pattern_set, solver_pattern_set
from ree.planner.domain.plan import Assignment, RosterPlan, WorkerRecord
from ree.planner.domain.shifts import Block, ShiftPattern, make_block

FULL = full_pattern_set(CONFIG, PatternSpec())
SOLVER = solver_pattern_set(CONFIG, PatternSpec())
MORNING = [p for p in FULL.weekday if p.kind == "continuous" and p.register_end < 17 * 60]
EVENING = [p for p in FULL.weekday if p.kind == "continuous" and p.register_end >= 21 * 60 + 30]
WEEKDAY_OFFSETS = [1, 2, 3, 4, 5, 6]
SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def valid_rosters(draw) -> RosterPlan:
    weeks = draw(st.integers(min_value=1, max_value=3))
    count = draw(st.integers(min_value=1, max_value=7))
    workers: dict[str, WorkerRecord] = {}
    assignments: list[Assignment] = []
    for number in range(count):
        worker_id = f"W{number + 1:03d}"
        pool = draw(st.sampled_from([MORNING, EVENING]))
        chosen = draw(st.sampled_from(pool))
        offsets = draw(
            st.lists(st.sampled_from(WEEKDAY_OFFSETS), min_size=2, max_size=5, unique=True)
        )
        for week in range(weeks):
            for offset in offsets:
                assignments.append(
                    Assignment(worker_id, START + timedelta(days=7 * week + offset), chosen)
                )
        contract = len(offsets) * chosen.paid_minutes / 60.0
        workers[worker_id] = WorkerRecord(worker_id=worker_id, contract_hours=contract)
    return RosterPlan(START, weeks, workers, assignments)


@given(valid_rosters())
@SETTINGS
def test_randomly_built_valid_rosters_pass_the_checker(plan: RosterPlan) -> None:
    report = check_roster(plan, RULES)
    assert report.ok, report.summary()


def worked_days(plan: RosterPlan, worker_id: str) -> list:
    return sorted(a.day for a in plan.assignments if a.worker_id == worker_id)


def with_assignments(plan: RosterPlan, assignments: list[Assignment]) -> RosterPlan:
    return RosterPlan(plan.start_date, plan.weeks, dict(plan.workers), assignments)


@given(valid_rosters(), st.data())
@SETTINGS
def test_a_second_pattern_on_a_worked_day_is_rejected(plan: RosterPlan, data) -> None:
    chosen = data.draw(st.sampled_from(plan.assignments))
    other = data.draw(st.sampled_from(MORNING + EVENING))
    broken = with_assignments(
        plan, [*plan.assignments, Assignment(chosen.worker_id, chosen.day, other)]
    )
    assert "TWO_PATTERNS_SAME_DAY" in check_roster(broken, RULES).codes("error")


@given(valid_rosters(), st.data(), st.integers(min_value=1, max_value=9))
@SETTINGS
def test_broken_margins_are_rejected(plan: RosterPlan, data, shift: int) -> None:
    chosen = data.draw(st.sampled_from(plan.assignments))
    block = chosen.pattern.blocks[0]
    moved = dataclasses.replace(block, paid_start=block.paid_start + shift)
    pattern = ShiftPattern("moved", chosen.pattern.kind, (moved, *chosen.pattern.blocks[1:]))
    kept = [a for a in plan.assignments if a is not chosen]
    broken = with_assignments(plan, [*kept, Assignment(chosen.worker_id, chosen.day, pattern)])
    assert "PATTERN_MARGINS" in check_roster(broken, RULES).codes("error")


@given(valid_rosters(), st.data())
@SETTINGS
def test_assigning_a_blocked_worker_is_rejected(plan: RosterPlan, data) -> None:
    chosen = data.draw(st.sampled_from(plan.assignments))
    worker = plan.workers[chosen.worker_id]
    blocked = dataclasses.replace(worker, blocked={chosen.day: ((0, 1440),)})
    workers = {**plan.workers, chosen.worker_id: blocked}
    broken = RosterPlan(plan.start_date, plan.weeks, workers, list(plan.assignments))
    assert "AVAILABILITY" in check_roster(broken, RULES).codes("error")


@given(valid_rosters(), st.data())
@SETTINGS
def test_an_extra_shift_breaks_the_contract_hours(plan: RosterPlan, data) -> None:
    worker_id = data.draw(st.sampled_from(sorted(plan.workers)))
    taken = set(worked_days(plan, worker_id))
    free = [d for d in plan.dates() if d not in taken and d.weekday() != 6]
    assume(free)
    day = data.draw(st.sampled_from(free))
    pattern = data.draw(st.sampled_from(MORNING + EVENING))
    broken = with_assignments(plan, [*plan.assignments, Assignment(worker_id, day, pattern)])
    assert "WEEKLY_HOURS_CONTRACT" in check_roster(broken, RULES).codes("error")


@given(valid_rosters(), st.data())
@SETTINGS
def test_a_closing_shift_followed_by_an_early_opening_is_rejected(plan: RosterPlan, data) -> None:
    worker_id = data.draw(st.sampled_from(sorted(plan.workers)))
    days = worked_days(plan, worker_id)
    pairs = [(d, d + timedelta(days=1)) for d in days if d + timedelta(days=1) in days]
    assume(pairs)
    first, second = data.draw(st.sampled_from(pairs))
    closing = next(p for p in EVENING if p.register_end == 21 * 60 + 30)
    opening = next(p for p in MORNING if p.register_start == 8 * 60 + 30)
    kept = [
        a for a in plan.assignments if not (a.worker_id == worker_id and a.day in (first, second))
    ]
    broken = with_assignments(
        plan, [*kept, Assignment(worker_id, first, closing), Assignment(worker_id, second, opening)]
    )
    assert "MIN_REST" in check_roster(broken, RULES).codes("error")


@given(valid_rosters(), st.data())
@SETTINGS
def test_seven_days_in_a_row_are_rejected(plan: RosterPlan, data) -> None:
    worker_id = data.draw(st.sampled_from(sorted(plan.workers)))
    pattern = next(a.pattern for a in plan.assignments if a.worker_id == worker_id)
    kept = [a for a in plan.assignments if a.worker_id != worker_id]
    week = data.draw(st.integers(min_value=0, max_value=plan.weeks - 1))
    days = [START + timedelta(days=7 * week + offset) for offset in range(7)]
    broken = with_assignments(plan, [*kept, *(Assignment(worker_id, d, pattern) for d in days)])
    codes = check_roster(broken, RULES).codes("error")
    assert "MAX_CONSECUTIVE_DAYS" in codes


def test_the_solver_pool_is_a_subset_of_the_valid_patterns() -> None:
    for pattern in (*SOLVER.weekday, *SOLVER.sunday):
        assert check_roster(
            RosterPlan(
                START,
                1,
                {"A": WorkerRecord("A", pattern.paid_minutes / 60.0)},
                [
                    Assignment(
                        "A", START + timedelta(days=(6 if pattern in SOLVER.sunday else 1)), pattern
                    )
                ],
            ),
            RULES,
        ).ok


def test_a_pattern_built_by_hand_with_the_right_margins_is_accepted() -> None:
    block: Block = make_block(570, 930, 10, 5)
    plan = RosterPlan(
        START,
        1,
        {"A": WorkerRecord("A", block.paid_minutes / 60.0)},
        [Assignment("A", START + timedelta(days=1), ShiftPattern("hand", "continuous", (block,)))],
    )
    assert check_roster(plan, RULES).ok
    with pytest.raises(KeyError):
        RosterPlan(START, 1, {}, []).workers["missing"]
